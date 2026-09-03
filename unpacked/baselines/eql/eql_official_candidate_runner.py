#!/usr/bin/env python3
"""Thin candidate runner around the bundled official EQL-Div Theano core.

No network, loss, optimizer, regularization, division, penalty, or active-unit
logic is reimplemented here. Those operations are executed by the unchanged
upstream ``official_eql/src/mlfg_final.py`` code.

For the system-identification adapter, this wrapper evaluates both upstream
checkpoints returned by one training run (``best_state`` and ``final_state``)
on the same external validation set after restoring the physical output scale.
The checkpoint with the lower physical-scale validation MSE is exported. This
changes only checkpoint selection outside the upstream optimizer; it does not
modify the official EQL training equations or schedule.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _metrics(y, yh):
    """Framework-compatible metrics on one output scale.

    The normalized MAE exactly mirrors MATLAB ``compute_regression_metrics``:
    compute one scale per output from sample standard deviation (N-1), then
    fall back to range, RMS magnitude, and finally one.  This avoids mixing the
    old EQL-specific MAE/mean(|y|) denominator with the common baseline metric.
    """
    y = np.asarray(y, dtype=float)
    yh = np.asarray(yh, dtype=float)
    if y.size == 0 or yh.size == 0:
        return {k: None for k in (
            "mse", "rmse", "mae", "nrmse", "nrmseRange", "nmae",
            "scaleByOutput", "rmseByOutput", "maeByOutput"
        )}
    e = yh - y
    mse = float(np.mean(e * e))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))
    mse_by_output = np.mean(e * e, axis=0)
    rmse_by_output = np.sqrt(mse_by_output)
    mae_by_output = np.mean(np.abs(e), axis=0)
    ddof = 1 if y.shape[0] > 1 else 0
    scale = np.std(y, axis=0, ddof=ddof)
    range_scale = np.max(y, axis=0) - np.min(y, axis=0)
    rms_scale = np.sqrt(np.mean(y * y, axis=0))
    bad = (~np.isfinite(scale)) | (scale < 1e-12)
    scale = np.where(bad, range_scale, scale)
    bad = (~np.isfinite(scale)) | (scale < 1e-12)
    scale = np.where(bad, rms_scale, scale)
    bad = (~np.isfinite(scale)) | (scale < 1e-12)
    scale = np.where(bad, 1.0, scale)
    nrmse_by_output = rmse_by_output / scale
    nmae_by_output = mae_by_output / scale
    yr = float(np.max(y) - np.min(y))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": float(np.mean(nrmse_by_output)),
        "nrmseRange": rmse / yr if yr > 0 else None,
        "nmae": float(np.mean(nmae_by_output)),
        "scaleByOutput": scale.tolist(),
        "rmseByOutput": rmse_by_output.tolist(),
        "maeByOutput": mae_by_output.tolist(),
    }


def _state_counts(state, threshold=1e-3):
    parameter_count = 0
    active_weights = 0
    active_biases = 0
    for layer_state in state:
        if len(layer_state) < 2:
            continue
        w = np.asarray(layer_state[0])
        b = np.asarray(layer_state[1])
        parameter_count += int(w.size + b.size)
        active_weights += int(np.sum(np.abs(w) >= threshold))
        active_biases += int(np.sum(np.abs(b) >= threshold))
    return parameter_count, active_weights, active_biases


def _load_dataset(path):
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def _evaluate_checkpoint(classifier, state, datasets, physical_y_scale):
    """Evaluate one upstream state on internal and original output scales."""
    classifier.set_state(state)
    (xtr, ytr), (xv, yv), (xt, yt) = datasets
    yhat_tr = np.asarray(classifier.evaluate(xtr), dtype=float)
    yhat_v = np.asarray(classifier.evaluate(xv), dtype=float)
    yhat_t = np.asarray(classifier.evaluate(xt), dtype=float)
    parameter_count, active_w, active_b = _state_counts(state)
    num_active = int(classifier.get_num_active_units())

    scale = np.asarray(physical_y_scale, dtype=float).reshape(1, -1)
    if scale.shape[1] != yv.shape[1]:
        raise ValueError("physical_y_scale dimension does not match EQL outputs.")

    return {
        "num_active": num_active,
        "parameter_count": parameter_count,
        "active_weight_count": active_w,
        "active_bias_count": active_b,
        "active_parameter_count": active_w + active_b,
        "train_metrics": _metrics(ytr, yhat_tr),
        "val_metrics": _metrics(yv, yhat_v),
        "test_metrics": _metrics(yt, yhat_t),
        "framework_train_metrics": _metrics(ytr * scale, yhat_tr * scale),
        "framework_val_metrics": _metrics(yv * scale, yhat_v * scale),
        "framework_test_metrics": _metrics(yt * scale, yhat_t * scale),
    }


def _finite_mse(record):
    try:
        value = float(record["framework_val_metrics"]["mse"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _select_checkpoint(mode, best_eval, final_eval):
    mode = str(mode or "final_state").strip().lower()
    records = {
        "best_state": best_eval,
        "final_state": final_eval,
    }
    if mode in ("physical_validation_mse", "physical_val_mse", "validation_mse_physical_scale"):
        # Accuracy first. Active-parameter count is only a deterministic tie
        # breaker; best_state is preferred when all reported quantities tie.
        name = min(
            records,
            key=lambda key: (
                _finite_mse(records[key]),
                int(records[key].get("active_parameter_count", 10**12)),
                0 if key == "best_state" else 1,
            ),
        )
        return name, "external_validation_mse_physical_scale"
    if mode in ("best_state", "upstream_best_state"):
        return "best_state", "forced_upstream_best_state"
    if mode in ("final_state", "upstream_final_state", "final"):
        return "final_state", "forced_upstream_final_state"
    raise ValueError("Unsupported checkpoint_selection_mode: %s" % mode)


def _dump_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path.resolve())


def _load_state(path):
    path = Path(path)
    with path.open("rb") as fh:
        return pickle.load(fh, encoding="latin1")



def _infer_stored_best_state_epoch(validation_errors, improvement_threshold=0.99):
    """Recover the epoch of the actual upstream ``best_state`` snapshot.

    The official core updates ``best_epoch`` for every new validation minimum,
    but updates ``best_state`` only when that minimum improves by at least the
    upstream ``improvement_threshold``.  Replaying that exact rule over the
    returned validation history identifies the epoch of the state that was
    actually serialized, without modifying the upstream training code.
    """
    history = np.asarray(validation_errors, dtype=float)
    if history.size == 0:
        return 0
    history = np.atleast_2d(history)
    running_best = float("inf")
    stored_epoch = 0
    for row in history:
        if row.size < 2:
            continue
        epoch, value = int(row[0]), float(row[1])
        if not np.isfinite(value):
            continue
        if value < running_best:
            if value < running_best * float(improvement_threshold):
                stored_epoch = epoch
            running_best = value
    return int(stored_epoch)

def _checkpoint_phase(epoch, reg_start, reg_end):
    epoch = int(epoch)
    if epoch <= 0:
        return "initialization"
    if epoch <= int(reg_start):
        return "pre_regularization"
    if epoch <= int(reg_end):
        return "l1_regularization"
    return "fixed_L0_post_regularization"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result_path = Path(cfg["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = {
        "status": "failed",
        "candidate_id": int(cfg["candidate_id"]),
        "paper_depth_L": int(cfg["paper_depth_L"]),
        "official_hidden_layers": int(cfg["official_hidden_layers"]),
        "lambda_l1": float(cfg["lambda_l1"]),
    }
    try:
        # Compatibility alias only; it does not alter EQL mathematics.
        if not hasattr(np, "asscalar"):
            np.asscalar = lambda a: np.asarray(a).item()  # type: ignore[attr-defined]

        theano_flags = str(cfg.get("theano_flags", "")).strip()
        candidate_compiledir = str(cfg.get("candidate_compiledir", "")).strip()
        if candidate_compiledir:
            compiledir_path = Path(candidate_compiledir).resolve()
            compiledir_path.mkdir(parents=True, exist_ok=True)
            # IMPORTANT (Windows/Theano 1.0): use the final ``compiledir``
            # rather than ``base_compiledir`` to avoid legacy path-length issues.
            lowered = theano_flags.lower()
            if "compiledir=" not in lowered and "base_compiledir=" not in lowered:
                private_flag = "compiledir=%s" % compiledir_path.as_posix()
                theano_flags = private_flag if not theano_flags else theano_flags + "," + private_flag
        if theano_flags:
            os.environ["THEANO_FLAGS"] = theano_flags

        src_root = Path(cfg["official_src_root"]).resolve()
        if not (src_root / "mlfg_final.py").is_file():
            raise FileNotFoundError("Official EQL mlfg_final.py not found: %s" % src_root)
        sys.path.insert(0, str(src_root))
        import mlfg_final as official_mlfg  # noqa: E402

        candidate_id = int(cfg["candidate_id"])
        random.seed(candidate_id)
        np.random.seed(candidate_id)

        datasets = _load_dataset(cfg["dataset_path"])
        n_epochs = int(cfg["epochs"])
        reg_start = int(cfg["reg_start"])
        reg_end = int(cfg["reg_end"])
        init_state_path = str(cfg.get("init_state_path", "") or "").strip()
        init_state = _load_state(init_state_path) if init_state_path else None
        trained = official_mlfg.test_mlfg(
            datasets=datasets,
            learning_rate=float(cfg["learning_rate"]),
            L1_reg=float(cfg["lambda_l1"]),
            L2_reg=float(cfg.get("lambda_l2", 0.0)),
            n_epochs=n_epochs,
            batch_size=int(cfg["batch_size"]),
            n_layer=int(cfg["official_hidden_layers"]),
            n_per_base=int(cfg["units_per_type"]),
            basefuncs1=[0, 1, 2],
            basefuncs2=[0],
            with_shortcuts=False,
            id=candidate_id,
            classifier=None,
            gradient=str(cfg["gradient"]),
            init_state=init_state,
            verbose=bool(cfg.get("verbose", False)),
            reg_start=reg_start,
            reg_end=reg_end,
            validate_every=int(cfg.get("validate_every", 10)),
            k=int(cfg.get("penalty_every", 50)),
        )
        classifier = trained["classifier"]
        final_state = classifier.get_state()
        best_state = trained["best_state"]
        physical_y_scale = np.asarray(
            cfg.get("physical_y_scale", np.ones((1, datasets[1][1].shape[1]))), dtype=float
        ).reshape(1, -1)

        final_eval = _evaluate_checkpoint(classifier, final_state, datasets, physical_y_scale)
        best_eval = _evaluate_checkpoint(classifier, best_state, datasets, physical_y_scale)
        selected_checkpoint, checkpoint_metric = _select_checkpoint(
            cfg.get("checkpoint_selection_mode", "final_state"), best_eval, final_eval
        )
        selected_state = best_state if selected_checkpoint == "best_state" else final_state
        selected_eval = best_eval if selected_checkpoint == "best_state" else final_eval
        classifier.set_state(selected_state)
        upstream_best_validation_epoch = int(trained.get("best_epoch", 0))
        best_epoch = _infer_stored_best_state_epoch(trained.get("val_errors", []), 0.99)
        final_epoch = int(n_epochs)
        selected_epoch = best_epoch if selected_checkpoint == "best_state" else final_epoch
        best_phase = _checkpoint_phase(best_epoch, reg_start, reg_end)
        final_phase = _checkpoint_phase(final_epoch, reg_start, reg_end)
        selected_phase = best_phase if selected_checkpoint == "best_state" else final_phase

        selected_state_path = _dump_state(cfg["state_path"], selected_state)
        final_state_path = _dump_state(
            cfg.get("final_state_path", str(Path(cfg["state_path"]).with_name("final_state.pkl"))),
            final_state,
        )
        best_state_path = _dump_state(
            cfg.get("best_state_path", str(Path(cfg["state_path"]).with_name("best_state.pkl"))),
            best_state,
        )

        result.update(
            status="ok",
            time_seconds=float(time.perf_counter() - t0),
            official_runtime_minutes=float(trained["runtime"]),
            epochs=n_epochs,
            reg_start=reg_start,
            reg_end=reg_end,
            gradient=str(cfg["gradient"]),
            learning_rate=float(cfg["learning_rate"]),
            lambda_l2=float(cfg.get("lambda_l2", 0.0)),
            candidate_source=str(cfg.get("candidate_source", "scratch_full_sweep")),
            restart_index=int(cfg.get("restart_index", 0)),
            warm_start_used=bool(init_state_path),
            init_state_path=str(Path(init_state_path).resolve()) if init_state_path else "",
            batch_size=int(cfg["batch_size"]),
            units_per_type=int(cfg["units_per_type"]),
            # Upstream fields are retained verbatim for the reference Vint-S scan.
            official_num_active=int(final_eval["num_active"]),
            official_best_num_active=int(best_eval["num_active"]),
            official_best_epoch=upstream_best_validation_epoch,
            upstream_best_validation_epoch=upstream_best_validation_epoch,
            actual_best_state_epoch=best_epoch,
            official_final_val_mse=float(trained["val_score"]),
            official_best_val_mse=float(trained["best_val_score"]),
            official_test_mse=float(trained["test_score"]),
            # Framework-exported checkpoint fields use original output units.
            selected_checkpoint=selected_checkpoint,
            checkpoint_selection_metric=checkpoint_metric,
            selected_checkpoint_epoch=selected_epoch,
            selected_checkpoint_phase=selected_phase,
            best_state_epoch=best_epoch,
            best_state_phase=best_phase,
            final_state_epoch=final_epoch,
            final_state_phase=final_phase,
            selected_num_active=int(selected_eval["num_active"]),
            parameter_count=int(selected_eval["parameter_count"]),
            active_weight_count=int(selected_eval["active_weight_count"]),
            active_bias_count=int(selected_eval["active_bias_count"]),
            active_parameter_count=int(selected_eval["active_parameter_count"]),
            train_metrics=selected_eval["train_metrics"],
            val_metrics=selected_eval["val_metrics"],
            test_metrics=selected_eval["test_metrics"],
            framework_train_metrics=selected_eval["framework_train_metrics"],
            framework_val_metrics=selected_eval["framework_val_metrics"],
            framework_test_metrics=selected_eval["framework_test_metrics"],
            framework_val_mse=float(selected_eval["framework_val_metrics"]["mse"]),
            best_state_framework_val_mse=float(best_eval["framework_val_metrics"]["mse"]),
            final_state_framework_val_mse=float(final_eval["framework_val_metrics"]["mse"]),
            best_state_active_parameter_count=int(best_eval["active_parameter_count"]),
            final_state_active_parameter_count=int(final_eval["active_parameter_count"]),
            best_state_metrics=best_eval,
            final_state_metrics=final_eval,
            state_path=selected_state_path,
            selected_state_path=selected_state_path,
            best_state_path=best_state_path,
            final_state_path=final_state_path,
        )
    except Exception as exc:
        result.update(
            time_seconds=float(time.perf_counter() - t0),
            error="%s: %s" % (type(exc).__name__, exc),
            traceback=traceback.format_exc(limit=30),
        )
    result_path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
