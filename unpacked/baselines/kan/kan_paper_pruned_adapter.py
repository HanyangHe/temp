#!/usr/bin/env python3
"""Pruned-KAN Feynman sweep using the bundled official pykan implementation.

The training order follows pyKAN's own ``kan/experiment.py::runner1`` helper
and the Feynman sweep ranges stated in Appendix P:

1. fix width=5 and sweep KAN depths 2..6;
2. initialize at G=3 and perform exactly one sparse LBFGS fit with
   lambda in {1e-2,1e-3};
3. immediately call the official ``model.prune()`` once;
4. carry the pruned model through G={5,10,20,50,100,200} using only the
   official ``model.refine()`` and one ordinary ``lamb=0`` LBFGS fit per grid;
5. after every grid, evaluate the framework validation split and stop at the
   first meaningful increase, restoring the previous validation-best grid;
6. reserve ID-test/OOD for final evaluation only.

Only the framework-supplied samples and data split replace pyKAN's generated
Feynman samples. No warm-up fit, repeated sparsification, post-prune recovery,
or custom KAN implementation is added. The only compute-saving overlay is
validation-based grid early stopping, motivated by the U-shaped test-error
curve reported for grid extension in the KAN paper.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from kan_paper_adapter_common import (
    add_pykan_path,
    fit_standardizer,
    json_safe,
    metrics,
    read_csv_matrix,
    seed_all,
    standardize,
    write_csv_matrix,
    write_json,
)


class _QuietTqdm:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, *_args, **_kwargs):
        return None


def _predict(model, x, torch, y_mu, y_sd):
    # This baseline never symbolifies an edge. Temporarily disabling the empty
    # symbolic front for evaluation is numerically identical and avoids paying
    # symbolic-dispatch overhead outside model.fit(). Training itself remains
    # the untouched official pyKAN fit/refine/prune path.
    old_symbolic_enabled = bool(getattr(model, "symbolic_enabled", False))
    model.symbolic_enabled = False
    try:
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=model.device)
            yp = model(xt).detach().cpu().numpy()
    finally:
        model.symbolic_enabled = old_symbolic_enabled
    return yp * y_sd + y_mu


def _shape_json(width):
    out = []
    for value in width:
        if isinstance(value, (list, tuple)):
            out.append([int(x) for x in value])
        else:
            try:
                if hasattr(value, "tolist"):
                    converted = value.tolist()
                    out.append(converted if isinstance(converted, list) else int(converted))
                else:
                    out.append(int(value))
            except Exception:
                out.append(str(value))
    return out


def _hidden_node_count(width) -> int:
    total = 0
    for value in list(width)[1:-1]:
        if isinstance(value, (list, tuple)):
            total += sum(int(x) for x in value)
        elif hasattr(value, "tolist"):
            converted = value.tolist()
            total += sum(int(x) for x in converted) if isinstance(converted, list) else int(converted)
        else:
            total += int(value)
    return total


def _clone_model(model, KAN):
    """Clone a numerical pyKAN checkpoint without changing official source code.

    ``copy.deepcopy(model)`` is not reliable after a forward pass because
    PyTorch caches non-leaf tensors. Reconstructing the same official KAN and
    loading a copied state_dict preserves the grid, pruned topology, masks,
    coefficients, and affine parameters while avoiding pyKAN auto-save files.
    No symbolic edge is fitted in this baseline.
    """
    clone = KAN(
        width=copy.deepcopy(model.width),
        grid=copy.deepcopy(model.grid),
        k=copy.deepcopy(model.k),
        mult_arity=copy.deepcopy(model.mult_arity),
        base_fun=model.base_fun_name,
        symbolic_enabled=bool(model.symbolic_enabled),
        affine_trainable=bool(model.affine_trainable),
        grid_eps=float(model.grid_eps),
        grid_range=copy.deepcopy(model.grid_range),
        sp_trainable=bool(model.sp_trainable),
        sb_trainable=bool(model.sb_trainable),
        auto_save=False,
        first_init=False,
        device=model.device,
    )
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    clone.auto_save = False
    return clone


def _active_size(model):
    """Count effective numerical KAN edge parameters after official masking."""
    active_edges = 0
    active_coeff = 0
    numeric_total = 0
    for layer in model.act_fun:
        mask = layer.mask.detach().cpu().numpy() > 0.5
        n_active = int(mask.sum())
        n_edges = int(mask.size)
        active_edges += n_active
        per_edge = int(layer.coef.shape[-1]) if bool(layer.coef.requires_grad) else 0
        if bool(layer.scale_base.requires_grad):
            per_edge += 1
        if bool(layer.scale_sp.requires_grad):
            per_edge += 1
        active_coeff += n_active * per_edge
        numeric_total += n_edges * per_edge
    return active_edges, active_coeff, numeric_total


def _assert_finite_metrics(metric_dict, stage: str) -> None:
    mse = float(metric_dict.get("mse", float("nan")))
    if not np.isfinite(mse):
        raise FloatingPointError(f"Non-finite validation MSE after {stage}.")


def _fit_official(model, dataset, *, opt: str, steps: int, lamb: float, lr: float):
    """Make exactly one direct call to the official ``MultKAN.fit`` API."""
    return model.fit(
        dataset,
        opt=str(opt),
        steps=max(int(steps), 1),
        log=max(int(steps), 1) + 1,
        lamb=float(lamb),
        lr=float(lr),
        batch=-1,
    )


def _stage_record(
    *,
    phase: str,
    grid: int,
    status: str,
    time_seconds: float,
    val_mse: float,
    accepted: bool,
    relative_improvement: float = float("nan"),
    stop_reason: str = "",
    error: str = "",
):
    # Identical fields keep MATLAB jsondecode on a regular struct array.
    return dict(
        phase=str(phase),
        grid=int(grid),
        status=str(status),
        time_seconds=float(time_seconds),
        val_mse=float(val_mse),
        val_rmse=float(np.sqrt(val_mse)) if np.isfinite(val_mse) and val_mse >= 0 else float("nan"),
        relative_improvement=float(relative_improvement),
        accepted=bool(accepted),
        stop_reason=str(stop_reason),
        error=str(error),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-pid")
    parser.add_argument("--control-file")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    total_start = time.perf_counter()
    paths = cfg["paths"]
    outdir = Path(cfg["work_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    Xtr = read_csv_matrix(paths["X_train"])
    Ytr = read_csv_matrix(paths["Y_train"])
    Xv = read_csv_matrix(paths["X_val"])
    Yv = read_csv_matrix(paths["Y_val"])
    Xt = read_csv_matrix(paths["X_test"])
    Yt = read_csv_matrix(paths["Y_test"])
    Xo = read_csv_matrix(paths["X_ood"])
    Yo = read_csv_matrix(paths["Y_ood"])

    if cfg.get("normalize_inputs", False):
        xmu, xsd = fit_standardizer(Xtr)
    else:
        xmu = np.zeros((1, Xtr.shape[1]))
        xsd = np.ones((1, Xtr.shape[1]))
    if cfg.get("normalize_outputs", False):
        ymu, ysd = fit_standardizer(Ytr)
    else:
        ymu = np.zeros((1, Ytr.shape[1]))
        ysd = np.ones((1, Ytr.shape[1]))

    Xtrn = standardize(Xtr, xmu, xsd)
    Xvn = standardize(Xv, xmu, xsd)
    Xtn = standardize(Xt, xmu, xsd)
    Xon = standardize(Xo, xmu, xsd) if Xo.size else Xo
    Ytrn = standardize(Ytr, ymu, ysd)
    Yvn = standardize(Yv, ymu, ysd)

    add_pykan_path(cfg["pykan_root"])
    import torch

    torch.set_default_dtype(
        torch.float64 if str(cfg.get("dtype", "float32")).lower() == "float64" else torch.float32
    )
    torch_threads = int(cfg.get("torch_num_threads", 0) or 0)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    multkan_module = importlib.import_module("kan.MultKAN")
    multkan_module.tqdm = _QuietTqdm
    KAN = multkan_module.KAN

    device = str(cfg.get("device", "cpu"))
    seed = int(cfg.get("seed", 1))
    fixed_width = int(cfg.get("width", 5))
    depths = [int(v) for v in cfg.get("depth_list", [2, 3, 4, 5, 6])]
    grids = [int(v) for v in cfg.get("grid_list", [3, 5, 10, 20, 50, 100, 200])]
    if not grids or any(v <= 0 for v in grids):
        raise ValueError("grid_list must contain positive grid sizes.")
    if any(grids[i] >= grids[i + 1] for i in range(len(grids) - 1)):
        raise ValueError("grid_list must be strictly increasing in the paper refinement order.")
    lambdas = [float(v) for v in cfg.get("sparsification_lambda_list", [1e-2, 1e-3])]
    spline_order = int(cfg.get("spline_order", 3))
    steps = int(cfg.get("steps_per_grid", 200))
    optimizer = str(cfg.get("optimizer", "LBFGS"))
    learning_rate = float(cfg.get("learning_rate", 1.0))
    node_th = float(cfg.get("prune_node_threshold", 1e-2))
    edge_th = float(cfg.get("prune_edge_threshold", 3e-2))
    grid_early_stop = bool(cfg.get("grid_early_stop", True))
    grid_early_stop_patience = max(int(cfg.get("grid_early_stop_patience", 1)), 1)
    grid_early_stop_relative_tolerance = max(
        float(cfg.get("grid_early_stop_relative_tolerance", 0.0)), 0.0
    )

    dataset = {
        "train_input": torch.as_tensor(Xtrn, dtype=torch.get_default_dtype(), device=device),
        "train_label": torch.as_tensor(Ytrn, dtype=torch.get_default_dtype(), device=device),
        # The framework validation split occupies pykan's held-out test slot
        # during fitting; final ID-test and OOD remain untouched.
        "test_input": torch.as_tensor(Xvn, dtype=torch.get_default_dtype(), device=device),
        "test_label": torch.as_tensor(Yvn, dtype=torch.get_default_dtype(), device=device),
    }

    nan_metrics = {name: float("nan") for name in ("mse", "rmse", "mae", "nrmse", "nrmseRange", "nmae")}
    candidates = []
    selected = None
    selected_model = None

    for lambda_index, sparsification_lambda in enumerate(lambdas):
        for depth in depths:
            shape = [Xtr.shape[1]] + [fixed_width] * max(depth - 1, 0) + [Ytr.shape[1]]
            candidate = dict(
                depth=depth,
                hidden_layer_count=max(depth - 1, 0),
                width=fixed_width,
                initial_shape=list(shape),
                sparsification_lambda=sparsification_lambda,
                seed=seed,
                status="failed",
                failed_stage="not_started",
                last_successful_stage="none",
                grid_stages=[],
                time_seconds=float("nan"),
                final_grid=None,
                last_attempted_grid=None,
                best_grid_val_mse=float("nan"),
                grid_stop_reason="",
                pruned_shape=[],
                final_shape=[],
                active_edge_count=None,
                active_coefficient_count=None,
                trainable_parameter_count=None,
                pre_prune_train_metrics=dict(nan_metrics),
                pre_prune_val_metrics=dict(nan_metrics),
                immediate_post_prune_train_metrics=dict(nan_metrics),
                immediate_post_prune_val_metrics=dict(nan_metrics),
                post_refit_train_metrics=dict(nan_metrics),
                post_refit_val_metrics=dict(nan_metrics),
                train_metrics=dict(nan_metrics),
                val_metrics=dict(nan_metrics),
                error="",
                traceback="",
            )
            candidate_start = time.perf_counter()
            model = None
            stage = f"initialize_grid_{grids[0]}"

            try:
                seed_all(seed, torch)
                checkpoint_dir = outdir / f"ckpt_depth{depth}_lambda{lambda_index}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                # Leave constructor defaults intact (including the empty
                # symbolic front). Official model.fit() automatically disables
                # an unused symbolic front; no custom model.speed() call is
                # needed or allowed during sparse training.
                model = KAN(
                    width=list(shape),
                    grid=grids[0],
                    k=spline_order,
                    seed=seed,
                    device=device,
                    auto_save=False,
                    ckpt_path=str(checkpoint_dir),
                )

                # pyKAN experiment.py order: one sparse fit at the initial grid.
                initial_grid = grids[0]
                stage = f"sparse_fit_grid_{initial_grid}"
                stage_start = time.perf_counter()
                _fit_official(
                    model,
                    dataset,
                    opt=optimizer,
                    steps=steps,
                    lamb=sparsification_lambda,
                    lr=learning_rate,
                )
                pre_prune_train = metrics(Ytr, _predict(model, Xtrn, torch, ymu, ysd))
                pre_prune_val = metrics(Yv, _predict(model, Xvn, torch, ymu, ysd))
                _assert_finite_metrics(pre_prune_val, stage)
                candidate["pre_prune_train_metrics"] = pre_prune_train
                candidate["pre_prune_val_metrics"] = pre_prune_val
                candidate["grid_stages"].append(
                    _stage_record(
                        phase="initial_sparse_fit",
                        grid=initial_grid,
                        status="ok",
                        time_seconds=time.perf_counter() - stage_start,
                        val_mse=pre_prune_val["mse"],
                        accepted=True,
                    )
                )
                candidate["last_successful_stage"] = stage

                # Exactly one official prune, immediately after the sparse fit,
                # matching kan/experiment.py::runner1. No extra fit is inserted.
                stage = "official_prune_after_initial_sparse_fit"
                stage_start = time.perf_counter()
                model = model.prune(node_th=node_th, edge_th=edge_th)
                model.auto_save = False
                if _hidden_node_count(model.width) <= 0:
                    raise RuntimeError("Official pruning removed every hidden node.")
                immediate_post_train = metrics(Ytr, _predict(model, Xtrn, torch, ymu, ysd))
                immediate_post_val = metrics(Yv, _predict(model, Xvn, torch, ymu, ysd))
                _assert_finite_metrics(immediate_post_val, stage)
                candidate["immediate_post_prune_train_metrics"] = immediate_post_train
                candidate["immediate_post_prune_val_metrics"] = immediate_post_val
                candidate["pruned_shape"] = _shape_json(model.width)
                candidate["grid_stages"].append(
                    _stage_record(
                        phase="official_prune",
                        grid=initial_grid,
                        status="ok",
                        time_seconds=time.perf_counter() - stage_start,
                        val_mse=immediate_post_val["mse"],
                        accepted=True,
                    )
                )
                candidate["last_successful_stage"] = stage

                # The post-prune G=3 model is the first eligible checkpoint.
                # Test/OOD are never inspected here: pyKAN's held-out ``test``
                # slot contains the framework validation split.
                best_model = _clone_model(model, KAN)
                best_grid = int(initial_grid)
                best_val_metrics = dict(immediate_post_val)
                best_train_metrics = dict(immediate_post_train)
                best_val_mse = float(immediate_post_val["mse"])
                bad_grid_count = 0
                last_attempted_grid = int(initial_grid)
                stop_reason = "grid_schedule_completed; selected_validation_best_checkpoint"

                # Official refinement path: each larger grid gets one lamb=0
                # fit. Stop when validation error rises and restore the prior
                # best grid. Comparing MSE or RMSE gives exactly the same order.
                for grid in grids[1:]:
                    stage = f"refine_to_grid_{grid}"
                    refine_start = time.perf_counter()
                    old_model = model
                    model = old_model.refine(grid)
                    model.auto_save = False
                    del old_model
                    gc.collect()
                    refine_seconds = time.perf_counter() - refine_start

                    stage = f"ordinary_fit_grid_{grid}"
                    fit_start = time.perf_counter()
                    _fit_official(
                        model,
                        dataset,
                        opt=optimizer,
                        steps=steps,
                        lamb=0.0,
                        lr=learning_rate,
                    )
                    last_attempted_grid = int(grid)
                    train_at_grid = metrics(Ytr, _predict(model, Xtrn, torch, ymu, ysd))
                    val_at_grid = metrics(Yv, _predict(model, Xvn, torch, ymu, ysd))
                    _assert_finite_metrics(val_at_grid, stage)
                    current_val_mse = float(val_at_grid["mse"])
                    denominator = max(abs(best_val_mse), np.finfo(float).tiny)
                    relative_improvement = (best_val_mse - current_val_mse) / denominator
                    is_new_best = current_val_mse < best_val_mse
                    meaningful_rise = current_val_mse > best_val_mse * (
                        1.0 + grid_early_stop_relative_tolerance
                    )

                    stage_status = "ok"
                    stage_stop_reason = ""
                    if is_new_best:
                        old_best_model = best_model
                        best_model = _clone_model(model, KAN)
                        best_grid = int(grid)
                        best_val_metrics = dict(val_at_grid)
                        best_train_metrics = dict(train_at_grid)
                        best_val_mse = current_val_mse
                        bad_grid_count = 0
                        del old_best_model
                        gc.collect()
                    elif meaningful_rise:
                        bad_grid_count += 1
                    else:
                        # Numerically flat within the configured tolerance.
                        bad_grid_count = 0

                    should_stop = (
                        grid_early_stop
                        and meaningful_rise
                        and bad_grid_count >= grid_early_stop_patience
                    )
                    if should_stop:
                        stage_status = "early_stop_trigger"
                        stage_stop_reason = (
                            f"validation_mse_increased_at_grid_{grid}; "
                            f"restore_best_grid_{best_grid}"
                        )

                    candidate["grid_stages"].append(
                        _stage_record(
                            phase="official_refine_and_ordinary_fit",
                            grid=grid,
                            status=stage_status,
                            time_seconds=refine_seconds + (time.perf_counter() - fit_start),
                            val_mse=current_val_mse,
                            relative_improvement=relative_improvement,
                            accepted=is_new_best,
                            stop_reason=stage_stop_reason,
                        )
                    )
                    candidate["last_successful_stage"] = stage

                    if should_stop:
                        stop_reason = stage_stop_reason
                        break

                # Discard the last attempted (possibly overfit) model and use
                # the saved validation-best checkpoint for candidate selection.
                if model is not best_model:
                    del model
                model = best_model
                best_model = None
                gc.collect()

                active_edges, active_coefficients, trainable_parameters = _active_size(model)
                if active_edges <= 0:
                    raise RuntimeError("Official pruning removed every numerical edge.")

                yp_train = _predict(model, Xtrn, torch, ymu, ysd)
                yp_val = _predict(model, Xvn, torch, ymu, ysd)
                final_train_metrics = metrics(Ytr, yp_train)
                final_val_metrics = metrics(Yv, yp_val)
                candidate["post_refit_train_metrics"] = final_train_metrics
                candidate["post_refit_val_metrics"] = final_val_metrics
                val_metrics = final_val_metrics
                _assert_finite_metrics(val_metrics, "validation_best_grid_selection_evaluation")

                candidate.update(
                    status="ok",
                    failed_stage="",
                    time_seconds=time.perf_counter() - candidate_start,
                    final_grid=int(best_grid),
                    last_attempted_grid=int(last_attempted_grid),
                    best_grid_val_mse=float(best_val_mse),
                    grid_stop_reason=stop_reason,
                    final_shape=_shape_json(model.width),
                    active_edge_count=active_edges,
                    active_coefficient_count=active_coefficients,
                    trainable_parameter_count=trainable_parameters,
                    train_metrics=final_train_metrics,
                    val_metrics=val_metrics,
                )

                selection_key = (val_metrics["mse"], active_coefficients, depth, lambda_index)
                if np.isfinite(selection_key[0]) and (selected is None or selection_key < selected[0]):
                    old_selected_model = selected_model
                    selected = (selection_key, candidate)
                    selected_model = model
                    if old_selected_model is not None and old_selected_model is not selected_model:
                        del old_selected_model
                        gc.collect()

            except Exception as error:
                candidate.update(
                    status="failed",
                    failed_stage=stage,
                    time_seconds=time.perf_counter() - candidate_start,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=20),
                )

            candidates.append(candidate)
            print(
                "[official pyKAN Feynman sweep] "
                f"depth={depth} width={fixed_width} lambda={sparsification_lambda:.3e} "
                f"status={candidate['status']} final_grid={candidate.get('final_grid')} "
                f"val_mse={candidate.get('val_metrics', {}).get('mse', float('nan')):.6e} "
                f"active={candidate.get('active_coefficient_count')} "
                f"failed_stage={candidate.get('failed_stage', '')} "
                f"error={candidate.get('error', '')} "
                f"time={candidate['time_seconds']:.3f}s",
                flush=True,
            )

            if model is not None and model is not selected_model:
                del model
            gc.collect()

    if selected is None or selected_model is None:
        failure_summary = "; ".join(
            f"depth={c['depth']},lambda={c['sparsification_lambda']:.3e},stage={c['failed_stage']},error={c['error']}"
            for c in candidates
        )
        raise RuntimeError(f"All pruned-KAN sweep candidates failed. {failure_summary}")

    selected_candidate = selected[1]
    yp_train = _predict(selected_model, Xtrn, torch, ymu, ysd)
    yp_val = _predict(selected_model, Xvn, torch, ymu, ysd)
    yp_test = _predict(selected_model, Xtn, torch, ymu, ysd)
    yp_ood = _predict(selected_model, Xon, torch, ymu, ysd) if Xo.size else np.empty((0, Ytr.shape[1]))

    write_csv_matrix(str(outdir / "Yhat_train.csv"), yp_train)
    write_csv_matrix(str(outdir / "Yhat_val.csv"), yp_val)
    write_csv_matrix(str(outdir / "Yhat_test.csv"), yp_test)
    write_csv_matrix(str(outdir / "Yhat_ood.csv"), yp_ood)

    best_by_lambda = []
    for sparsification_lambda in lambdas:
        ok_candidates = [
            c
            for c in candidates
            if c["status"] == "ok"
            and abs(c["sparsification_lambda"] - sparsification_lambda)
            <= max(1e-15, abs(sparsification_lambda) * 1e-12)
        ]
        if ok_candidates:
            best = min(
                ok_candidates,
                key=lambda item: (
                    item["val_metrics"]["mse"],
                    item["active_coefficient_count"],
                    item["depth"],
                ),
            )
            best_by_lambda.append(
                dict(
                    sparsification_lambda=sparsification_lambda,
                    depth=best["depth"],
                    selected_grid=best["final_grid"],
                    grid_stop_reason=best["grid_stop_reason"],
                    val_mse=best["val_metrics"]["mse"],
                    active_coefficient_count=best["active_coefficient_count"],
                    pruned_shape=best["pruned_shape"],
                )
            )

    result = dict(
        method="Official pyKAN Feynman pruned refinement sweep with validation grid early stop",
        protocol="official_pykan_feynman_pruned_refinement_validation_early_stop",
        selection_metric="validation_mse",
        total_time_seconds=time.perf_counter() - total_start,
        selected_candidate_time_seconds=selected_candidate["time_seconds"],
        selected_depth=selected_candidate["depth"],
        selected_hidden_layer_count=selected_candidate["hidden_layer_count"],
        selected_width=selected_candidate["width"],
        selected_sparsification_lambda=selected_candidate["sparsification_lambda"],
        selected_grid=selected_candidate["final_grid"],
        selected_last_attempted_grid=selected_candidate["last_attempted_grid"],
        selected_best_grid_val_mse=selected_candidate["best_grid_val_mse"],
        selected_grid_stop_reason=selected_candidate["grid_stop_reason"],
        selected_grid_stages=selected_candidate["grid_stages"],
        selected_initial_shape=selected_candidate["initial_shape"],
        selected_pruned_shape=selected_candidate["pruned_shape"],
        selected_final_shape=selected_candidate["final_shape"],
        active_edge_count=selected_candidate["active_edge_count"],
        active_coefficient_count=selected_candidate["active_coefficient_count"],
        trainable_parameter_count=selected_candidate["trainable_parameter_count"],
        selected_pre_prune_train_metrics=selected_candidate["pre_prune_train_metrics"],
        selected_pre_prune_val_metrics=selected_candidate["pre_prune_val_metrics"],
        selected_immediate_post_prune_train_metrics=selected_candidate["immediate_post_prune_train_metrics"],
        selected_immediate_post_prune_val_metrics=selected_candidate["immediate_post_prune_val_metrics"],
        selected_post_refit_train_metrics=selected_candidate["post_refit_train_metrics"],
        selected_post_refit_val_metrics=selected_candidate["post_refit_val_metrics"],
        train_metrics=metrics(Ytr, yp_train),
        val_metrics=metrics(Yv, yp_val),
        test_metrics=metrics(Yt, yp_test),
        ood_metrics=metrics(Yo, yp_ood),
        candidate_count=len(candidates),
        best_by_lambda=best_by_lambda,
        candidates=candidates,
        settings=dict(
            one_fit_per_grid=True,
            grid_schedule=grids,
            steps_per_grid=steps,
            sparse_fit_only_at_initial_grid=True,
            prune_once_after_initial_sparse_fit=True,
            refinement_fit_lambda=0.0,
            extra_post_prune_recovery_fit=False,
            validation_grid_early_stop=grid_early_stop,
            validation_grid_early_stop_patience=grid_early_stop_patience,
            validation_grid_early_stop_relative_tolerance=grid_early_stop_relative_tolerance,
            test_and_ood_used_for_grid_selection=False,
            pykan_fit_defaults_for_grid_updates=True,
            explicit_model_speed_call=False,
        ),
        normalization=dict(
            enabled_inputs=bool(cfg.get("normalize_inputs", False)),
            enabled_outputs=bool(cfg.get("normalize_outputs", False)),
            x_mean=xmu, x_std=xsd, y_mean=ymu, y_std=ysd,
        ),
    )
    write_json(str(outdir / "result.json"), json_safe(result))
    print(
        "[official pyKAN Feynman sweep] selected "
        f"depth={selected_candidate['depth']} width={fixed_width} "
        f"lambda={selected_candidate['sparsification_lambda']:.3e} "
        f"grid={selected_candidate['final_grid']} "
        f"shape={selected_candidate['pruned_shape']} val_mse={result['val_metrics']['mse']:.6e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
