#!/usr/bin/env python3
"""Accuracy-first pyKAN sweep for the SoftSaturatedLorenz96 SI benchmark.

This adapter uses the bundled official pyKAN implementation without modifying
upstream source.  The case-local protocol is:

1. train an unregularized accuracy model at the minimum admissible grid;
2. refine the unpruned accuracy model over increasing grids with validation
   patience;
3. optionally continue from the previous smaller-sample selected checkpoint;
4. from each accuracy checkpoint, try mild sparsification, official pruning,
   zero-lambda recovery, and the remaining grid-refinement schedule;
5. retain the validation-best model among the untouched accuracy branch,
   recovered unpruned branch, and recovered pruned branch;
6. before every official prune/refine call, explicitly refresh pyKAN's cache
   with the current training inputs;
7. enforce only the cross-sample minimum grid G (not active-edge/parameter
   counts), and include the previous model unchanged as a candidate so a larger
   nested sample set cannot worsen the fixed validation score merely because of
   an optimization failure.

ID-test and OOD labels are reserved for final evaluation only.  Candidate
selection uses the same denormalized physical-scale validation MSE as the other
baselines.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    def __init__(self, iterable: Iterable, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, *_args, **_kwargs):
        return None


def _predict(model, x, torch, y_mu, y_sd):
    old_symbolic_enabled = bool(getattr(model, "symbolic_enabled", False))
    model.symbolic_enabled = False
    try:
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=model.device)
            yp = model(xt).detach().cpu().numpy()
    finally:
        model.symbolic_enabled = old_symbolic_enabled
    return yp * y_sd + y_mu


def _tensor_payload(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return {
        "shape": [int(v) for v in array.shape],
        "data": np.asarray(array, dtype=float).ravel(order="F").tolist(),
    }


def _export_portable_model(model, x_mu, x_sd, y_mu, y_sd):
    layers = []
    for layer_index, layer in enumerate(model.act_fun):
        dim_sum = int(model.width[layer_index + 1][0])
        dim_mult = int(model.width[layer_index + 1][1])
        if bool(model.mult_homo):
            mult_arities = [int(model.mult_arity)] * dim_mult
        else:
            raw = model.mult_arity[layer_index + 1] if dim_mult else []
            mult_arities = [int(v) for v in raw]
        layers.append({
            "grid": _tensor_payload(layer.grid),
            "coef": _tensor_payload(layer.coef),
            "scale_base": _tensor_payload(layer.scale_base),
            "scale_spline": _tensor_payload(layer.scale_sp),
            "mask": _tensor_payload(layer.mask),
            "spline_order": int(layer.k),
            "base_function": str(model.base_fun_name),
            "subnode_scale": _tensor_payload(model.subnode_scale[layer_index]),
            "subnode_bias": _tensor_payload(model.subnode_bias[layer_index]),
            "node_scale": _tensor_payload(model.node_scale[layer_index]),
            "node_bias": _tensor_payload(model.node_bias[layer_index]),
            "sum_node_count": dim_sum,
            "multiplication_node_count": dim_mult,
            "multiplication_arities": mult_arities,
        })
    return {
        "format": "phdnn_portable_pykan_v1",
        "input_id_zero_based": [int(v) for v in model.input_id.detach().cpu().numpy().ravel()],
        "depth": int(model.depth),
        "layers": layers,
        "normalization": {
            "x_mean": _tensor_payload(x_mu),
            "x_std": _tensor_payload(x_sd),
            "y_mean": _tensor_payload(y_mu),
            "y_std": _tensor_payload(y_sd),
        },
    }


def _shape_json(width):
    out = []
    for value in width:
        if isinstance(value, (list, tuple)):
            out.append([int(x) for x in value])
        elif hasattr(value, "tolist"):
            converted = value.tolist()
            out.append(converted if isinstance(converted, list) else int(converted))
        else:
            out.append(int(value))
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
    return model.fit(
        dataset,
        opt=str(opt),
        steps=max(int(steps), 1),
        log=max(int(steps), 1) + 1,
        lamb=float(lamb),
        lr=float(lr),
        batch=-1,
    )


def _refresh_train_cache(model, dataset) -> None:
    """Ensure prune/refine importance and grid initialization use train data."""
    model.get_act(dataset["train_input"])


def _evaluate(model, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd):
    train_metrics = metrics(Ytr, _predict(model, Xtrn, torch, ymu, ysd))
    val_metrics = metrics(Yv, _predict(model, Xvn, torch, ymu, ysd))
    _assert_finite_metrics(val_metrics, "candidate_evaluation")
    return train_metrics, val_metrics


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


def _normalization_from_cfg(cfg, Xtr, Ytr):
    fixed = cfg.get("fixed_normalization", None)
    if isinstance(fixed, dict) and all(
        key in fixed for key in ("x_mean", "x_std", "y_mean", "y_std")
    ):
        xmu = np.asarray(fixed["x_mean"], dtype=float).reshape(1, -1)
        xsd = np.asarray(fixed["x_std"], dtype=float).reshape(1, -1)
        ymu = np.asarray(fixed["y_mean"], dtype=float).reshape(1, -1)
        ysd = np.asarray(fixed["y_std"], dtype=float).reshape(1, -1)
        if xmu.shape[1] != Xtr.shape[1] or ymu.shape[1] != Ytr.shape[1]:
            raise ValueError("Inherited KAN normalization dimensions do not match current data.")
        if np.any(~np.isfinite(xmu)) or np.any(~np.isfinite(ymu)):
            raise ValueError("Inherited KAN normalization contains non-finite means.")
        if np.any(~np.isfinite(xsd)) or np.any(xsd <= 0) or np.any(~np.isfinite(ysd)) or np.any(ysd <= 0):
            raise ValueError("Inherited KAN normalization contains invalid standard deviations.")
        return xmu, xsd, ymu, ysd, True

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
    return xmu, xsd, ymu, ysd, False


def _constructor_payload(model):
    return dict(
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
    )


def _save_native_checkpoint(path, model, torch, metadata: Dict[str, Any]) -> None:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {
        "format": "phdnn_pykan_native_warm_start_v1",
        "constructor": _constructor_payload(model),
        "state_dict": state,
        "metadata": metadata,
    }
    torch.save(payload, str(path))


def _load_native_checkpoint(path, KAN, torch, device):
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("format") != "phdnn_pykan_native_warm_start_v1":
        raise ValueError("Unsupported KAN warm-start checkpoint format.")
    constructor = dict(payload["constructor"])
    constructor.update(auto_save=False, first_init=False, device=device)
    model = KAN(**constructor)
    model.load_state_dict(payload["state_dict"])
    model.auto_save = False
    return model, dict(payload.get("metadata", {}))


def _ordered_grid_schedule(grid_list: Sequence[int], minimum_grid: int) -> List[int]:
    schedule = sorted({int(g) for g in grid_list if int(g) >= int(minimum_grid)} | {int(minimum_grid)})
    if any(g <= 0 for g in schedule):
        raise ValueError("KAN grids must be positive.")
    return schedule


def _refine_path(
    model,
    *,
    current_grid: int,
    grid_schedule: Sequence[int],
    dataset,
    Xtrn,
    Ytr,
    Xvn,
    Yv,
    torch,
    KAN,
    ymu,
    ysd,
    optimizer: str,
    steps: int,
    learning_rate: float,
    patience: int,
    relative_tolerance: float,
    phase_prefix: str,
    include_current_fit: bool,
):
    """Run lambda=0 recovery/refinement and restore validation-best grid."""
    stages = []
    working = model
    best_model = None
    best_grid = int(current_grid)
    best_train = None
    best_val = None
    best_val_mse = float("inf")
    last_attempted_grid = int(current_grid)
    bad_count = 0
    stop_reason = "grid_schedule_completed; selected_validation_best_checkpoint"

    if include_current_fit:
        stage_start = time.perf_counter()
        _fit_official(
            working,
            dataset,
            opt=optimizer,
            steps=steps,
            lamb=0.0,
            lr=learning_rate,
        )
        train_m, val_m = _evaluate(working, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd)
        best_model = _clone_model(working, KAN)
        best_train = dict(train_m)
        best_val = dict(val_m)
        best_val_mse = float(val_m["mse"])
        stages.append(_stage_record(
            phase=f"{phase_prefix}_fit",
            grid=current_grid,
            status="ok",
            time_seconds=time.perf_counter() - stage_start,
            val_mse=best_val_mse,
            accepted=True,
        ))
    else:
        train_m, val_m = _evaluate(working, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd)
        best_model = _clone_model(working, KAN)
        best_train = dict(train_m)
        best_val = dict(val_m)
        best_val_mse = float(val_m["mse"])
        stages.append(_stage_record(
            phase=f"{phase_prefix}_checkpoint",
            grid=current_grid,
            status="ok",
            time_seconds=0.0,
            val_mse=best_val_mse,
            accepted=True,
        ))

    for grid in [int(g) for g in grid_schedule if int(g) > int(current_grid)]:
        stage_start = time.perf_counter()
        try:
            _refresh_train_cache(working, dataset)
            old = working
            working = old.refine(grid)
            working.auto_save = False
            del old
            gc.collect()
            _fit_official(
                working,
                dataset,
                opt=optimizer,
                steps=steps,
                lamb=0.0,
                lr=learning_rate,
            )
            train_m, val_m = _evaluate(working, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd)
            current_mse = float(val_m["mse"])
            denominator = max(abs(best_val_mse), np.finfo(float).tiny)
            relative_improvement = (best_val_mse - current_mse) / denominator
            is_new_best = current_mse < best_val_mse
            meaningful_rise = current_mse > best_val_mse * (1.0 + relative_tolerance)
            last_attempted_grid = int(grid)
            stage_status = "ok"
            stage_reason = ""
            if is_new_best:
                old_best = best_model
                best_model = _clone_model(working, KAN)
                best_grid = int(grid)
                best_train = dict(train_m)
                best_val = dict(val_m)
                best_val_mse = current_mse
                bad_count = 0
                del old_best
                gc.collect()
            elif meaningful_rise:
                bad_count += 1
            else:
                # A change inside the tolerance band is not evidence of a
                # persistent rise, so the lookahead counter is reset.
                bad_count = 0
            should_stop = meaningful_rise and bad_count >= patience
            if should_stop:
                stage_status = "early_stop_trigger"
                stage_reason = (
                    f"validation_mse_increased_for_{bad_count}_grids_at_grid_{grid}; "
                    f"restore_best_grid_{best_grid}"
                )
                stop_reason = stage_reason
            stages.append(_stage_record(
                phase=f"{phase_prefix}_refine_and_fit",
                grid=grid,
                status=stage_status,
                time_seconds=time.perf_counter() - stage_start,
                val_mse=current_mse,
                accepted=is_new_best,
                relative_improvement=relative_improvement,
                stop_reason=stage_reason,
            ))
            if should_stop:
                break
        except Exception as error:
            stages.append(_stage_record(
                phase=f"{phase_prefix}_refine_failure",
                grid=grid,
                status="failed",
                time_seconds=time.perf_counter() - stage_start,
                val_mse=float("nan"),
                accepted=False,
                stop_reason=f"restore_best_grid_{best_grid}",
                error=f"{type(error).__name__}: {error}",
            ))
            stop_reason = f"grid_{grid}_failed; restore_best_grid_{best_grid}"
            break

    if working is not None:
        del working
    model_out = best_model
    return model_out, best_grid, last_attempted_grid, best_train, best_val, stages, stop_reason


def _branch_summary(model, grid, train_m, val_m, source, sparsification_lambda, stages, KAN):
    active_edges, active_coefficients, trainable_parameters = _active_size(model)
    return dict(
        model=model,
        grid=int(grid),
        train_metrics=dict(train_m),
        val_metrics=dict(val_m),
        source=str(source),
        sparsification_lambda=float(sparsification_lambda),
        stages=list(stages),
        active_edge_count=int(active_edges),
        active_coefficient_count=int(active_coefficients),
        trainable_parameter_count=int(trainable_parameters),
        final_shape=_shape_json(model.width),
    )


def _select_branch(branches: Sequence[Dict[str, Any]]):
    valid = [b for b in branches if np.isfinite(float(b["val_metrics"]["mse"]))]
    if not valid:
        raise RuntimeError("No finite KAN branch is available for validation selection.")
    return min(
        valid,
        key=lambda b: (
            float(b["val_metrics"]["mse"]),
            int(b["active_coefficient_count"]),
            int(b["grid"]),
        ),
    )


def _process_accuracy_checkpoint(
    accuracy_model,
    *,
    accuracy_grid: int,
    base_source: str,
    accuracy_train,
    accuracy_val,
    base_stages,
    grid_schedule,
    sparsification_lambdas,
    dataset,
    Xtrn,
    Ytr,
    Xvn,
    Yv,
    torch,
    KAN,
    ymu,
    ysd,
    optimizer,
    sparse_steps,
    recovery_steps,
    learning_rate,
    node_th,
    edge_th,
    prune_guard_enable,
    prune_max_relative_increase,
    grid_patience,
    grid_relative_tolerance,
):
    branches = []
    accuracy_branch = _branch_summary(
        _clone_model(accuracy_model, KAN),
        accuracy_grid,
        accuracy_train,
        accuracy_val,
        f"{base_source}_accuracy_unpruned",
        0.0,
        list(base_stages),
        KAN,
    )
    accuracy_branch.update(
        pruning_accepted=False,
        pruning_guard_reason="accuracy_branch_selected_without_sparsification",
        pruned_shape=[],
        pre_prune_train_metrics=dict(accuracy_train),
        pre_prune_val_metrics=dict(accuracy_val),
        immediate_post_prune_train_metrics={name: float("nan") for name in ("mse", "rmse", "mae", "nrmse", "nrmseRange", "nmae")},
        immediate_post_prune_val_metrics={name: float("nan") for name in ("mse", "rmse", "mae", "nrmse", "nrmseRange", "nmae")},
        post_refit_train_metrics=dict(accuracy_train),
        post_refit_val_metrics=dict(accuracy_val),
        last_attempted_grid=int(accuracy_grid),
        grid_stop_reason="accuracy_validation_best_checkpoint",
    )
    branches.append(accuracy_branch)

    for sparse_lambda in sparsification_lambdas:
        sparse_model = _clone_model(accuracy_model, KAN)
        branch_stages = list(base_stages)
        stage_start = time.perf_counter()
        _fit_official(
            sparse_model,
            dataset,
            opt=optimizer,
            steps=sparse_steps,
            lamb=float(sparse_lambda),
            lr=learning_rate,
        )
        sparse_train, sparse_val = _evaluate(
            sparse_model, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd
        )
        branch_stages.append(_stage_record(
            phase="mild_sparsification_fit",
            grid=accuracy_grid,
            status="ok",
            time_seconds=time.perf_counter() - stage_start,
            val_mse=sparse_val["mse"],
            accepted=True,
        ))

        unpruned_sparse = _clone_model(sparse_model, KAN)
        pruned_model = None
        immediate_post_train = {}
        immediate_post_val = {}
        pruned_shape = []
        prune_error = ""
        try:
            stage_start = time.perf_counter()
            _refresh_train_cache(sparse_model, dataset)
            pruned_model = sparse_model.prune(node_th=node_th, edge_th=edge_th)
            sparse_model = None
            pruned_model.auto_save = False
            if _hidden_node_count(pruned_model.width) <= 0:
                raise RuntimeError("Official pruning removed every hidden node.")
            immediate_post_train, immediate_post_val = _evaluate(
                pruned_model, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd
            )
            pruned_shape = _shape_json(pruned_model.width)
            branch_stages.append(_stage_record(
                phase="official_prune_train_cache",
                grid=accuracy_grid,
                status="ok",
                time_seconds=time.perf_counter() - stage_start,
                val_mse=immediate_post_val["mse"],
                accepted=True,
            ))
        except Exception as error:
            prune_error = f"{type(error).__name__}: {error}"
            if sparse_model is not None:
                del sparse_model
                sparse_model = None
            if pruned_model is not None:
                del pruned_model
                pruned_model = None
            branch_stages.append(_stage_record(
                phase="official_prune_failure",
                grid=accuracy_grid,
                status="failed",
                time_seconds=0.0,
                val_mse=float("nan"),
                accepted=False,
                stop_reason="continue_unpruned_recovery_and_grid_refinement",
                error=prune_error,
            ))
            gc.collect()

        # The unpruned sparse fallback always completes the same downstream
        # zero-lambda recovery and grid-refinement schedule.
        unpruned_model, unpruned_grid, unpruned_last_grid, unpruned_train, unpruned_val, unpruned_stages, unpruned_stop = _refine_path(
            unpruned_sparse,
            current_grid=accuracy_grid,
            grid_schedule=grid_schedule,
            dataset=dataset,
            Xtrn=Xtrn,
            Ytr=Ytr,
            Xvn=Xvn,
            Yv=Yv,
            torch=torch,
            KAN=KAN,
            ymu=ymu,
            ysd=ysd,
            optimizer=optimizer,
            steps=recovery_steps,
            learning_rate=learning_rate,
            patience=grid_patience,
            relative_tolerance=grid_relative_tolerance,
            phase_prefix="unpruned_zero_lambda_recovery",
            include_current_fit=True,
        )
        unpruned_branch = _branch_summary(
            unpruned_model,
            unpruned_grid,
            unpruned_train,
            unpruned_val,
            f"{base_source}_unpruned_recovered",
            sparse_lambda,
            branch_stages + unpruned_stages,
            KAN,
        )
        unpruned_branch.update(
            pruning_accepted=False,
            pruning_guard_reason=(
                "pruned_branch_failed; completed_unpruned_recovery: " + prune_error
                if prune_error else "unpruned_recovery_reference"
            ),
            pruned_shape=pruned_shape,
            pre_prune_train_metrics=dict(sparse_train),
            pre_prune_val_metrics=dict(sparse_val),
            immediate_post_prune_train_metrics=dict(immediate_post_train),
            immediate_post_prune_val_metrics=dict(immediate_post_val),
            post_refit_train_metrics=dict(unpruned_train),
            post_refit_val_metrics=dict(unpruned_val),
            last_attempted_grid=int(unpruned_last_grid),
            grid_stop_reason=str(unpruned_stop),
        )

        chosen_sparse_branch = unpruned_branch
        if pruned_model is not None:
            pruned_model, pruned_grid, pruned_last_grid, pruned_train, pruned_val, pruned_stages, pruned_stop = _refine_path(
                pruned_model,
                current_grid=accuracy_grid,
                grid_schedule=grid_schedule,
                dataset=dataset,
                Xtrn=Xtrn,
                Ytr=Ytr,
                Xvn=Xvn,
                Yv=Yv,
                torch=torch,
                KAN=KAN,
                ymu=ymu,
                ysd=ysd,
                optimizer=optimizer,
                steps=recovery_steps,
                learning_rate=learning_rate,
                patience=grid_patience,
                relative_tolerance=grid_relative_tolerance,
                phase_prefix="pruned_zero_lambda_recovery",
                include_current_fit=True,
            )
            pruned_branch = _branch_summary(
                pruned_model,
                pruned_grid,
                pruned_train,
                pruned_val,
                f"{base_source}_pruned_recovered",
                sparse_lambda,
                branch_stages + pruned_stages,
                KAN,
            )
            allowed = float(unpruned_val["mse"]) * (1.0 + prune_max_relative_increase)
            accept_pruned = (not prune_guard_enable) or float(pruned_val["mse"]) <= allowed
            reason = (
                "pruned_recovered_validation_branch_selected"
                if accept_pruned
                else "pruned_recovered_validation_mse_worse_than_unpruned_recovered"
            )
            pruned_branch.update(
                pruning_accepted=bool(accept_pruned),
                pruning_guard_reason=reason,
                pruned_shape=pruned_shape,
                pre_prune_train_metrics=dict(sparse_train),
                pre_prune_val_metrics=dict(sparse_val),
                immediate_post_prune_train_metrics=dict(immediate_post_train),
                immediate_post_prune_val_metrics=dict(immediate_post_val),
                post_refit_train_metrics=dict(pruned_train),
                post_refit_val_metrics=dict(pruned_val),
                last_attempted_grid=int(pruned_last_grid),
                grid_stop_reason=str(pruned_stop),
            )
            if accept_pruned and float(pruned_val["mse"]) <= float(unpruned_val["mse"]):
                chosen_sparse_branch = pruned_branch
                del unpruned_branch["model"]
            else:
                del pruned_branch["model"]
            gc.collect()

        branches.append(chosen_sparse_branch)

    selected = _select_branch(branches)
    for branch in branches:
        if branch is not selected and "model" in branch:
            del branch["model"]
    gc.collect()
    return selected, branches


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

    xmu, xsd, ymu, ysd, inherited_normalization = _normalization_from_cfg(cfg, Xtr, Ytr)
    Xtrn = standardize(Xtr, xmu, xsd)
    Xvn = standardize(Xv, xmu, xsd)
    Xtn = standardize(Xt, xmu, xsd)
    Xon = standardize(Xo, xmu, xsd) if Xo.size else Xo
    Ytrn = standardize(Ytr, ymu, ysd)
    Yvn = standardize(Yv, ymu, ysd)

    add_pykan_path(cfg["pykan_root"])
    import torch

    torch.set_default_dtype(
        torch.float64 if str(cfg.get("dtype", "float64")).lower() == "float64" else torch.float32
    )
    torch_threads = int(cfg.get("torch_num_threads", 0) or 0)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    multkan_module = importlib.import_module("kan.MultKAN")
    multkan_module.tqdm = _QuietTqdm
    KAN = multkan_module.KAN

    device = str(cfg.get("device", "cpu"))
    seed = int(cfg.get("seed", 1))
    fixed_width = int(cfg.get("width", 8))
    requested_depths = [int(v) for v in cfg.get("depth_list", [2, 3, 4, 5, 6])]
    minimum_depth = int(cfg.get("minimum_depth", min(requested_depths)))
    depths = [d for d in requested_depths if d >= minimum_depth]
    if not depths:
        raise ValueError("No KAN depth remains after applying minimum_depth.")

    requested_grids = [int(v) for v in cfg.get("grid_list", [3, 5, 10, 20, 50, 100, 200])]
    minimum_grid = int(cfg.get("minimum_grid", min(requested_grids)))
    grids = _ordered_grid_schedule(requested_grids, minimum_grid)
    spline_order = int(cfg.get("spline_order", 3))
    sparsification_lambdas = [float(v) for v in cfg.get("sparsification_lambda_list", [1e-5, 1e-4, 1e-3])]
    accuracy_steps = int(cfg.get("accuracy_steps_per_grid", cfg.get("steps_per_grid", 200)))
    sparse_steps = int(cfg.get("sparsification_steps", cfg.get("steps_per_grid", 200)))
    recovery_steps = int(cfg.get("recovery_steps_per_grid", cfg.get("steps_per_grid", 200)))
    optimizer = str(cfg.get("optimizer", "LBFGS"))
    learning_rate = float(cfg.get("learning_rate", 1.0))
    node_th = float(cfg.get("prune_node_threshold", 1e-2))
    edge_th = float(cfg.get("prune_edge_threshold", 3e-2))
    grid_patience = max(int(cfg.get("grid_early_stop_patience", 2)), 1)
    grid_relative_tolerance = max(float(cfg.get("grid_early_stop_relative_tolerance", 0.015)), 0.0)
    depth_early_stop = bool(cfg.get("depth_early_stop", True))
    depth_patience = max(int(cfg.get("depth_early_stop_patience", 2)), 1)
    depth_relative_tolerance = max(float(cfg.get("depth_early_stop_relative_tolerance", 0.0)), 0.0)
    prune_guard_enable = bool(cfg.get("prune_validation_guard_enable", True))
    prune_max_relative_increase = max(float(cfg.get("prune_max_relative_validation_increase", 0.0)), 0.0)
    warm_start_enable = bool(cfg.get("warm_start_enable", True))
    warm_start_checkpoint_path = str(cfg.get("warm_start_checkpoint_path", "") or "")

    dataset = {
        "train_input": torch.as_tensor(Xtrn, dtype=torch.get_default_dtype(), device=device),
        "train_label": torch.as_tensor(Ytrn, dtype=torch.get_default_dtype(), device=device),
        "test_input": torch.as_tensor(Xvn, dtype=torch.get_default_dtype(), device=device),
        "test_label": torch.as_tensor(Yvn, dtype=torch.get_default_dtype(), device=device),
    }

    nan_metrics = {name: float("nan") for name in ("mse", "rmse", "mae", "nrmse", "nrmseRange", "nmae")}
    candidates: List[Dict[str, Any]] = []
    selected_model = None
    selected_candidate = None
    depth_stop_records = []
    best_depth_val = float("inf")
    best_depth = None
    bad_depth_count = 0

    def register_candidate(candidate: Dict[str, Any], model) -> None:
        nonlocal selected_model, selected_candidate
        candidates.append(candidate)
        if candidate.get("status") != "ok":
            return
        key = (
            float(candidate["val_metrics"]["mse"]),
            int(candidate["active_coefficient_count"]),
            int(candidate["grid"]),
            int(candidate["depth"]),
        )
        old_key = None
        if selected_candidate is not None:
            old_key = (
                float(selected_candidate["val_metrics"]["mse"]),
                int(selected_candidate["active_coefficient_count"]),
                int(selected_candidate["grid"]),
                int(selected_candidate["depth"]),
            )
        if old_key is None or key < old_key:
            old_model = selected_model
            selected_model = model
            selected_candidate = candidate
            if old_model is not None and old_model is not selected_model:
                del old_model
        elif model is not None:
            del model
        gc.collect()

    # The unchanged previous smaller-N model is a valid inherited candidate.
    if warm_start_enable and warm_start_checkpoint_path:
        warm_path = Path(warm_start_checkpoint_path)
        if warm_path.is_file():
            warm_start = time.perf_counter()
            try:
                warm_model, warm_meta = _load_native_checkpoint(warm_path, KAN, torch, device)
                warm_depth = int(warm_meta.get("depth", warm_model.depth))
                warm_grid = int(warm_meta.get("grid", getattr(warm_model, "grid", minimum_grid)))
                if warm_grid < minimum_grid:
                    raise ValueError(
                        f"Warm-start grid {warm_grid} is below required minimum_grid={minimum_grid}."
                    )
                warm_train, warm_val = _evaluate(
                    warm_model, Xtrn, Ytr, Xvn, Yv, torch, ymu, ysd
                )
                warm_stages = [_stage_record(
                    phase="previous_sample_checkpoint_unchanged",
                    grid=warm_grid,
                    status="ok",
                    time_seconds=0.0,
                    val_mse=warm_val["mse"],
                    accepted=True,
                )]
                warm_accuracy_model, warm_accuracy_grid, warm_last_grid, warm_accuracy_train, warm_accuracy_val, warm_refine_stages, warm_stop = _refine_path(
                    _clone_model(warm_model, KAN),
                    current_grid=warm_grid,
                    grid_schedule=grids,
                    dataset=dataset,
                    Xtrn=Xtrn,
                    Ytr=Ytr,
                    Xvn=Xvn,
                    Yv=Yv,
                    torch=torch,
                    KAN=KAN,
                    ymu=ymu,
                    ysd=ysd,
                    optimizer=optimizer,
                    steps=accuracy_steps,
                    learning_rate=learning_rate,
                    patience=grid_patience,
                    relative_tolerance=grid_relative_tolerance,
                    phase_prefix="warm_start_accuracy",
                    include_current_fit=True,
                )
                warm_selected, _ = _process_accuracy_checkpoint(
                    warm_accuracy_model,
                    accuracy_grid=warm_accuracy_grid,
                    base_source="warm_start",
                    accuracy_train=warm_accuracy_train,
                    accuracy_val=warm_accuracy_val,
                    base_stages=warm_stages + warm_refine_stages,
                    grid_schedule=grids,
                    sparsification_lambdas=sparsification_lambdas,
                    dataset=dataset,
                    Xtrn=Xtrn,
                    Ytr=Ytr,
                    Xvn=Xvn,
                    Yv=Yv,
                    torch=torch,
                    KAN=KAN,
                    ymu=ymu,
                    ysd=ysd,
                    optimizer=optimizer,
                    sparse_steps=sparse_steps,
                    recovery_steps=recovery_steps,
                    learning_rate=learning_rate,
                    node_th=node_th,
                    edge_th=edge_th,
                    prune_guard_enable=prune_guard_enable,
                    prune_max_relative_increase=prune_max_relative_increase,
                    grid_patience=grid_patience,
                    grid_relative_tolerance=grid_relative_tolerance,
                )
                # Also retain the unchanged previous model itself.  This is the
                # validation non-regression guard for nested sample sweeps.
                unchanged = _branch_summary(
                    warm_model,
                    warm_grid,
                    warm_train,
                    warm_val,
                    "previous_sample_checkpoint_unchanged",
                    float(warm_meta.get("sparsification_lambda", 0.0)),
                    warm_stages,
                    KAN,
                )
                unchanged.update(
                    pruning_accepted=bool(warm_meta.get("pruning_accepted", False)),
                    pruning_guard_reason="cross_sample_validation_non_regression_candidate",
                    pruned_shape=warm_meta.get("pruned_shape", []),
                    pre_prune_train_metrics=dict(nan_metrics),
                    pre_prune_val_metrics=dict(nan_metrics),
                    immediate_post_prune_train_metrics=dict(nan_metrics),
                    immediate_post_prune_val_metrics=dict(nan_metrics),
                    post_refit_train_metrics=dict(warm_train),
                    post_refit_val_metrics=dict(warm_val),
                    last_attempted_grid=warm_grid,
                    grid_stop_reason="previous_sample_checkpoint_retained",
                )
                chosen = _select_branch([warm_selected, unchanged])
                other = unchanged if chosen is warm_selected else warm_selected
                if "model" in other:
                    del other["model"]
                candidate = dict(
                    depth=warm_depth,
                    hidden_layer_count=max(warm_depth - 1, 0),
                    width=fixed_width,
                    initial_shape=warm_meta.get("initial_shape", _shape_json(chosen["model"].width)),
                    sparsification_lambda=chosen["sparsification_lambda"],
                    seed=seed,
                    status="ok",
                    failed_stage="",
                    time_seconds=time.perf_counter() - warm_start,
                    grid=chosen["grid"],
                    final_grid=chosen["grid"],
                    last_attempted_grid=chosen["last_attempted_grid"],
                    best_grid_val_mse=chosen["val_metrics"]["mse"],
                    grid_stop_reason=chosen["grid_stop_reason"],
                    grid_stages=chosen["stages"],
                    pruned_shape=chosen["pruned_shape"],
                    final_shape=chosen["final_shape"],
                    active_edge_count=chosen["active_edge_count"],
                    active_coefficient_count=chosen["active_coefficient_count"],
                    trainable_parameter_count=chosen["trainable_parameter_count"],
                    train_metrics=chosen["train_metrics"],
                    val_metrics=chosen["val_metrics"],
                    pruning_accepted=chosen["pruning_accepted"],
                    pruning_guard_reason=chosen["pruning_guard_reason"],
                    selected_structure_source=chosen["source"],
                    pre_prune_train_metrics=chosen["pre_prune_train_metrics"],
                    pre_prune_val_metrics=chosen["pre_prune_val_metrics"],
                    immediate_post_prune_train_metrics=chosen["immediate_post_prune_train_metrics"],
                    immediate_post_prune_val_metrics=chosen["immediate_post_prune_val_metrics"],
                    post_refit_train_metrics=chosen["post_refit_train_metrics"],
                    post_refit_val_metrics=chosen["post_refit_val_metrics"],
                    warm_start=True,
                    error="",
                    traceback="",
                )
                register_candidate(candidate, chosen["model"])
                del warm_accuracy_model
            except Exception as error:
                candidates.append(dict(
                    depth=-1,
                    hidden_layer_count=-1,
                    width=fixed_width,
                    initial_shape=[],
                    sparsification_lambda=float("nan"),
                    seed=seed,
                    status="failed",
                    failed_stage="warm_start",
                    time_seconds=time.perf_counter() - warm_start,
                    grid=None,
                    final_grid=None,
                    last_attempted_grid=None,
                    best_grid_val_mse=float("nan"),
                    grid_stop_reason="",
                    grid_stages=[],
                    pruned_shape=[],
                    final_shape=[],
                    active_edge_count=None,
                    active_coefficient_count=None,
                    trainable_parameter_count=None,
                    train_metrics=dict(nan_metrics),
                    val_metrics=dict(nan_metrics),
                    pruning_accepted=False,
                    pruning_guard_reason="warm_start_failed",
                    selected_structure_source="warm_start_failed",
                    pre_prune_train_metrics=dict(nan_metrics),
                    pre_prune_val_metrics=dict(nan_metrics),
                    immediate_post_prune_train_metrics=dict(nan_metrics),
                    immediate_post_prune_val_metrics=dict(nan_metrics),
                    post_refit_train_metrics=dict(nan_metrics),
                    post_refit_val_metrics=dict(nan_metrics),
                    warm_start=True,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=20),
                ))

    for depth in depths:
        candidate_start = time.perf_counter()
        shape = [Xtr.shape[1]] + [fixed_width] * max(depth - 1, 0) + [Ytr.shape[1]]
        model = None
        try:
            seed_all(seed, torch)
            checkpoint_dir = outdir / f"ckpt_depth{depth}_scratch"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model = KAN(
                width=list(shape),
                grid=grids[0],
                k=spline_order,
                seed=seed,
                device=device,
                auto_save=False,
                ckpt_path=str(checkpoint_dir),
            )
            accuracy_model, accuracy_grid, accuracy_last_grid, accuracy_train, accuracy_val, accuracy_stages, accuracy_stop = _refine_path(
                model,
                current_grid=grids[0],
                grid_schedule=grids,
                dataset=dataset,
                Xtrn=Xtrn,
                Ytr=Ytr,
                Xvn=Xvn,
                Yv=Yv,
                torch=torch,
                KAN=KAN,
                ymu=ymu,
                ysd=ysd,
                optimizer=optimizer,
                steps=accuracy_steps,
                learning_rate=learning_rate,
                patience=grid_patience,
                relative_tolerance=grid_relative_tolerance,
                phase_prefix="scratch_accuracy",
                include_current_fit=True,
            )
            chosen, _ = _process_accuracy_checkpoint(
                accuracy_model,
                accuracy_grid=accuracy_grid,
                base_source="scratch",
                accuracy_train=accuracy_train,
                accuracy_val=accuracy_val,
                base_stages=accuracy_stages,
                grid_schedule=grids,
                sparsification_lambdas=sparsification_lambdas,
                dataset=dataset,
                Xtrn=Xtrn,
                Ytr=Ytr,
                Xvn=Xvn,
                Yv=Yv,
                torch=torch,
                KAN=KAN,
                ymu=ymu,
                ysd=ysd,
                optimizer=optimizer,
                sparse_steps=sparse_steps,
                recovery_steps=recovery_steps,
                learning_rate=learning_rate,
                node_th=node_th,
                edge_th=edge_th,
                prune_guard_enable=prune_guard_enable,
                prune_max_relative_increase=prune_max_relative_increase,
                grid_patience=grid_patience,
                grid_relative_tolerance=grid_relative_tolerance,
            )
            candidate = dict(
                depth=depth,
                hidden_layer_count=max(depth - 1, 0),
                width=fixed_width,
                initial_shape=list(shape),
                sparsification_lambda=chosen["sparsification_lambda"],
                seed=seed,
                status="ok",
                failed_stage="",
                time_seconds=time.perf_counter() - candidate_start,
                grid=chosen["grid"],
                final_grid=chosen["grid"],
                last_attempted_grid=chosen["last_attempted_grid"],
                best_grid_val_mse=chosen["val_metrics"]["mse"],
                grid_stop_reason=chosen["grid_stop_reason"],
                grid_stages=chosen["stages"],
                pruned_shape=chosen["pruned_shape"],
                final_shape=chosen["final_shape"],
                active_edge_count=chosen["active_edge_count"],
                active_coefficient_count=chosen["active_coefficient_count"],
                trainable_parameter_count=chosen["trainable_parameter_count"],
                train_metrics=chosen["train_metrics"],
                val_metrics=chosen["val_metrics"],
                pruning_accepted=chosen["pruning_accepted"],
                pruning_guard_reason=chosen["pruning_guard_reason"],
                selected_structure_source=chosen["source"],
                pre_prune_train_metrics=chosen["pre_prune_train_metrics"],
                pre_prune_val_metrics=chosen["pre_prune_val_metrics"],
                immediate_post_prune_train_metrics=chosen["immediate_post_prune_train_metrics"],
                immediate_post_prune_val_metrics=chosen["immediate_post_prune_val_metrics"],
                post_refit_train_metrics=chosen["post_refit_train_metrics"],
                post_refit_val_metrics=chosen["post_refit_val_metrics"],
                warm_start=False,
                error="",
                traceback="",
            )
            register_candidate(candidate, chosen["model"])
            del accuracy_model

            current_val = float(candidate["val_metrics"]["mse"])
            previous_best = best_depth_val
            is_new_best = current_val < best_depth_val
            meaningful_rise = np.isfinite(best_depth_val) and current_val > best_depth_val * (
                1.0 + depth_relative_tolerance
            )
            if is_new_best:
                best_depth_val = current_val
                best_depth = depth
                bad_depth_count = 0
            elif meaningful_rise:
                bad_depth_count += 1
            else:
                bad_depth_count = 0
            should_stop = depth_early_stop and meaningful_rise and bad_depth_count >= depth_patience
            reason = ""
            if should_stop:
                reason = (
                    f"validation_mse_increased_for_{bad_depth_count}_depths_at_depth_{depth}; "
                    f"restore_best_depth_{best_depth}"
                )
            depth_stop_records.append(dict(
                sparsification_lambda=float(candidate["sparsification_lambda"]),
                depth=int(depth),
                status="early_stop_trigger" if should_stop else "ok",
                val_mse=current_val,
                previous_best_val_mse=previous_best,
                accepted=is_new_best,
                bad_depth_count=bad_depth_count,
                stop_reason=reason,
            ))
            if should_stop:
                break
        except Exception as error:
            if model is not None:
                del model
            candidates.append(dict(
                depth=depth,
                hidden_layer_count=max(depth - 1, 0),
                width=fixed_width,
                initial_shape=list(shape),
                sparsification_lambda=float("nan"),
                seed=seed,
                status="failed",
                failed_stage="scratch_accuracy_or_sparse_branch",
                time_seconds=time.perf_counter() - candidate_start,
                grid=None,
                final_grid=None,
                last_attempted_grid=None,
                best_grid_val_mse=float("nan"),
                grid_stop_reason="",
                grid_stages=[],
                pruned_shape=[],
                final_shape=[],
                active_edge_count=None,
                active_coefficient_count=None,
                trainable_parameter_count=None,
                train_metrics=dict(nan_metrics),
                val_metrics=dict(nan_metrics),
                pruning_accepted=False,
                pruning_guard_reason="candidate_failed",
                selected_structure_source="candidate_failed",
                pre_prune_train_metrics=dict(nan_metrics),
                pre_prune_val_metrics=dict(nan_metrics),
                immediate_post_prune_train_metrics=dict(nan_metrics),
                immediate_post_prune_val_metrics=dict(nan_metrics),
                post_refit_train_metrics=dict(nan_metrics),
                post_refit_val_metrics=dict(nan_metrics),
                warm_start=False,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(limit=20),
            ))
            depth_stop_records.append(dict(
                sparsification_lambda=float("nan"),
                depth=int(depth),
                status="candidate_failed",
                val_mse=float("nan"),
                previous_best_val_mse=best_depth_val,
                accepted=False,
                bad_depth_count=bad_depth_count,
                stop_reason="",
            ))
        gc.collect()

    if selected_candidate is None or selected_model is None:
        failure_summary = "; ".join(
            f"depth={c.get('depth')},source={c.get('selected_structure_source')},"
            f"status={c.get('status')},error={c.get('error')}" for c in candidates
        )
        raise RuntimeError(f"No valid KAN candidate. {failure_summary}")

    yp_train = _predict(selected_model, Xtrn, torch, ymu, ysd)
    yp_val = _predict(selected_model, Xvn, torch, ymu, ysd)
    yp_test = _predict(selected_model, Xtn, torch, ymu, ysd)
    yp_ood = _predict(selected_model, Xon, torch, ymu, ysd) if Xo.size else np.empty((0, Ytr.shape[1]))

    write_csv_matrix(str(outdir / "Yhat_train.csv"), yp_train)
    write_csv_matrix(str(outdir / "Yhat_val.csv"), yp_val)
    write_csv_matrix(str(outdir / "Yhat_test.csv"), yp_test)
    write_csv_matrix(str(outdir / "Yhat_ood.csv"), yp_ood)

    native_checkpoint_path = outdir / "selected_kan_native_checkpoint.pt"
    _save_native_checkpoint(
        native_checkpoint_path,
        selected_model,
        torch,
        metadata=dict(
            depth=int(selected_candidate["depth"]),
            grid=int(selected_candidate["grid"]),
            configured_width=int(fixed_width),
            sparsification_lambda=float(selected_candidate["sparsification_lambda"]),
            pruning_accepted=bool(selected_candidate["pruning_accepted"]),
            pruned_shape=selected_candidate["pruned_shape"],
            initial_shape=selected_candidate["initial_shape"],
        ),
    )

    best_by_lambda = []
    for sparse_lambda in [0.0] + list(sparsification_lambdas):
        matching = [
            c for c in candidates
            if c.get("status") == "ok"
            and np.isfinite(float(c.get("sparsification_lambda", float("nan"))))
            and abs(float(c["sparsification_lambda"]) - sparse_lambda)
            <= max(1e-15, abs(sparse_lambda) * 1e-12)
        ]
        if matching:
            best = min(matching, key=lambda c: float(c["val_metrics"]["mse"]))
            best_by_lambda.append(dict(
                sparsification_lambda=float(sparse_lambda),
                depth=int(best["depth"]),
                selected_grid=int(best["grid"]),
                grid_stop_reason=str(best["grid_stop_reason"]),
                val_mse=float(best["val_metrics"]["mse"]),
                active_coefficient_count=int(best["active_coefficient_count"]),
                pruned_shape=best["final_shape"],
                selected_structure_source=str(best["selected_structure_source"]),
                pruning_accepted=bool(best["pruning_accepted"]),
            ))

    normalization = dict(
        enabled_inputs=bool(cfg.get("normalize_inputs", False)),
        enabled_outputs=bool(cfg.get("normalize_outputs", False)),
        inherited_from_previous_sample=bool(inherited_normalization),
        x_mean=xmu,
        x_std=xsd,
        y_mean=ymu,
        y_std=ysd,
    )

    result = dict(
        method="Official pyKAN SI accuracy-first sweep with grid inheritance",
        protocol="official_pykan_si_accuracy_first_prune_guard_grid_inheritance",
        selection_metric="validation_mse",
        total_time_seconds=time.perf_counter() - total_start,
        selected_candidate_time_seconds=float(selected_candidate["time_seconds"]),
        selected_depth=int(selected_candidate["depth"]),
        selected_hidden_layer_count=int(selected_candidate["hidden_layer_count"]),
        selected_width=int(fixed_width),
        selected_sparsification_lambda=float(selected_candidate["sparsification_lambda"]),
        selected_grid=int(selected_candidate["grid"]),
        selected_last_attempted_grid=int(selected_candidate["last_attempted_grid"]),
        selected_best_grid_val_mse=float(selected_candidate["val_metrics"]["mse"]),
        selected_grid_stop_reason=str(selected_candidate["grid_stop_reason"]),
        selected_grid_stages=selected_candidate["grid_stages"],
        selected_initial_shape=selected_candidate["initial_shape"],
        selected_pruned_shape=selected_candidate["pruned_shape"],
        selected_final_shape=selected_candidate["final_shape"],
        selected_pruning_accepted=bool(selected_candidate["pruning_accepted"]),
        selected_pruning_guard_reason=str(selected_candidate["pruning_guard_reason"]),
        selected_structure_source=str(selected_candidate["selected_structure_source"]),
        selected_warm_start=bool(selected_candidate.get("warm_start", False)),
        active_edge_count=int(selected_candidate["active_edge_count"]),
        active_coefficient_count=int(selected_candidate["active_coefficient_count"]),
        trainable_parameter_count=int(selected_candidate["trainable_parameter_count"]),
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
        configured_candidate_count=len(depths) + (1 if warm_start_checkpoint_path else 0),
        depth_early_stop=depth_early_stop,
        depth_stop_records=depth_stop_records,
        best_by_lambda=best_by_lambda,
        candidates=candidates,
        normalization=normalization,
        portable_model=_export_portable_model(selected_model, xmu, xsd, ymu, ysd),
        native_checkpoint_path=str(native_checkpoint_path.resolve()),
        requested_depth_list=requested_depths,
        eligible_depth_list=depths,
        minimum_depth=minimum_depth,
        requested_grid_list=requested_grids,
        eligible_grid_list=grids,
        minimum_grid=minimum_grid,
        grid_early_stop_patience=grid_patience,
        grid_early_stop_relative_tolerance=grid_relative_tolerance,
        warm_start_enable=warm_start_enable,
        warm_start_checkpoint_requested=warm_start_checkpoint_path,
        warm_start_normalization_inherited=bool(inherited_normalization),
        prune_validation_guard_enable=prune_guard_enable,
        prune_max_relative_validation_increase=prune_max_relative_increase,
        settings=dict(
            accuracy_first=True,
            accuracy_lambda=0.0,
            sparsification_after_accuracy=True,
            sparsification_lambda_list=sparsification_lambdas,
            prune_after_sparsification=True,
            zero_lambda_recovery_after_prune_or_fallback=True,
            fallback_completes_grid_refinement=True,
            train_cache_refreshed_before_prune_and_refine=True,
            cross_sample_complexity_definition="minimum_grid_G_only",
            previous_checkpoint_unchanged_candidate=True,
            test_and_ood_used_for_selection=False,
        ),
    )
    write_json(str(outdir / "result.json"), json_safe(result))
    print(
        "[official pyKAN SI accuracy-first] selected "
        f"depth={selected_candidate['depth']} width={fixed_width} "
        f"lambda={selected_candidate['sparsification_lambda']:.3e} "
        f"grid={selected_candidate['grid']} "
        f"source={selected_candidate['selected_structure_source']} "
        f"val_mse={result['val_metrics']['mse']:.6e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
