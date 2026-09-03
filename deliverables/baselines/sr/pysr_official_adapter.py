#!/usr/bin/env python3
"""Official PySR adapter for the MATLAB PhDN baseline framework.

This script is intentionally small: MATLAB exports data/config files, this
script calls the official PySRRegressor, then writes predictions and metadata
back for MATLAB to evaluate and summarize.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib import metadata as importlib_metadata
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


_WINDOWS_JOB_HANDLE = None


def _install_windows_kill_on_close_job() -> None:
    """Place this adapter and descendants in a kill-on-close Windows Job."""
    global _WINDOWS_JOB_HANDLE
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
        _WINDOWS_JOB_HANDLE = job
        print(f"[PySR lifetime] Job Object active; adapter PID={os.getpid()}", flush=True)
    except Exception as exc:
        print(f"[PySR lifetime] WARNING: Job Object setup failed: {exc!r}", flush=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return True


def _start_lifetime_monitor(parent_pid: int, control_file: str) -> None:
    control = Path(control_file) if control_file else None

    def monitor() -> None:
        while True:
            time.sleep(0.25)
            if control is not None and not control.exists():
                print("[PySR lifetime] MATLAB control file removed; exiting.", flush=True)
                os._exit(130)
            if parent_pid > 0 and not _pid_alive(parent_pid):
                print("[PySR lifetime] Parent MATLAB no longer exists; exiting.", flush=True)
                os._exit(131)

    threading.Thread(target=monitor, name="matlab-lifetime-monitor", daemon=True).start()


def _install_exit_signal_handlers() -> None:
    def terminate(signum: int, _frame: Any) -> None:
        print(f"[PySR lifetime] Received signal {signum}; exiting.", flush=True)
        os._exit(128 + int(signum))
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, terminate)
            except Exception:
                pass


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if (not p.exists()) or p.stat().st_size == 0:
        return np.zeros((0, 0), dtype=float)
    arr = np.loadtxt(str(p), delimiter=',')
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.asarray(arr, dtype=float)


def _save_matrix(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        Path(path).write_text('', encoding='utf-8')
    else:
        np.savetxt(path, arr, delimiter=',')



def _pysr_version_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the pinned PySR 2 pre-release required for equation guesses."""
    try:
        installed = importlib_metadata.version('pysr')
    except importlib_metadata.PackageNotFoundError:
        installed = '0'
    required = str(config.get('minimum_pysr_version', '2.0.0a2')).strip() or '2.0.0a2'
    require_v2 = bool(config.get('require_pysr2', True))
    guesses_enabled = bool(config.get('initial_guesses_enable', False))
    try:
        from packaging.version import Version
        installed_ok = Version(installed) >= Version(required)
        is_v2 = Version(installed).major >= 2
    except Exception:
        installed_ok = installed == required
        is_v2 = installed.startswith('2.')
    if (require_v2 and not is_v2) or (guesses_enabled and not installed_ok):
        raise RuntimeError(
            'This framework requires PySR >= ' + required +
            ' for official equation guesses, but the selected Python environment has PySR ' +
            installed + '. Install with: python -m pip install --upgrade --pre "pysr==' +
            required + '"')
    return {
        'installed': installed,
        'required': required,
        'require_pysr2': require_v2,
        'version_requirement_satisfied': installed_ok,
    }


def _normalize_initial_guess_expression(value: Any, config: Dict[str, Any]) -> str:
    """Translate public MATLAB/PySR grammar aliases to registered PySR names.

    Stage-0 exposes ordinary mathematical spelling to users, such as
    ``sin(x1)``, ``cos(x1)``, and ``sqrt(expr)``.  The typed generator route
    internally represents the fixed trigonometric atoms as derived variables
    and registers protected square root under the Julia name ``sqrt_abs``.
    PySR parses constructor-level guesses against those registered names, so
    the same alias translation used by the search grammar must also be applied
    to every initial guess before it reaches Julia.
    """
    expression = str(value).strip()
    if not expression:
        return expression

    # Rewrite direct sin(x1)/cos(x1) calls only when the explicitly selected
    # strict fixed-atom mode is active.  In the ordinary operator mode, keep
    # every trigonometric call unchanged so guesses such as sin(2*x1),
    # cos(x1+x3), or nested user-authored expressions are parsed by PySR using
    # the registered sin/cos unary operators.
    strict_atoms = bool(config.get('strict_trig_atoms_only', False))
    if _single_generator_typed_prior_enabled(config) and strict_atoms:
        expression = re.sub(
            r'(?<![A-Za-z0-9_])sin\s*\(\s*x1\s*\)',
            'sin_x1_atom', expression)
        expression = re.sub(
            r'(?<![A-Za-z0-9_])cos\s*\(\s*x1\s*\)',
            'cos_x1_atom', expression)

    # In this adapter the public grammar alias ``sqrt`` is always registered
    # with PySR as the protected Julia operator ``sqrt_abs``.  A guess must
    # therefore use sqrt_abs as well; otherwise DynamicExpressions rejects the
    # tree before evolution starts.  The pattern intentionally does not touch
    # an already-normalized ``sqrt_abs(...)`` call.
    expression = re.sub(
        r'(?<![A-Za-z0-9_])sqrt\s*\(',
        'sqrt_abs(', expression)
    return expression


def _shared_initial_guesses(config: Dict[str, Any],
                            n_outputs: int) -> Tuple[List[str], float, Dict[str, Any]]:
    """Return one guess library using the adapter's canonical scope semantics.

    ``single_output_rescue`` existed briefly in the MATLAB rescue controller as
    a descriptive token, but it is not a distinct PySR API scope.  A rescue fit
    already has exactly one target, so its pre-filtered guess library is simply
    shared across the one unresolved output.  Accept that legacy token only for
    a true single-output fit and normalize it to the canonical scope.
    """
    enabled = bool(config.get('initial_guesses_enable', False))
    raw = config.get('initial_guesses', [])
    if isinstance(raw, str):
        raw = [raw]
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        raise TypeError('initial_guesses must be a string list shared by all unresolved outputs.')
    guesses: List[str] = []
    seen = set()
    for value in raw:
        guess = _normalize_initial_guess_expression(value, config)
        if not guess:
            continue
        if guess not in seen:
            seen.add(guess)
            guesses.append(guess)
    fraction = float(config.get('fraction_replaced_guesses', 0.05))
    if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
        raise ValueError('fraction_replaced_guesses must be finite and in [0, 1].')
    if enabled and not guesses:
        # The MATLAB public API uses an empty list to mean "no prior".  Be
        # defensive against stale or hand-edited configs that leave the flag
        # enabled while the library is empty, and follow the no-prior path.
        enabled = False
        config['initial_guesses_enable'] = False
        print(
            '[PySR 2 initial guesses] empty library detected; guesses disabled.',
            flush=True)
    requested_scope = str(config.get(
        'initial_guess_scope', 'shared_all_unresolved_outputs')).strip().lower()
    scope = requested_scope
    legacy_rescue_scope_normalized = False
    if scope == 'single_output_rescue':
        if int(n_outputs) != 1:
            raise ValueError(
                'initial_guess_scope="single_output_rescue" is a legacy alias that is '
                'valid only for a true single-output targeted-rescue fit.')
        scope = 'shared_all_unresolved_outputs'
        config['initial_guess_scope'] = scope
        legacy_rescue_scope_normalized = True
        print(
            '[PySR 2 initial guesses] normalized legacy single-output rescue scope '
            'to shared_all_unresolved_outputs.',
            flush=True)
    elif scope != 'shared_all_unresolved_outputs':
        raise ValueError(
            'Only initial_guess_scope="shared_all_unresolved_outputs" is supported in the '
            'native Stage-0 search.')
    periodic_reinjection_enabled = bool(enabled and fraction > 0.0)
    return (guesses if enabled else [], fraction, {
        'enabled': enabled,
        'scope': scope,
        'requested_scope': requested_scope,
        'legacy_single_output_rescue_scope_normalized': legacy_rescue_scope_normalized,
        'count': len(guesses) if enabled else 0,
        'fraction_replaced_guesses': fraction,
        'periodic_reinjection_enabled': periodic_reinjection_enabled,
        'injection_policy': ('official_periodic_population_replacement'
                             if periodic_reinjection_enabled else 'disabled'),
        'same_library_for_every_unresolved_output': True,
    })


def _configure_native_shared_guesses(model_kwargs: Dict[str, Any],
                                      config: Dict[str, Any],
                                      n_outputs: int) -> Dict[str, Any]:
    """Attach the shared guess library to the PySR 2 constructor arguments.

    PySR 2 stores ``guesses`` and ``fraction_replaced_guesses`` as estimator
    parameters. A positive replacement fraction activates the official soft
    periodic re-injection policy; it does not protect or freeze any subtree.
    For native multi-output regression its public API requires a list of guess
    lists, one per output. We therefore duplicate the same flat library for
    every unresolved output; the contents remain output-agnostic.
    """
    guesses, fraction, guess_stats = _shared_initial_guesses(config, n_outputs)
    config['_resolved_initial_guesses'] = list(guesses)
    config['_resolved_pysr_initial_guesses'] = []

    if not guesses:
        guess_stats['pysr_argument_location'] = 'disabled'
        guess_stats['pysr_constructor_shape'] = 'none'
        return guess_stats

    if n_outputs < 1:
        raise ValueError('n_outputs must be at least one when configuring PySR guesses.')

    if n_outputs == 1:
        pysr_guesses: Any = list(guesses)
        constructor_shape = 'flat_single_output_library'
    else:
        pysr_guesses = [list(guesses) for _ in range(n_outputs)]
        constructor_shape = f'{n_outputs}_identical_output_libraries'

    model_kwargs['guesses'] = pysr_guesses
    model_kwargs['fraction_replaced_guesses'] = fraction
    config['_resolved_pysr_initial_guesses'] = pysr_guesses
    guess_stats['pysr_argument_location'] = 'PySRRegressor_constructor'
    guess_stats['pysr_constructor_shape'] = constructor_shape

    print(
        '[PySR 2 initial guesses] one shared library applied to every unresolved output; ' +
        json.dumps(guess_stats, sort_keys=True), flush=True)
    print('[PySR 2 initial guesses] library=' + json.dumps(guesses), flush=True)
    return guess_stats


def _fit_native_model_once(model: Any, Xtr: np.ndarray, fit_target: np.ndarray,
                           feature_names: List[str], config: Dict[str, Any]) -> None:
    """Run one PySR fit call while preserving the feature-name fallback."""
    if bool(config.get('_pysr_generic_zero_based_feature_names', False)):
        model.fit(Xtr, fit_target)
        return

    fit_kwargs: Dict[str, Any] = {'variable_names': feature_names}
    try:
        model.fit(Xtr, fit_target, **fit_kwargs)
        config['_pysr_generic_zero_based_feature_names'] = False
    except TypeError as exc:
        message = str(exc)
        if 'variable_names' not in message:
            raise
        print(
            '[PySR compatibility] fit(variable_names=...) unsupported; '
            f'using generic feature-name mapping: {exc!r}',
            flush=True)
        config['_pysr_generic_zero_based_feature_names'] = True
        model.fit(Xtr, fit_target)


def _machine_precision_early_stop_plan(config: Dict[str, Any],
                                       Ytr: np.ndarray) -> Dict[str, Any]:
    """Build the all-output, scale-aware early-stop plan.

    PySR's native ``early_stop_condition`` is equation-local.  In a native
    multi-output search, stopping when one easy output reaches the target can
    starve a harder output.  The adapter therefore checks the full-training
    Hall-of-Fame loss of every unresolved output between warm-start chunks and
    stops only when all outputs meet their own scale-aware MSE threshold.
    """
    requested = max(1, int(config.get('niterations', 100)))
    enabled = bool(config.get('machine_precision_early_stop_enable', False))
    abs_floor = max(0.0, float(config.get(
        'machine_precision_early_stop_abs_mse', 1e-12)))
    rel_floor = max(0.0, float(config.get(
        'machine_precision_early_stop_rel_mse', 1e-12)))
    interval = max(1, int(config.get(
        'machine_precision_early_stop_check_interval', 50)))
    min_iterations = max(1, int(config.get(
        'machine_precision_early_stop_min_iterations', interval)))
    interval = min(interval, requested)
    min_iterations = min(min_iterations, requested)

    target = np.asarray(Ytr, dtype=float)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    output_scales = np.maximum(1.0, np.mean(np.square(target), axis=0))
    thresholds = np.maximum(abs_floor, rel_floor * output_scales)
    return {
        'enabled': enabled,
        'requested_iterations': requested,
        'check_interval_iterations': interval,
        'minimum_iterations': min_iterations,
        'absolute_mse_floor': abs_floor,
        'relative_mse_floor': rel_floor,
        'output_scales': [float(v) for v in output_scales],
        'output_mse_thresholds': [float(v) for v in thresholds],
        'all_unresolved_outputs_required': True,
        'loss_source': 'full_training_hall_of_fame',
        'warm_start_chunking': enabled,
    }


def _best_hall_of_fame_losses(model: Any, n_outputs: int) -> List[float]:
    """Return the minimum finite full-data Hall-of-Fame loss per output."""
    equations = getattr(model, 'equations_', None)
    if n_outputs == 1 and not isinstance(equations, list):
        tables = [equations]
    elif isinstance(equations, list):
        tables = list(equations)
    else:
        tables = []

    losses: List[float] = []
    for output_index in range(n_outputs):
        if output_index >= len(tables) or tables[output_index] is None:
            losses.append(float('inf'))
            continue
        table = tables[output_index]
        try:
            values = np.asarray(table['loss'], dtype=float).reshape(-1)
        except Exception:
            losses.append(float('inf'))
            continue
        finite = values[np.isfinite(values)]
        losses.append(float(np.min(finite)) if finite.size else float('inf'))
    return losses


def _set_warm_start_iteration_chunk(model: Any, niterations: int) -> None:
    """Configure one additional warm-start chunk without rebuilding the model."""
    try:
        model.set_params(niterations=int(niterations), warm_start=True)
        return
    except Exception as exc:
        # PySRRegressor is sklearn-like, but retain a narrow compatibility
        # fallback for alpha builds whose set_params signature differed.
        try:
            setattr(model, 'niterations', int(niterations))
            setattr(model, 'warm_start', True)
            print(
                '[PySR compatibility] set_params(niterations, warm_start) '
                f'fallback used: {exc!r}', flush=True)
            return
        except Exception:
            raise RuntimeError(
                'Machine-precision early stopping requires PySR warm_start '
                'and mutable niterations support.') from exc


def _json_finite_list(values: List[float]) -> List[Any]:
    return [float(v) if np.isfinite(v) else None for v in values]


def _fit_native_model(model: Any, Xtr: np.ndarray, fit_target: np.ndarray,
                      feature_names: List[str], config: Dict[str, Any],
                      Ytr: np.ndarray, n_outputs: int) -> Dict[str, Any]:
    """Run PySR with optional all-output machine-precision early stopping."""
    plan = _machine_precision_early_stop_plan(config, Ytr)
    thresholds = np.asarray(plan['output_mse_thresholds'], dtype=float)
    requested = int(plan['requested_iterations'])
    history: List[Dict[str, Any]] = []

    if not bool(plan['enabled']):
        _fit_native_model_once(model, Xtr, fit_target, feature_names, config)
        losses = _best_hall_of_fame_losses(model, n_outputs)
        return {
            **plan,
            'triggered': False,
            'completed_iterations': requested,
            'best_hall_of_fame_losses': _json_finite_list(losses),
            'outputs_reached': [
                bool(np.isfinite(loss) and loss <= thresholds[j])
                for j, loss in enumerate(losses)
            ],
            'check_history': history,
        }

    print(
        '[PySR machine-precision early stop] enabled; '
        f"all outputs required; requested={requested}; "
        f"check_interval={plan['check_interval_iterations']}; "
        f"minimum_iterations={plan['minimum_iterations']}; "
        f"thresholds={plan['output_mse_thresholds']}",
        flush=True)

    completed = 0
    triggered = False
    best_losses = [float('inf')] * n_outputs
    while completed < requested:
        chunk = min(int(plan['check_interval_iterations']), requested - completed)
        _set_warm_start_iteration_chunk(model, chunk)
        _fit_native_model_once(model, Xtr, fit_target, feature_names, config)
        completed += chunk
        best_losses = _best_hall_of_fame_losses(model, n_outputs)
        outputs_reached = [
            bool(np.isfinite(loss) and loss <= thresholds[j])
            for j, loss in enumerate(best_losses)
        ]
        history.append({
            'completed_iterations': completed,
            'best_hall_of_fame_losses': _json_finite_list(best_losses),
            'outputs_reached': outputs_reached,
        })
        print(
            '[PySR machine-precision early stop] '
            f'completed={completed}/{requested}; '
            f'best_full_training_losses={_json_finite_list(best_losses)}; '
            f'outputs_reached={outputs_reached}',
            flush=True)
        if completed >= int(plan['minimum_iterations']) and all(outputs_reached):
            triggered = True
            print(
                '[PySR machine-precision early stop] all unresolved outputs '
                f'reached threshold after {completed} iterations; ending this restart.',
                flush=True)
            break

    return {
        **plan,
        'triggered': triggered,
        'completed_iterations': completed,
        'best_hall_of_fame_losses': _json_finite_list(best_losses),
        'outputs_reached': [
            bool(np.isfinite(loss) and loss <= thresholds[j])
            for j, loss in enumerate(best_losses)
        ],
        'check_history': history,
    }

def _operator_lists(config: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    # Custom-motif registration is handled below.
    binary_in = config.get('binary_operators', ['+', '-', '*', '/'])
    unary_in = config.get('unary_operators', ['inv', 'sqrt'])
    if isinstance(binary_in, str):
        binary_in = [binary_in]
    if isinstance(unary_in, str):
        unary_in = [unary_in]

    binary_ops: List[str] = []
    extra_sympy: Dict[str, Any] = {}
    for op in binary_in:
        op = str(op).strip()
        if not op:
            continue
        if op in {'plus', '+'}:
            binary_ops.append('+')
        elif op in {'minus', '-'}:
            binary_ops.append('-')
        elif op in {'times', '*'}:
            binary_ops.append('*')
        elif op in {'divide', '/'}:
            binary_ops.append('/')
        elif op.lower() == 'op_custom1' or op.lower().startswith('op_custom1('):
            # Shared level-2 physical motif. The Julia definition is exact on
            # its real domain and returns NaN for invalid evolutionary
            # intermediates instead of throwing a domain exception.
            binary_ops.append(
                'op_custom1(a, b) = (z = exp(a) + inv(b); '
                'isfinite(z) && z > zero(z) ? log(z) : oftype(z, NaN))')
            import sympy as sp
            extra_sympy['op_custom1'] = lambda a, b: sp.log(sp.exp(a) + 1 / b)
        else:
            binary_ops.append(op)
    if not binary_ops:
        binary_ops = ['+', '-', '*', '/']

    unary_ops: List[str] = []
    if unary_in:
        import sympy as sp
        for op in unary_in:
            op = str(op).strip()
            if not op:
                continue
            low = op.lower()
            if low in {'inv', 'inverse'}:
                unary_ops.append('inv(x) = 1 / x')
                extra_sympy['inv'] = lambda x: 1 / x
            elif low in {'square', 'sqr'}:
                unary_ops.append('square(x) = x^2')
                extra_sympy['square'] = lambda x: x**2
            elif low in {'cube'}:
                unary_ops.append('cube(x) = x^3')
                extra_sympy['cube'] = lambda x: x**3
            elif low in {'sqrt', 'sqrt_abs', 'protected_sqrt'}:
                # Protected square root is more robust for broad SR search on
                # domains that include negative intermediate values.
                unary_ops.append('sqrt_abs(x) = sqrt(abs(x))')
                extra_sympy['sqrt_abs'] = lambda x: sp.sqrt(sp.Abs(x))
            elif low in {'smoothsat', 'smooth_sat', 'softsat'} or low.startswith('smoothsat('):
                # Generic bounded odd saturation, reusable by system-ID tasks.
                # one(x) keeps the Julia definition type-stable for Float32/64.
                unary_ops.append('smoothsat(x) = x / sqrt(one(x) + x^2)')
                extra_sympy['smoothsat'] = lambda x: x / sp.sqrt(1 + x**2)
            elif low in {'exp', 'sin', 'cos', 'log', 'abs'}:
                unary_ops.append(low)
            else:
                unary_ops.append(op)
    return binary_ops, unary_ops, extra_sympy


def _expression_text(value: Any) -> str:
    """Serialize a PySR/SymPy expression without truncating fitted constants."""
    if isinstance(value, str):
        return value
    try:
        import sympy as sp
        if isinstance(value, sp.Basic):
            # PySR commonly stores fitted constants as Float32-valued SymPy
            # Floats. Default string conversion may print too few digits for
            # a round trip. Expand only floating atoms to 17 decimal digits
            # while preserving exact integers, rationals, pi, and operators.
            replacements = {
                atom: sp.Float(float(atom), 17)
                for atom in value.atoms(sp.Float)
            }
            value = value.xreplace(replacements) if replacements else value
            return sp.sstr(value, full_prec=True)
    except Exception:
        pass
    return str(value)


def _safe_get_best(model: Any) -> Dict[str, Any]:
    try:
        best = model.get_best()
        if hasattr(best, 'to_dict'):
            best = best.to_dict()
        if isinstance(best, dict):
            return best
    except Exception:
        pass
    return {}


def _best_expression(model: Any, best: Dict[str, Any]) -> str:
    for key in ('sympy_format', 'equation', 'lambda_format'):
        if key in best:
            return _expression_text(best[key])
    try:
        return str(model.sympy())
    except Exception:
        pass
    try:
        return str(model.get_best().get('equation'))
    except Exception:
        pass
    return '<expression unavailable>'


def _complexity(best: Dict[str, Any]) -> float:
    for key in ('complexity', 'size'):
        if key in best:
            try:
                return float(best[key])
            except Exception:
                pass
    return float('nan')



def _top_equations_from_model(model: Any, config: Dict[str, Any], rank_mode: str = 'loss') -> List[Dict[str, Any]]:
    """Extract top equations from the PySR equation table for MATLAB reporting.

    This does not change PySR model selection.  It only exposes the equation
    archive/Pareto table so the user can inspect alternative SR structures.

    rank_mode='loss' reports the lowest-loss expressions after the configured
    near-best-loss and max-complexity filters.  rank_mode='score' reports the
    highest-score expressions after the max-complexity filter; it intentionally
    does not apply the near-best-loss filter so a compact high-score structure
    is not hidden by a slightly lower-loss expression.
    """
    topk = int(config.get('top_k_expressions_to_report', 10) or 0)
    if topk <= 0:
        return []
    try:
        df = model.equations_.copy()
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []

    rank_mode = str(rank_mode or 'loss').lower().strip()

    max_complexity = config.get('max_report_complexity', None)
    try:
        max_complexity = float(max_complexity)
    except Exception:
        max_complexity = None
    if max_complexity is not None and np.isfinite(max_complexity) and 'complexity' in df.columns:
        df = df[df['complexity'].astype(float) <= max_complexity]

    if rank_mode == 'loss' and 'loss' in df.columns:
        try:
            best_loss = float(np.nanmin(df['loss'].to_numpy(dtype=float)))
        except Exception:
            best_loss = np.nan
        loss_multiplier = config.get('equation_loss_multiplier', None)
        try:
            loss_multiplier = float(loss_multiplier)
        except Exception:
            loss_multiplier = None
        if loss_multiplier is not None and np.isfinite(loss_multiplier) and np.isfinite(best_loss) and best_loss >= 0:
            # Use a tiny floor so exact-loss cases still report several simple
            # alternatives if their loss is close to numerical precision.
            threshold = max(best_loss * loss_multiplier, best_loss + 1e-300)
            df = df[df['loss'].astype(float) <= threshold]

    sort_cols: List[str] = []
    ascending: List[bool] = []
    if rank_mode == 'score':
        if 'score' in df.columns:
            sort_cols.append('score')
            ascending.append(False)
        if 'loss' in df.columns:
            sort_cols.append('loss')
            ascending.append(True)
        if 'complexity' in df.columns:
            sort_cols.append('complexity')
            ascending.append(True)
    else:
        if 'loss' in df.columns:
            sort_cols.append('loss')
            ascending.append(True)
        if 'complexity' in df.columns:
            sort_cols.append('complexity')
            ascending.append(True)
        if 'score' in df.columns:
            sort_cols.append('score')
            ascending.append(False)
    if sort_cols:
        try:
            df = df.sort_values(sort_cols, ascending=ascending)
        except Exception:
            pass

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if len(records) >= topk:
            break
        expr = None
        for key in ('sympy_format', 'equation'):
            if key in row and row[key] is not None:
                expr = _expression_text(row[key])
                break
        if expr is None:
            expr = str(row.to_dict())
        expr = _canonical_display_expression(expr, config)
        if not _expression_respects_operator_occurrence_limits(expr, config):
            continue
        rank = len(records) + 1
        item: Dict[str, Any] = {'rank': rank, 'rank_mode': rank_mode, 'expression': expr}
        for key in ('complexity', 'loss', 'score'):
            if key in row:
                try:
                    val = float(row[key])
                    item[key] = val if np.isfinite(val) else str(row[key])
                except Exception:
                    item[key] = str(row[key])
        records.append(item)
    return records

def _predict_equation_row(model: Any, X: np.ndarray, row_label: Any, row: Any) -> np.ndarray:
    """Evaluate one equation-table row without changing the fitted search."""
    if X.size == 0:
        return np.zeros((0, 1), dtype=float)
    try:
        fn = row.get('lambda_format', None)
    except Exception:
        fn = None
    if callable(fn):
        try:
            y = fn(X)
            return np.asarray(y, dtype=float).reshape(-1, 1)
        except Exception:
            pass
    for idx in (row_label, int(row_label) if isinstance(row_label, (int, np.integer)) else None):
        if idx is None:
            continue
        try:
            y = model.predict(X, index=idx)
            return np.asarray(y, dtype=float).reshape(-1, 1)
        except Exception:
            pass
    raise RuntimeError(f'Unable to evaluate PySR equation row {row_label!r}.')


def _mse(yhat: np.ndarray, y: np.ndarray) -> float:
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if yhat.size != y.size or y.size == 0 or not np.all(np.isfinite(yhat)):
        return float('inf')
    return float(np.mean((yhat - y) ** 2))


def _equation_expression(row: Any) -> str:
    # Preserve the native searched tree for reports and structure signatures.
    for key in ('equation', 'sympy_format'):
        try:
            value = row.get(key, None)
        except Exception:
            value = None
        if value is not None:
            return _expression_text(value)
    return '<expression unavailable>'


def _real_input_sympy_expression(value: Any, config: Dict[str, Any]) -> Any:
    """Canonicalize a SymPy expression under the real-valued input contract.

    PySR evaluates the supplied floating-point feature matrix over the reals.
    Some SymPy export paths nevertheless create symbols without ``real=True``.
    For example, ``sqrt_abs(exp(x1))`` can then be serialized as
    ``exp(re(x1)/2)``.  The two forms are identical for our real input
    variables, but the latter leaks the bookkeeping operator ``re`` into the
    MATLAB expression compiler.  Rebuild only known input symbols with a real
    assumption and ask SymPy to refine assumption-dependent wrappers.  This
    changes no fitted constants or searched tree semantics.
    """
    try:
        import sympy as sp
        if not isinstance(value, sp.Basic):
            return value
        configured = {str(name) for name in (config.get('variable_names') or [])}
        replacements = {}
        for symbol in value.free_symbols:
            name = str(symbol)
            if name in configured or re.fullmatch(r'x\d+', name):
                replacements[symbol] = sp.Symbol(name, real=True)
        if replacements:
            value = value.xreplace(replacements)
            value = sp.refine(value)
        return value
    except Exception:
        return value


def _expand_known_custom_operator_expression(value: Any, config: Dict[str, Any]) -> Any:
    """Expand supported atomic SR motifs before MATLAB/PhDN compilation."""
    text = str(value)
    known_names = ('op_custom1', 'smoothsat')
    if not any(name in text for name in known_names):
        return value
    try:
        import sympy as sp
        variable_names = config.get('variable_names') or []
        local_dict: Dict[str, Any] = {
            str(name): sp.Symbol(str(name), real=True) for name in variable_names
        }
        local_dict.update({
            'op_custom1': lambda a, b: sp.log(sp.exp(a) + 1 / b),
            'smoothsat': lambda x: x / sp.sqrt(1 + x**2),
            'inv': lambda x: 1 / x,
            'square': lambda x: x**2,
            'cube': lambda x: x**3,
            'sqrt_abs': lambda x: sp.sqrt(sp.Abs(x)),
        })
        return sp.sympify(text.replace('^', '**'), locals=local_dict)
    except Exception:
        return value


_REAL_VARIABLE_WRAPPER_RE = re.compile(
    r'\b(?:re|real|conjugate|conj)\(\s*(x\d+)\s*\)', re.IGNORECASE)
_IMAG_REAL_VARIABLE_RE = re.compile(r'\b(?:im|imag)\(\s*(x\d+)\s*\)', re.IGNORECASE)


def _canonicalize_real_input_wrappers(text: str) -> str:
    """Remove scalar complex wrappers that are identities on real features."""
    text = _REAL_VARIABLE_WRAPPER_RE.sub(r'\1', str(text))
    return _IMAG_REAL_VARIABLE_RE.sub('0', text)


def _compiler_equation_expression(row: Any, config: Dict[str, Any]) -> str:
    """Return a physical-variable compiler expression with full precision."""
    for key in ('sympy_format', 'equation'):
        try:
            value = row.get(key, None)
        except Exception:
            value = None
        if value is not None:
            canonical_text = _canonical_display_expression(
                _expression_text(value), config)
            value = _expand_known_custom_operator_expression(
                canonical_text, config)
            value = _real_input_sympy_expression(value, config)
            text = _canonicalize_real_input_wrappers(_expression_text(value))
            remaining_custom = re.search(
                r'\b(?:op_custom1|smoothsat)\s*\(', text)
            if remaining_custom:
                raise RuntimeError(
                    f'{remaining_custom.group(0).split("(")[0]} remained atomic in the '
                    'compiler expression; the PySR SymPy expansion could not be constructed.')
            _assert_no_internal_generator_feature_tokens(text, config)
            return text
    return '<expression unavailable>'


_VARIABLE_TOKEN_RE = re.compile(r'(?<![A-Za-z_0-9])x(\d+)(?![A-Za-z_0-9])')


def _canonical_display_expression(expression: str, config: Dict[str, Any]) -> str:
    """Restore PySR feature columns to physical-variable expressions."""
    text = str(expression)
    names = list(config.get('variable_names') or [])
    generic_zero_based = bool(
        config.get('_pysr_generic_zero_based_feature_names', False))
    tokens = [int(m.group(1)) for m in _VARIABLE_TOKEN_RE.finditer(text)]
    if generic_zero_based or 0 in tokens:
        def replace(match: re.Match) -> str:
            index = int(match.group(1))
            if 0 <= index < len(names):
                return str(names[index])
            return f'x{index + 1}'
        text = _VARIABLE_TOKEN_RE.sub(replace, text)

    text = re.sub(
        r'(?<![A-Za-z0-9_])sin_x1_atom(?![A-Za-z0-9_])',
        'sin(x1)', text)
    text = re.sub(
        r'(?<![A-Za-z0-9_])cos_x1_atom(?![A-Za-z0-9_])',
        'cos(x1)', text)
    _assert_no_internal_generator_feature_tokens(text, config)
    return text


def _assert_no_internal_generator_feature_tokens(
        expression: str, config: Dict[str, Any]) -> None:
    """Fail early if typed-generator auxiliary columns escape the adapter."""
    if not _single_generator_typed_prior_enabled(config):
        return
    text = str(expression)
    leaked = re.search(
        r'(?<![A-Za-z0-9_])(?:sin_x1_atom|cos_x1_atom)'
        r'(?![A-Za-z0-9_])',
        text)
    if leaked:
        raise RuntimeError(
            'Internal typed-generator feature leaked into exported expression: '
            f'{leaked.group(0)}')
    bad_index = [
        int(m.group(1)) for m in _VARIABLE_TOKEN_RE.finditer(text)
        if int(m.group(1)) > 4
    ]
    if bad_index:
        raise RuntimeError(
            'Internal PySR feature index leaked into exported generator '
            f'expression: x{min(bad_index)}')


_NUMBER_TOKEN_RE = re.compile(r'(?<![A-Za-z_])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
_STRUCTURE_TOKEN_RE = re.compile(r'x\d+|[A-Za-z_][A-Za-z_0-9]*|\*\*|[+\-*/()]')


def _structure_signature(expression: str) -> str:
    """Return a constant-free expression skeleton for restart matching."""
    text = _NUMBER_TOKEN_RE.sub('#', str(expression).replace(' ', ''))
    return text.lower()


def _structure_ngrams(signature: str) -> set:
    tokens = _STRUCTURE_TOKEN_RE.findall(signature)
    grams = {f'u:{token}' for token in tokens}
    grams.update(f'b:{tokens[i]}>{tokens[i+1]}' for i in range(len(tokens) - 1))
    grams.update(f't:{tokens[i]}>{tokens[i+1]}>{tokens[i+2]}' for i in range(len(tokens) - 2))
    return grams


def _structure_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ga = a.get('_structure_ngrams', set())
    gb = b.get('_structure_ngrams', set())
    union = ga | gb
    if not union:
        return 0.0
    return 1.0 - len(ga & gb) / len(union)


def _percentile_scores(values: List[float]) -> np.ndarray:
    """Map larger-is-better values to stable tie-aware mid-rank percentiles.

    The former unique-value rank mapped a one-element pool to 1 and a
    two-element pool to the extreme pair 0/1.  Those endpoints express much
    more confidence than a tiny candidate pool contains.  The empirical-CDF
    mid-rank plotting position, (rank - 0.5) / n, instead maps one finite
    value to the neutral 0.5 and two distinct values to 0.25/0.75.  Ties use
    their average observation rank rather than their rank among unique values.
    """
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.size, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    finite_values = arr[finite]
    n_finite = int(finite_values.size)
    for value in np.unique(finite_values):
        n_less = int(np.sum(finite_values < value))
        n_equal = int(np.sum(finite_values == value))
        average_rank = n_less + 0.5 * (n_equal + 1.0)  # one-based mid-rank
        out[arr == value] = (average_rank - 0.5) / float(n_finite)
    return out


def _soft_validation_score(validation_mse: float, best_validation_mse: float,
                           validation_multiplier: float) -> float:
    """Continuous validation evidence inside the hard rho-qualified pool.

    The best candidate receives 1 and a candidate at rho times the best MSE
    receives 0, with linear decay in log-error ratio.  The hard rho threshold
    remains unchanged; this term only distinguishes candidates that passed it.
    """
    mse = float(validation_mse)
    best = float(best_validation_mse)
    rho = float(validation_multiplier)
    if not np.isfinite(mse) or mse < 0.0:
        return 0.0
    if best <= np.finfo(float).tiny:
        return 1.0 if mse <= np.finfo(float).tiny else 0.0
    if not np.isfinite(rho) or rho <= 1.0:
        return 1.0 if mse <= best else 0.0
    ratio = max(1.0, mse / best)
    return float(np.clip(1.0 - np.log(ratio) / np.log(rho), 0.0, 1.0))


def _selection_mse_floor(target: np.ndarray, config: Dict[str, Any]) -> float:
    """Return a scale-aware numerical tie floor for structure selection.

    Validation differences below this floor are not treated as scientific
    evidence for a more complicated tree.  The raw MSE is still reported, but
    candidate selection uses the floored MSE and prefers lower complexity when
    the effective errors are tied.
    """
    try:
        absolute_floor = float(
            config.get('structure_machine_error_abs_mse_floor', 0.0))
    except Exception:
        absolute_floor = 0.0
    try:
        relative_floor = float(
            config.get('structure_machine_error_rel_mse_floor', 0.0))
    except Exception:
        relative_floor = 0.0
    if not np.isfinite(absolute_floor) or absolute_floor < 0.0:
        absolute_floor = 0.0
    if not np.isfinite(relative_floor) or relative_floor < 0.0:
        relative_floor = 0.0
    y = np.asarray(target, dtype=float).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size:
        target_energy = float(np.mean(finite * finite))
    else:
        target_energy = 0.0
    scale = max(1.0, target_energy) if np.isfinite(target_energy) else 1.0
    return float(max(absolute_floor, relative_floor * scale, np.finfo(float).tiny))


def _score_structure_candidates(records: List[Dict[str, Any]], config: Dict[str, Any],
                                selection_mse_floor: float) -> List[int]:
    """Score near-best-validation candidates without new fits or evaluations.

    Simplification and expansion neighborhoods are induced by structurally
    nearby candidates already present on the exported Pareto table.  This
    reuses their external-validation predictions, so the scoring overhead is
    only O(K^2) string/set work for the validation-ratio-qualified pool.
    """
    if not records:
        return []
    validation_multiplier = float(config.get('structure_validation_multiplier', 4.0) or 4.0)
    if not np.isfinite(validation_multiplier) or validation_multiplier < 1.0:
        validation_multiplier = 4.0
    max_distance = float(config.get('structure_neighborhood_max_distance', 0.55) or 0.55)
    min_distance = float(config.get('structure_neighborhood_min_distance', 0.10) or 0.10)
    min_distance = min(max(min_distance, np.finfo(float).eps), max_distance)
    complexity_window = float(config.get('structure_neighborhood_complexity_window', 8.0) or 8.0)
    frontier_clip = float(config.get('structure_frontier_max_abs', 20.0) or 20.0)
    if not np.isfinite(frontier_clip) or frontier_clip <= 0:
        frontier_clip = 20.0
    eps = np.finfo(float).tiny
    selection_mse_floor = max(float(selection_mse_floor), eps)
    for rec in records:
        raw_mse = float(rec['validation_mse'])
        rec['selection_mse_floor'] = selection_mse_floor
        rec['validation_mse_effective'] = max(raw_mse, selection_mse_floor)
        rec['validation_floor_tied'] = bool(raw_mse <= selection_mse_floor)

    # Below the numerical floor, validation errors are treated as tied and the
    # simpler structure is preferred. Raw MSE remains the final deterministic
    # tie-breaker only after effective MSE and complexity.
    validation_order = sorted(
        range(len(records)),
        key=lambda i: (records[i]['validation_mse_effective'],
                       records[i]['complexity'], records[i]['validation_mse']))
    best_validation = float(records[validation_order[0]]['validation_mse_effective'])
    validation_limit = max(best_validation * validation_multiplier,
                           selection_mse_floor)
    eligible = [i for i in validation_order
                if float(records[i]['validation_mse_effective']) <= validation_limit]
    if not eligible:
        eligible = [validation_order[0]]

    for rec in records:
        signature = _structure_signature(rec.get('expression', ''))
        rec['structure_signature'] = signature
        rec['_structure_ngrams'] = _structure_ngrams(signature)
        rec['structure_eligible'] = False
        rec['frontier_score_raw'] = 0.0
        rec['frontier_score_valid'] = False
        rec['frontier_score_normalized'] = 0.0
        rec['validation_soft_score'] = 0.0
        rec['structure_score'] = -1e300

    # Validation Pareto representatives, one lowest-error row per complexity.
    by_complexity: Dict[float, int] = {}
    for i in validation_order:
        c = float(records[i]['complexity'])
        if c not in by_complexity:
            by_complexity[c] = i
    frontier: List[int] = []
    best_seen = float('inf')
    for c in sorted(by_complexity):
        i = by_complexity[c]
        e = float(records[i]['validation_mse_effective'])
        if e < best_seen:
            frontier.append(i)
            best_seen = e

    # One two-sided, multi-scale frontier-prominence statistic replaces the
    # partly redundant Slocal and K terms.  It asks whether the candidate lies
    # below an interpolation of *both* simpler and more-complex validation
    # neighbors, then normalizes the prominence by a robust local log-loss
    # scale.  Endpoints have no valid two-sided evidence.
    frontier_values: List[float] = []
    for i in eligible:
        rec = records[i]
        rec['structure_eligible'] = True
        ci = float(rec['complexity'])
        ei = max(float(rec['validation_mse_effective']), eps)
        zi = np.log(ei)
        scale_scores: List[float] = []
        for radius_fraction in (0.50, 0.75, 1.00):
            radius = max(1.0, complexity_window * radius_fraction)
            left = []
            right = []
            for j in frontier:
                if i == j:
                    continue
                cj = float(records[j]['complexity'])
                if abs(cj - ci) > radius:
                    continue
                distance = _structure_distance(rec, records[j])
                if distance < min_distance or distance > max_distance:
                    continue
                item = (cj, np.log(max(float(records[j]['validation_mse_effective']), eps)))
                if cj < ci:
                    left.append(item)
                elif cj > ci:
                    right.append(item)
            if not left or not right:
                continue
            cl = float(np.median([x[0] for x in left]))
            cr = float(np.median([x[0] for x in right]))
            zl = float(np.median([x[1] for x in left]))
            zr = float(np.median([x[1] for x in right]))
            if not (cl < ci < cr):
                continue
            zref = zl + (zr - zl) * (ci - cl) / max(cr - cl, eps)
            neighborhood_z = np.asarray([x[1] for x in left + right] + [zi], dtype=float)
            med = float(np.median(neighborhood_z))
            mad = float(np.median(np.abs(neighborhood_z - med)))
            robust_scale = max(1.4826 * mad, 0.05)
            scale_scores.append(float(np.clip((zref - zi) / robust_scale,
                                              -frontier_clip, frontier_clip)))
        valid = bool(scale_scores)
        if valid:
            persistence = float(np.mean(np.asarray(scale_scores) > 0.0))
            raw = float(np.median(scale_scores) * (0.5 + 0.5 * persistence))
            raw = float(np.clip(raw, -frontier_clip, frontier_clip))
        else:
            raw = float('nan')
        rec['frontier_score_raw'] = raw if np.isfinite(raw) else 0.0
        rec['frontier_score_valid'] = valid
        frontier_values.append(raw)

    frontier_normalized = _robust_unit_scores(frontier_values)
    w_validation = float(config.get('structure_validation_weight', 0.20) or 0.0)
    if not np.isfinite(w_validation):
        w_validation = 0.20
    w_validation = float(np.clip(w_validation, 0.0, 1.0))
    has_two_sided_frontier = any(np.isfinite(value) for value in frontier_values)
    for pos, i in enumerate(eligible):
        rec = records[i]
        rec['frontier_score_normalized'] = float(frontier_normalized[pos])
        rec['validation_soft_score'] = _soft_validation_score(
            rec['validation_mse_effective'], best_validation, validation_multiplier)
        if has_two_sided_frontier and not bool(rec.get('frontier_score_valid', False)):
            # A frontier endpoint has no two-sided evidence. If at least one
            # interior candidate exists, a missing side cannot become a free
            # zero-slope advantage.
            rec['structure_score'] = -1e300
        else:
            rec['structure_score'] = float(
                (1.0 - w_validation) * frontier_normalized[pos]
                + w_validation * rec['validation_soft_score'])
        rec.pop('_structure_ngrams', None)
    for rec in records:
        rec.pop('_structure_ngrams', None)
    return sorted(
        eligible,
        key=lambda i: (-float(records[i]['structure_score']),
                       records[i]['validation_mse_effective'],
                       records[i]['complexity'],
                       records[i]['validation_mse']))


def _choose_structure_core_index(
        records: List[Dict[str, Any]], structure_order: List[int],
        config: Dict[str, Any], selection_mse_floor: float
        ) -> Tuple[int, str]:
    """Choose the Stage-0 core with an explicit machine-floor tie rule."""
    if not records:
        raise ValueError('Cannot select a Stage-0 core from an empty record list.')

    selection_policy = str(config.get('selection_policy', 'pysr_native_score')).strip().lower()
    native_score_policy = selection_policy in {
        'pysr_native_score', 'native_pysr_score', 'original_pysr_score', 'original_score'
    }
    if native_score_policy:
        # Clean ablation: use PySR's exported native score directly.  Do not
        # apply the PhDN numerical-floor simplicity rule, external-validation
        # composite score, or structure-frontier score before this decision.
        finite_score_indices: List[int] = []
        for i, rec in enumerate(records):
            try:
                score = float(rec.get('score', np.nan))
            except Exception:
                score = np.nan
            if np.isfinite(score):
                finite_score_indices.append(i)
        if finite_score_indices:
            core_idx = min(
                finite_score_indices,
                key=lambda i: (-float(records[i]['score']),
                               float(records[i].get('validation_mse', float('inf'))),
                               float(records[i].get('complexity', float('inf')))))
            return core_idx, 'maximum_native_pysr_score'
        core_idx = min(
            range(len(records)),
            key=lambda i: (float(records[i].get('validation_mse', float('inf'))),
                           float(records[i].get('complexity', float('inf')))))
        return core_idx, 'external_validation_fallback_no_finite_native_score'

    machine_floor_indices = [
        i for i, rec in enumerate(records)
        if float(rec.get('validation_mse', float('inf'))) <= selection_mse_floor
    ]
    if machine_floor_indices:
        core_idx = min(
            machine_floor_indices,
            key=lambda i: (float(records[i].get('complexity', float('inf'))),
                           float(records[i].get('validation_mse', float('inf')))))
        return core_idx, 'machine_floor_simplicity_tie'

    structure_enabled = bool(config.get('structure_score_enable', False))
    if structure_enabled and structure_order:
        return structure_order[0], 'machine_floor_aware_soft_validation_structure_score'

    finite_score_indices: List[int] = []
    for i, rec in enumerate(records):
        try:
            score = float(rec.get('score', np.nan))
        except Exception:
            score = np.nan
        if np.isfinite(score):
            finite_score_indices.append(i)
    if finite_score_indices:
        core_idx = min(
            finite_score_indices,
            key=lambda i: (-float(records[i]['score']),
                           float(records[i].get('validation_mse', float('inf'))),
                           float(records[i].get('complexity', float('inf')))))
        return core_idx, 'maximum_native_pysr_score'

    core_idx = min(
        range(len(records)),
        key=lambda i: (float(records[i].get('validation_mse', float('inf'))),
                       float(records[i].get('complexity', float('inf')))))
    return core_idx, 'external_validation_fallback_no_finite_score'


def _robust_unit_scores(values: List[float]) -> np.ndarray:
    """Map finite evidence to (0, 1) without tiny-pool percentile extremes."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=float)
    finite = np.isfinite(arr)
    x = arr[finite]
    if x.size == 0:
        return out
    if x.size == 1:
        out[finite] = 0.5
        return out
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        out[finite] = 0.5
        return out
    z = np.clip((x - center) / scale, -8.0, 8.0)
    out[finite] = 1.0 / (1.0 + np.exp(-z))
    return out



def _make_external_candidate_report(records: List[Dict[str, Any]], validation_best_idx: int,
                                    core_idx: int, topk: int, rank_mode: str,
                                    selected_role: str = 'restart-local-core') -> List[Dict[str, Any]]:
    """Build a display-only ranking of semantically unique Pareto candidates.

    The tables use external validation MSE, structural complexity, native
    PySR score, or the low-cost PhDN structure score.
    """
    if topk <= 0 or not records:
        return []
    if rank_mode == 'complexity':
        order = sorted(range(len(records)),
                       key=lambda i: (records[i]['complexity'], records[i]['validation_mse']))
    elif rank_mode == 'best_score':
        def score_key(i: int) -> Tuple[float, float, float]:
            raw = records[i].get('score', float('-inf'))
            try:
                score = float(raw)
            except Exception:
                score = float('-inf')
            if not np.isfinite(score):
                score = float('-inf')
            return (-score, records[i]['validation_mse'], records[i]['complexity'])
        order = sorted(range(len(records)), key=score_key)
    elif rank_mode == 'structure_score':
        eligible = [i for i, rec in enumerate(records)
                    if bool(rec.get('structure_eligible', False))
                    and float(rec.get('structure_score', -1e300)) > -1e299]
        order = sorted(eligible, key=lambda i: (
            -float(records[i].get('structure_score', float('-inf'))),
            records[i].get('validation_mse_effective', records[i]['validation_mse']),
            records[i]['complexity'], records[i]['validation_mse']))
    else:
        order = sorted(range(len(records)),
                       key=lambda i: (records[i]['validation_mse'], records[i]['complexity']))
    best_val_raw = float(records[validation_best_idx]['validation_mse'])
    best_val_den = max(best_val_raw, np.finfo(float).tiny)
    report: List[Dict[str, Any]] = []
    for rank, i in enumerate(order[:topk], start=1):
        rec = dict(records[i])
        rec['rank'] = rank
        rec['rank_mode'] = rank_mode
        value = float(rec['validation_mse'])
        if best_val_raw <= np.finfo(float).tiny:
            rec['relative_to_best_validation_mse'] = 1.0 if value <= np.finfo(float).tiny else float('inf')
        else:
            rec['relative_to_best_validation_mse'] = value / best_val_den
        rec['selection_role'] = selected_role if i == core_idx else 'none'
        rec['ranking_scope'] = 'restart_local'
        rec['relative_error_scope'] = 'restart_local_validation_best'
        report.append(rec)
    return report


def _single_generator_typed_prior_enabled(config: Dict[str, Any]) -> bool:
    mode = str(config.get('typed_physical_constraints', '') or '').strip().lower()
    return mode in {'single_generator_dynamic', 'smib_avr', 'generator_voltage_typed'}


def _parse_atomic_expression_for_constraints(expression: str, config: Dict[str, Any]) -> Any:
    """Parse an SR expression for case-local physical admissibility checks."""
    try:
        import sympy as sp
        variable_names = config.get('variable_names') or []
        local_dict: Dict[str, Any] = {
            str(name): sp.Symbol(str(name), real=True) for name in variable_names
        }
        local_dict.update({
            'inv': lambda x: 1 / x,
            'square': lambda x: x**2,
            'cube': lambda x: x**3,
            'sqrt_abs': lambda x: sp.sqrt(sp.Abs(x)),
        })
        return sp.sympify(str(expression).replace('^', '**'), locals=local_dict)
    except Exception:
        return None



def _expression_respects_single_generator_typed_prior(expression: str, config: Dict[str, Any]) -> bool:
    """Case-local admissibility rules for the known SMIB state identities.

    Rules:
      1. every sin/cos argument may depend only on x1;
      2. no state variable may occur in a denominator.
    """
    if not _single_generator_typed_prior_enabled(config):
        return True
    try:
        import sympy as sp
        expr = _parse_atomic_expression_for_constraints(expression, config)
        if expr is None:
            return False
        names = config.get('variable_names') or ['x1', 'x2', 'x3', 'x4']
        symbols = {str(name): sp.Symbol(str(name), real=True) for name in names}
        x1 = symbols.get('x1', sp.Symbol('x1', real=True))
        state_symbols = set(symbols.values())

        # Rule 1: trigonometric arguments contain no state except x1.
        for node in sp.preorder_traversal(expr):
            if node.func in (sp.sin, sp.cos):
                state_deps = set(node.args[0].free_symbols).intersection(state_symbols)
                if not state_deps.issubset({x1}):
                    return False

        # Rule 2: after rational combination, the denominator must be state-free.
        _, denominator = sp.fraction(sp.together(expr))
        if set(denominator.free_symbols).intersection(state_symbols):
            return False
        return True
    except Exception:
        return False

def _operator_occurrence_limits(config: Dict[str, Any]) -> Dict[str, int]:
    """Return nonnegative case-local total operator occurrence limits."""
    raw = config.get('max_operator_occurrences', {}) or {}
    limits: Dict[str, int] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            try:
                limit = int(value)
            except Exception:
                continue
            if limit >= 0:
                limits[str(name).strip().lower()] = limit
    return limits


def _expression_respects_operator_occurrence_limits(expression: str, config: Dict[str, Any]) -> bool:
    """Reject sibling repeats not covered by PySR nested_constraints."""
    limits = _operator_occurrence_limits(config)
    if not limits:
        return True
    text = str(expression or '')
    for name, limit in limits.items():
        pattern = r'(?<![A-Za-z0-9_])' + re.escape(name) + r'\s*\('
        if len(re.findall(pattern, text, flags=re.IGNORECASE)) > limit:
            return False
    return True


def _external_validation_selection(model: Any, Xtr: np.ndarray, ytr: np.ndarray,
                                   Xval: np.ndarray, yval: np.ndarray,
                                   Xte: np.ndarray, Xood: np.ndarray,
                                   config: Dict[str, Any]) -> Tuple[
                                       Dict[str, Any], List[Dict[str, Any]],
                                       List[Dict[str, Any]], List[Dict[str, Any]],
                                       List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Select one Stage-0 core and export its low-cost structure candidate pool."""
    try:
        df = model.equations_.copy()
    except Exception as exc:
        raise RuntimeError('PySR did not expose an equation/Pareto table.') from exc
    if df is None or len(df) == 0:
        raise RuntimeError('PySR equation/Pareto table is empty.')

    records: List[Dict[str, Any]] = []
    predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for position, (row_label, row) in enumerate(df.iterrows()):
        try:
            ytr_hat = _predict_equation_row(model, Xtr, position, row)
            yval_hat = _predict_equation_row(model, Xval, position, row)
            yte_hat = _predict_equation_row(model, Xte, position, row)
            yood_hat = _predict_equation_row(model, Xood, position, row) if Xood.size else np.zeros((0, 1))
        except Exception:
            continue
        train_mse = _mse(ytr_hat, ytr)
        val_mse = _mse(yval_hat, yval)
        if not np.isfinite(val_mse):
            continue
        try:
            complexity = float(row.get('complexity', np.nan))
        except Exception:
            complexity = float('nan')
        if not np.isfinite(complexity):
            complexity = float(10**9)
        try:
            display_expression = _canonical_display_expression(
                _equation_expression(row), config)
            compiler_expression = _compiler_equation_expression(row, config)
        except Exception:
            continue
        if not _expression_respects_operator_occurrence_limits(display_expression, config):
            continue
        if not _expression_respects_single_generator_typed_prior(display_expression, config):
            continue
        rec: Dict[str, Any] = {
            'table_position': int(position),
            'table_index': int(row_label) if isinstance(row_label, (int, np.integer)) else str(row_label),
            'expression': display_expression,
            'compiler_expression': compiler_expression,
            'complexity': complexity,
            'train_mse': train_mse,
            'validation_mse': val_mse,
            'loss': 1e300,
            'score': -1e300,
        }
        for key in ('loss', 'score'):
            try:
                value = float(row.get(key, np.nan))
                if np.isfinite(value):
                    rec[key] = value
            except Exception:
                pass
        records.append(rec)
        predictions.append((ytr_hat, yval_hat, yte_hat, yood_hat))

    if not records:
        raise RuntimeError('No finite PySR equation could be evaluated on the external validation set.')

    # Numerical semantic de-duplication keeps the lower-validation/lower-complexity
    # representative of numerically equivalent candidates.  It changes neither
    # the represented function or the later structure-score policy among
    # distinct functions.
    tol = float(config.get('semantic_dedup_tolerance', 1e-8) or 0.0)
    order = sorted(range(len(records)), key=lambda i: (records[i]['validation_mse'], records[i]['complexity']))
    keep: List[int] = []
    for i in order:
        pi = np.vstack((predictions[i][0], predictions[i][1])).reshape(-1)
        duplicate = False
        for j in keep:
            pj = np.vstack((predictions[j][0], predictions[j][1])).reshape(-1)
            scale = max(1.0, float(np.max(np.abs(pi))), float(np.max(np.abs(pj))))
            if pi.shape == pj.shape and np.max(np.abs(pi - pj)) <= tol * scale:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    records = [records[i] for i in keep]
    predictions = [predictions[i] for i in keep]

    selection_mse_floor = _selection_mse_floor(yval, config)
    structure_order = _score_structure_candidates(
        records, config, selection_mse_floor)

    validation_best_idx = min(
        range(len(records)),
        key=lambda i: (records[i]['validation_mse'], records[i]['complexity']))
    core_idx, selection_rule = _choose_structure_core_index(
        records, structure_order, config, selection_mse_floor)

    def selected(i: int) -> Dict[str, Any]:
        rec = dict(records[i])
        ytr_hat, yval_hat, yte_hat, yood_hat = predictions[i]
        rec['_predictions'] = (ytr_hat, yval_hat, yte_hat, yood_hat)
        rec['selection_role'] = 'restart-local-core'
        rec['selection_rule'] = selection_rule
        rec['selection_mse_floor'] = selection_mse_floor
        rec['selection_prefers_simplicity_within_floor'] = True
        rec['ranking_scope'] = 'restart_local'
        rec['relative_error_scope'] = 'restart_local_validation_best'
        return rec

    topk = int(config.get('top_k_expressions_to_report', 10) or 10)
    score_order = sorted(
        range(len(records)),
        key=lambda i: (-float(records[i].get('score', float('-inf')))
                       if np.isfinite(float(records[i].get('score', float('-inf')))) else float('inf'),
                       records[i]['validation_mse'], records[i]['complexity']))

    ranking_topk = int(config.get('candidate_ranking_top_k', 20) or 0)
    validation_report = _make_external_candidate_report(
        records, validation_best_idx, core_idx, ranking_topk, 'validation_mse',
        'restart-local-core')
    complexity_report = _make_external_candidate_report(
        records, validation_best_idx, core_idx, ranking_topk, 'complexity',
        'restart-local-core')
    best_score_report = _make_external_candidate_report(
        records, validation_best_idx, core_idx, ranking_topk, 'best_score',
        'restart-local-core')
    structure_score_report = _make_external_candidate_report(
        records, validation_best_idx, core_idx, ranking_topk, 'structure_score',
        'restart-local-core')
    if selection_rule == 'machine_floor_simplicity_tie':
        # Make the selection report reflect the actual floor-aware decision,
        # rather than showing a different structure-score row as rank one.
        structure_score_report = sorted(
            structure_score_report,
            key=lambda rec: (0 if rec.get('selection_role') == 'restart-local-core' else 1,
                             rec.get('complexity', float('inf')),
                             rec.get('validation_mse', float('inf'))))
        for rank, rec in enumerate(structure_score_report, start=1):
            rec['rank'] = rank
            rec['rank_mode'] = 'machine_floor_simplicity_tie'

    # Parser fallback follows the same score-first policy.  Only one expression
    # is ever compiled as the Stage-1 core; lower-score rows are retained solely
    # as emergency parser fallbacks.
    fallback_indices: List[int] = [core_idx]
    for i in structure_order:
        if i not in fallback_indices:
            fallback_indices.append(i)
    for i in score_order:
        if i not in fallback_indices:
            fallback_indices.append(i)
    # Every validation-ratio-qualified structure candidate must cross the
    # Python/MATLAB boundary for restart merging.  The display top-K controls
    # table length only; it must not silently truncate the selection pool.
    fallback_indices = fallback_indices[:max(1, topk, ranking_topk, len(structure_order))]
    fallback_candidates: List[Dict[str, Any]] = []
    for rank, i in enumerate(fallback_indices, start=1):
        rec = selected(i)
        rec['compile_fallback_rank'] = rank
        rec['selection_role'] = 'restart-local-core' if i == core_idx else 'candidate-pool'
        fallback_candidates.append(rec)

    return (selected(core_idx), validation_report, complexity_report,
            best_score_report, structure_score_report, fallback_candidates, len(records))

def _strip_private_prediction_field(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out.pop('_predictions', None)
    return out


class _OutputEquationView:
    """Expose one output table while retaining the fitted multi-output model.

    PySR stores ``equations_`` as one DataFrame per output after a 2-D ``Y``
    fit.  The validation/structure scorer is deliberately output-local, so it
    receives this view and can never rank candidates with a mixed-output loss.
    """

    def __init__(self, model: Any, output_index: int, n_outputs: int):
        self._model = model
        self._output_index = int(output_index)
        self._n_outputs = int(n_outputs)
        tables = model.equations_
        self.equations_ = tables[output_index] if isinstance(tables, list) else tables

    def predict(self, X: np.ndarray, index: int) -> np.ndarray:
        if self._n_outputs == 1:
            return np.asarray(self._model.predict(X, index=index), dtype=float)
        # PySR requires one positional equation index for every output.  The
        # non-target columns are discarded, so their first valid rows are used.
        indices = [0] * self._n_outputs
        indices[self._output_index] = int(index)
        prediction = np.asarray(self._model.predict(X, index=indices), dtype=float)
        return prediction[:, self._output_index]


def _population_allocation(config: Dict[str, Any], n_outputs: int) -> Dict[str, Any]:
    """Resolve PySR's per-output population count from the wrapper budget.

    Official PySR applies one common ``populations`` value to every output.  In
    fixed-total mode we therefore use the largest equal integer allocation that
    does not exceed the configured total.  A truly unequal/adaptive offspring
    quota requires a SymbolicRegression.jl scheduler extension and is not
    represented as available here.
    """
    requested = max(1, int(config.get('populations', 8)))
    mode = str(config.get('population_budget_mode', 'per_output')).strip().lower()
    if mode == 'fixed_total':
        if requested < n_outputs:
            raise ValueError(
                'Fixed-total population budget must be at least the number of unresolved outputs '
                f'({requested} < {n_outputs}).')
        per_output = max(1, requested // n_outputs)
    else:
        mode = 'per_output'
        per_output = requested
    return {
        'mode': mode,
        'configured_population_budget': requested,
        'populations_per_output': per_output,
        'effective_total_logical_populations': per_output * n_outputs,
        'unassigned_population_remainder': max(0, requested - per_output * n_outputs)
            if mode == 'fixed_total' else 0,
        'output_quota_policy': 'equal_protected_output_quota',
        'adaptive_offspring_quota_available': False,
    }


def _make_model_kwargs(config: Dict[str, Any], Xtr: np.ndarray, n_outputs: int
                       ) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    binary_ops, unary_ops, extra_sympy = _operator_lists(config)
    variable_names = config.get('variable_names') or [f'x{i+1}' for i in range(Xtr.shape[1])]
    variable_complexities = config.get('_variable_complexities', None)
    allocation = _population_allocation(config, n_outputs)
    batching = bool(config.get('batching', False))
    requested_batch_size = max(1, int(config.get('batch_size', 50)))
    effective_batch_size = min(requested_batch_size, max(1, int(Xtr.shape[0])))
    allocation = dict(allocation)
    allocation.update({
        'batching': batching,
        'requested_batch_size': requested_batch_size,
        'effective_batch_size': effective_batch_size if batching else int(Xtr.shape[0]),
        'full_data_hall_of_fame_evaluation': True,
    })
    model_kwargs: Dict[str, Any] = dict(
        niterations=int(config.get('niterations', 100)),
        populations=int(allocation['populations_per_output']),
        population_size=int(config.get('population_size', 50)),
        maxsize=int(config.get('maxsize', 30)),
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        model_selection=str(config.get('model_selection', 'best')),
        parsimony=float(config.get('parsimony', 1e-6)),
        random_state=int(config.get('random_state', 1)),
        verbosity=int(config.get('verbosity', 1)),
        progress=bool(config.get('progress', False)),
    )
    # Optional case-local evolutionary controls. The generator demo sets these
    # explicitly; other cases omit them and retain their existing PySR defaults.
    # Deliberately do not set ncycles_per_iteration here.
    evolution_controls: Dict[str, Any] = {
        'ncycles_per_iteration': 'unchanged_pysr_default',
    }
    if 'tournament_selection_n' in config:
        value = max(1, int(round(float(config['tournament_selection_n']))))
        model_kwargs['tournament_selection_n'] = value
        evolution_controls['tournament_selection_n'] = value
    if 'tournament_selection_p' in config:
        value = min(1.0, max(np.finfo(float).eps,
                             float(config['tournament_selection_p'])))
        model_kwargs['tournament_selection_p'] = value
        evolution_controls['tournament_selection_p'] = value
    if 'annealing' in config:
        value = bool(config['annealing'])
        model_kwargs['annealing'] = value
        evolution_controls['annealing'] = value
    if 'annealing_alpha' in config:
        value = max(np.finfo(float).eps, float(config['annealing_alpha']))
        model_kwargs['alpha'] = value
        evolution_controls['alpha'] = value
    if 'crossover_probability' in config:
        value = min(1.0, max(0.0, float(config['crossover_probability'])))
        model_kwargs['crossover_probability'] = value
        evolution_controls['crossover_probability'] = value
    if 'weight_optimize' in config:
        value = max(0.0, float(config['weight_optimize']))
        model_kwargs['weight_optimize'] = value
        evolution_controls['weight_optimize'] = value
    mutation_weight_keys = (
        'weight_add_node',
        'weight_insert_node',
        'weight_delete_node',
        'weight_do_nothing',
        'weight_mutate_constant',
        'weight_mutate_operator',
        'weight_mutate_feature',
        'weight_swap_operands',
        'weight_rotate_tree',
        'weight_randomize',
        'weight_simplify',
    )
    for key in mutation_weight_keys:
        if key in config:
            value = max(0.0, float(config[key]))
            model_kwargs[key] = value
            evolution_controls[key] = value
    if 'optimize_probability' in config:
        value = min(1.0, max(0.0, float(config['optimize_probability'])))
        model_kwargs['optimize_probability'] = value
        evolution_controls['optimize_probability'] = value
    if 'should_simplify' in config:
        value = bool(config['should_simplify'])
        model_kwargs['should_simplify'] = value
        evolution_controls['should_simplify'] = value
    allocation['evolution_controls'] = evolution_controls
    if len(evolution_controls) > 1:
        print(
            '[PySR case-local evolution controls] ' +
            json.dumps(evolution_controls, sort_keys=True),
            flush=True)
    if batching:
        model_kwargs['batching'] = True
        model_kwargs['batch_size'] = effective_batch_size
    if bool(config.get('machine_precision_early_stop_enable', False)):
        # Repeated .fit() calls add bounded iteration chunks to the same search.
        model_kwargs['warm_start'] = True
    deterministic = bool(config.get('deterministic', False))
    parallelism = str(config.get('parallelism', 'multithreading'))
    if deterministic:
        # Official PySR requires parallelism to be disabled for reproducibility.
        model_kwargs['deterministic'] = True
        model_kwargs['parallelism'] = 'serial'
    elif parallelism:
        model_kwargs['parallelism'] = parallelism
    maxdepth = config.get('maxdepth', None)
    if maxdepth is not None:
        try:
            model_kwargs['maxdepth'] = int(maxdepth)
        except Exception:
            pass
    if extra_sympy:
        model_kwargs['extra_sympy_mappings'] = extra_sympy
    if isinstance(variable_complexities, list) and len(variable_complexities) == Xtr.shape[1]:
        model_kwargs['complexity_of_variables'] = [float(v) for v in variable_complexities]
    operator_complexities = config.get('operator_complexities', {}) or {}

    # Complexity entries are legal only for operators that are actually
    # registered in this PySR fit. Inactive entries are ignored before the
    # Julia backend is constructed.
    binary_names = {str(op).split('(', 1)[0].strip().lower() for op in binary_ops}
    unary_names = {str(op).split('(', 1)[0].strip().lower() for op in unary_ops}
    active_operator_names = binary_names | unary_names
    cleaned_complexities: Dict[str, Any] = {}
    dropped_inactive_complexities: List[str] = []
    if isinstance(operator_complexities, dict):
        for name, value in operator_complexities.items():
            operator_name = str(name).strip()
            normalized_name = operator_name.lower()
            if normalized_name not in active_operator_names:
                dropped_inactive_complexities.append(operator_name)
                continue
            try:
                numeric_value = float(value)
            except Exception:
                continue
            if np.isfinite(numeric_value) and numeric_value > 0:
                cleaned_complexities[operator_name] = numeric_value

    if dropped_inactive_complexities:
        print(
            '[PySR grammar] ignored complexity entries for inactive operators: ' +
            ', '.join(sorted(dropped_inactive_complexities)),
            flush=True)

    # Reusable custom motifs are charged according to their primitive-tree
    # expansion rather than as unit-cost atomic operators.  Explicit user
    # values still override these defaults.
    if 'smoothsat' in unary_names:
        # z/sqrt(1+z^2): four primitive operators plus an additional penalty
        # for duplicating the argument subtree in the expanded tree.
        cleaned_complexities.setdefault('smoothsat', 6.0)
    if cleaned_complexities:
        model_kwargs['complexity_of_operators'] = cleaned_complexities

    # Prevent recursive custom-motif abuse.
    nested_constraints: Dict[str, Dict[str, int]] = {}
    if 'smoothsat' in unary_names:
        nested_constraints['smoothsat'] = {'smoothsat': 0}

    # Optional case-local motif-direction restriction. Disabled by default so
    # shared/Feynman behavior is unchanged. When enabled, sin/cos cannot
    # contain smoothsat.
    if bool(config.get('forbid_motif_reverse_nesting', False)):
        for trig_name in ('sin', 'cos'):
            if trig_name in unary_names and 'smoothsat' in unary_names:
                nested_constraints.setdefault(trig_name, {})['smoothsat'] = 0

    # Optional case-local structural restrictions. Disabled by default so
    # shared/Feynman behavior is unchanged. A value of zero means that the
    # inner operator may not occur anywhere below the outer operator.
    if bool(config.get('forbid_nested_trig', False)):
        if 'sin' in unary_names:
            nested_constraints.setdefault('sin', {})
            nested_constraints['sin']['sin'] = 0
            if 'cos' in unary_names:
                nested_constraints['sin']['cos'] = 0
        if 'cos' in unary_names:
            nested_constraints.setdefault('cos', {})
            if 'sin' in unary_names:
                nested_constraints['cos']['sin'] = 0
            nested_constraints['cos']['cos'] = 0

    if bool(config.get('forbid_nested_square', False)) and 'square' in unary_names:
        nested_constraints.setdefault('square', {})['square'] = 0

    # The protected sqrt operator is registered with PySR as ``sqrt_abs``;
    # target that exact Julia operator name rather than the MATLAB-facing
    # grammar alias ``sqrt``.
    if bool(config.get('forbid_nested_sqrt', False)) and 'sqrt_abs' in unary_names:
        nested_constraints.setdefault('sqrt_abs', {})['sqrt_abs'] = 0

    if nested_constraints:
        model_kwargs['nested_constraints'] = nested_constraints
        print(
            '[PySR grammar] nested constraints=' +
            json.dumps(nested_constraints, sort_keys=True),
            flush=True)

    return model_kwargs, binary_ops, unary_ops, allocation


def _make_pysr_model(model_kwargs: Dict[str, Any], deterministic: bool) -> Any:
    from pysr import PySRRegressor

    try:
        return PySRRegressor(**model_kwargs)
    except TypeError as exc:
        # Compatibility fallback for older PySR releases.  First remove the
        # optional per-variable complexity vector; then apply the legacy serial
        # API conversion when deterministic execution was requested.
        legacy_kwargs = dict(model_kwargs)
        changed = False
        if 'complexity_of_variables' in legacy_kwargs:
            legacy_kwargs.pop('complexity_of_variables', None)
            changed = True
        if deterministic and 'parallelism' in legacy_kwargs:
            legacy_kwargs.pop('parallelism', None)
            legacy_kwargs['procs'] = 0
            legacy_kwargs['multithreading'] = False
            changed = True
        if changed:
            try:
                model = PySRRegressor(**legacy_kwargs)
            except TypeError as fallback_exc:
                fallback_message = str(fallback_exc)
                if ('guesses' in legacy_kwargs and
                        ('guesses' in fallback_message or
                         'fraction_replaced_guesses' in fallback_message)):
                    raise RuntimeError(
                        'The selected PySR build does not expose constructor-level equation '
                        'guesses. The v75 adapter requires pysr==2.0.0a2 or a compatible '
                        'PySR 2 build.') from fallback_exc
                raise
            print(f'[PySR compatibility] optional keyword fallback used: {exc!r}', flush=True)
            return model
        message = str(exc)
        if ('guesses' in model_kwargs and
                ('guesses' in message or 'fraction_replaced_guesses' in message)):
            raise RuntimeError(
                'The selected PySR build does not expose constructor-level equation guesses. '
                'The v75 adapter requires pysr==2.0.0a2 or a compatible PySR 2 build.') from exc
        raise


def _export_fitted_output(model_view: _OutputEquationView, output_index: int,
                          Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray,
                          yval: np.ndarray, Xte: np.ndarray, Xood: np.ndarray,
                          config: Dict[str, Any], work_dir: Path, fit_time: float,
                          binary_ops: List[str], unary_ops: List[str],
                          allocation: Dict[str, Any]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    deterministic = bool(config.get('deterministic', False))
    parallelism = str(config.get('parallelism', 'multithreading'))

    (core, validation_ranking, complexity_ranking,
     best_score_ranking, structure_score_ranking,
     fallback_candidates, candidate_count) = _external_validation_selection(
        model_view, Xtr, ytr, Xval, yval, Xte, Xood, config
    )
    yhat_tr, yhat_val, yhat_te, yhat_ood = core['_predictions']

    exported_fallback: List[Dict[str, Any]] = []
    for candidate in fallback_candidates:
        meta = _strip_private_prediction_field(candidate)
        rank = int(candidate['compile_fallback_rank'])
        prefix = work_dir / f'candidate_y{output_index+1}_{rank:03d}'
        pred_tr, pred_val, pred_te, pred_ood = candidate['_predictions']
        paths = {
            'train': str(prefix) + '_train.csv',
            'validation': str(prefix) + '_val.csv',
            'test': str(prefix) + '_test.csv',
            'ood': str(prefix) + '_ood.csv',
        }
        _save_matrix(paths['train'], pred_tr)
        _save_matrix(paths['validation'], pred_val)
        _save_matrix(paths['test'], pred_te)
        _save_matrix(paths['ood'], pred_ood)
        meta['prediction_paths'] = paths
        exported_fallback.append(meta)

    equations_csv = work_dir / f'equations_y{output_index+1}.csv'
    try:
        exported_equations = model_view.equations_.copy()
        for column in ('equation', 'sympy_format'):
            if column in exported_equations.columns:
                exported_equations[column] = exported_equations[column].map(
                    lambda value: _canonical_display_expression(_expression_text(value), config))
        exported_equations.to_csv(equations_csv, index=False)
    except Exception:
        equations_csv.write_text('', encoding='utf-8')

    out_meta = dict(
        output_index=output_index + 1,
        expression=core['expression'],
        complexity=core['complexity'],
        selection_mode=str(core.get('selection_rule', 'native_pysr_score')),
        selected_core=_strip_private_prediction_field(core),
        selected_best_score=_strip_private_prediction_field(core),
        # Backward-compatible alias; no second/simple tree is selected.
        selected_best=_strip_private_prediction_field(core),
        # Backward-compatible alias: this is now the external-validation ranking.
        external_validation_equations=validation_ranking,
        external_validation_ranked_by_mse=validation_ranking,
        external_validation_ranked_by_complexity=complexity_ranking,
        external_validation_ranked_by_best_score=best_score_ranking,
        external_validation_ranked_by_structure_score=structure_score_ranking,
        external_validation_candidate_count=candidate_count,
        compile_fallback_candidates=exported_fallback,
        deterministic=deterministic,
        parallelism='serial' if deterministic else parallelism,
        batching=bool(allocation.get('batching', False)),
        requested_batch_size=int(allocation.get('requested_batch_size', Xtr.shape[0])),
        effective_batch_size=int(allocation.get('effective_batch_size', Xtr.shape[0])),
        full_data_hall_of_fame_evaluation=True,
        equations_csv=str(equations_csv),
        fit_time_seconds=fit_time,
        fit_time_accounting='shared_native_multioutput_call_total',
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        multi_output_search=True,
        population_allocation=allocation,
        initial_guess_scope=str(config.get('initial_guess_scope', 'shared_all_unresolved_outputs')),
        initial_guesses_enabled=bool(config.get('initial_guesses_enable', False)),
        initial_guesses=list(config.get('_resolved_initial_guesses', [])),
        fraction_replaced_guesses=float(config.get('fraction_replaced_guesses', 0.05)),
        top_expressions=_top_equations_from_model(model_view, config, 'loss'),
        top_expressions_by_loss=_top_equations_from_model(model_view, config, 'loss'),
        top_expressions_by_score=_top_equations_from_model(model_view, config, 'score'),
    )
    return yhat_tr, yhat_val, yhat_te, yhat_ood, out_meta


def _train_outputs_native(Xtr: np.ndarray, Ytr: np.ndarray, Xval: np.ndarray,
                          Yval: np.ndarray, Xte: np.ndarray, Xood: np.ndarray,
                          config: Dict[str, Any], work_dir: Path
                          ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
                                     List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Run one native PySR fit for all unresolved outputs in this restart."""
    n_outputs = int(Ytr.shape[1])
    model_kwargs, binary_ops, unary_ops, allocation = _make_model_kwargs(
        config, Xtr, n_outputs)
    guess_stats = _configure_native_shared_guesses(model_kwargs, config, n_outputs)
    model = _make_pysr_model(model_kwargs, bool(config.get('deterministic', False)))
    fit_target = Ytr[:, 0] if n_outputs == 1 else Ytr
    print(
        '[PySR multi-output] one native fit; '
        f'outputs={n_outputs}; allocation={json.dumps(allocation, sort_keys=True)}',
        flush=True)
    if bool(allocation.get('batching', False)):
        print(
            '[PySR batching] mini-batch evolution enabled; '
            f"requested={allocation['requested_batch_size']}; "
            f"effective={allocation['effective_batch_size']}; "
            'Hall-of-Fame comparisons use the full training set.',
            flush=True)
    t0 = time.time()
    feature_names = list(
        config.get('variable_names') or
        [f'x{i+1}' for i in range(Xtr.shape[1])])
    machine_precision_early_stop = _fit_native_model(
        model, Xtr, fit_target, feature_names, config, Ytr, n_outputs)
    fit_time = time.time() - t0
    if n_outputs > 1:
        if not isinstance(model.equations_, list) or len(model.equations_) != n_outputs:
            raise RuntimeError(
                'Installed PySR did not return one equation table per output after a 2-D Y fit. '
                'Use a PySR release with native multi-output PySRRegressor support.')

    yhat_tr_all: List[np.ndarray] = []
    yhat_val_all: List[np.ndarray] = []
    yhat_te_all: List[np.ndarray] = []
    yhat_ood_all: List[np.ndarray] = []
    outputs: List[Dict[str, Any]] = []
    for output_index in range(n_outputs):
        view = _OutputEquationView(model, output_index, n_outputs)
        yhat_tr, yhat_val, yhat_te, yhat_ood, meta = _export_fitted_output(
            view, output_index, Xtr, Ytr[:, output_index], Xval,
            Yval[:, output_index], Xte, Xood, config, work_dir, fit_time,
            binary_ops, unary_ops, allocation)
        yhat_tr_all.append(yhat_tr)
        yhat_val_all.append(yhat_val)
        yhat_te_all.append(yhat_te)
        yhat_ood_all.append(yhat_ood)
        outputs.append(meta)
    return (yhat_tr_all, yhat_val_all, yhat_te_all, yhat_ood_all, outputs,
            {'fit_time_seconds': fit_time, 'population_allocation': allocation,
             'evolution_controls': dict(allocation.get('evolution_controls', {})),
             'batching': bool(allocation.get('batching', False)),
             'requested_batch_size': int(allocation.get('requested_batch_size', Xtr.shape[0])),
             'effective_batch_size': int(allocation.get('effective_batch_size', Xtr.shape[0])),
             'full_data_hall_of_fame_evaluation': True,
             'initial_guess_stats': guess_stats,
             'machine_precision_early_stop': machine_precision_early_stop,
             'machine_precision_early_stop_enabled': bool(
                 machine_precision_early_stop.get('enabled', False)),
             'machine_precision_early_stop_triggered': bool(
                 machine_precision_early_stop.get('triggered', False)),
             'niterations_requested': int(
                 machine_precision_early_stop.get('requested_iterations',
                                                  config.get('niterations', 100))),
             'niterations_completed': int(
                 machine_precision_early_stop.get('completed_iterations',
                                                  config.get('niterations', 100)))})




def _prepare_single_generator_typed_features(
        Xsets: List[np.ndarray], config: Dict[str, Any]
        ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Expose only fixed sin(x1)/cos(x1) atoms for the generator case.

    Recursive sin/cos are removed from the grammar, so shifted, scaled,
    polynomial, or nested trigonometric arguments cannot be generated.
    """
    if not _single_generator_typed_prior_enabled(config):
        return Xsets, {}

    strict_atoms = bool(config.get('strict_trig_atoms_only', False))
    if not strict_atoms:
        # General operator mode: preserve the original state matrix and keep
        # sin/cos in the PySR unary grammar.  This is the required behavior for
        # arbitrary editable expressions such as sin(2*x1), cos(x1+x3), etc.
        unary = {str(op).strip().lower() for op in config.get('unary_operators', [])}
        missing = sorted({'sin', 'cos'} - unary)
        if missing:
            raise ValueError(
                'strict_trig_atoms_only=false requires ordinary sin/cos unary '
                'operators in the PySR grammar; missing: ' + ', '.join(missing))
        return Xsets, {
            'strict_trig_atoms_only': False,
            'fixed_trig_features': [],
            'general_trig_operators': ['sin', 'cos'],
        }

    if not Xsets or Xsets[0].ndim != 2 or Xsets[0].shape[1] < 4:
        raise ValueError('Typed generator feature preparation requires at least four state columns.')

    names = list(config.get('variable_names') or [f'x{i+1}' for i in range(Xsets[0].shape[1])])
    original_n = len(names)
    augmented: List[np.ndarray] = []
    for X in Xsets:
        X = np.asarray(X, dtype=float)
        sx = np.sin(X[:, 0:1])
        cx = np.cos(X[:, 0:1])
        augmented.append(np.hstack([X, sx, cx]))
    names += ['sin_x1_atom', 'cos_x1_atom']

    config['unary_operators'] = [
        op for op in config.get('unary_operators', [])
        if str(op).strip().lower() not in {'sin', 'cos'}
    ]
    config['variable_names'] = names
    config['_variable_complexities'] = [1.0] * (original_n + 2)
    return augmented, {
        'strict_trig_atoms_only': True,
        'fixed_trig_features': ['sin(x1)', 'cos(x1)'],
        'general_trig_operators': [],
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--parent-pid', type=int, default=0)
    parser.add_argument('--control-file', default='')
    args = parser.parse_args()

    _install_exit_signal_handlers()
    print(f'[PySR adapter] pid={os.getpid()}; parent_matlab_pid={args.parent_pid}; control_file={args.control_file}', flush=True)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    paper_root = config.get('pysr_paper_root') or ''
    if paper_root and os.path.isdir(paper_root):
        sys.path.insert(0, paper_root)

    work_dir = Path(config['work_dir'])
    work_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    paths = config['paths']
    try:
        Xtr = _load_matrix(paths['X_train'])
        Ytr = _load_matrix(paths['Y_train'])
        Xval = _load_matrix(paths['X_val'])
        Yval = _load_matrix(paths['Y_val'])
        Xte = _load_matrix(paths['X_test'])
        Yte = _load_matrix(paths['Y_test'])
        Xood = _load_matrix(paths['X_ood'])
        _ = Yval, Yte  # MATLAB computes final metrics from saved predictions.

        if Ytr.ndim == 1:
            Ytr = Ytr.reshape(-1, 1)
        if Ytr.shape[0] != Xtr.shape[0] and Ytr.shape[1] == Xtr.shape[0]:
            Ytr = Ytr.T

        [Xtr, Xval, Xte, Xood], typed_feature_stats = _prepare_single_generator_typed_features(
            [Xtr, Xval, Xte, Xood], config)
        if typed_feature_stats:
            print('[typed generator prior] fixed trigonometric atoms: ' +
                  json.dumps(typed_feature_stats, sort_keys=True), flush=True)
    except BaseException as exc:
        import traceback
        result = {
            'ok': False,
            'stage': 'typed_feature_preparation',
            'error': 'Single-generator typed-feature preprocessing failed.',
            'exception': repr(exc),
            'traceback': traceback.format_exc(),
            'python_executable': sys.executable,
        }
        (work_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        return 3

    try:
        import pysr  # noqa: F401
        pysr_version_info = _pysr_version_info(config)
        print('[PySR version] ' + json.dumps(pysr_version_info, sort_keys=True), flush=True)
    except Exception as exc:
        result = {
            'ok': False,
            'error': 'Could not import or validate the required official PySR 2 package.',
            'exception': repr(exc),
            'python_executable': sys.executable,
        }
        (work_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        return 2
    (yhat_tr_all, yhat_val_all, yhat_te_all, yhat_ood_all, outputs,
     multi_output_stats) = _train_outputs_native(
        Xtr, Ytr, Xval, Yval, Xte, Xood, config, work_dir)
    multi_output_stats['typed_features'] = typed_feature_stats

    Yhat_tr = np.hstack(yhat_tr_all)
    Yhat_val = np.hstack(yhat_val_all)
    Yhat_te = np.hstack(yhat_te_all)
    if Xood.size > 0:
        Yhat_ood = np.hstack(yhat_ood_all)
    else:
        Yhat_ood = np.zeros((0, Ytr.shape[1]), dtype=float)

    _save_matrix(str(work_dir / 'Yhat_train.csv'), Yhat_tr)
    _save_matrix(str(work_dir / 'Yhat_val.csv'), Yhat_val)
    _save_matrix(str(work_dir / 'Yhat_test.csv'), Yhat_te)
    _save_matrix(str(work_dir / 'Yhat_ood.csv'), Yhat_ood)

    result = {
        'ok': True,
        'backend': 'official_pysr',
        'variable_index_base': 1,
        'python_executable': sys.executable,
        'pysr_version': pysr_version_info['installed'],
        'minimum_pysr_version': pysr_version_info['required'],
        'pysr_paper_root': paper_root,
        'grammar_casemode': config.get('grammar_casemode'),
        'deterministic': bool(config.get('deterministic', False)),
        'parallelism': 'serial' if bool(config.get('deterministic', False)) else config.get('parallelism', 'multithreading'),
        'random_state': int(config.get('random_state', 1)),
        'batching': bool(config.get('batching', False)),
        'requested_batch_size': max(1, int(config.get('batch_size', 50))),
        'effective_batch_size': min(max(1, int(config.get('batch_size', 50))), int(Xtr.shape[0]))
            if bool(config.get('batching', False)) else int(Xtr.shape[0]),
        'full_data_hall_of_fame_evaluation': True,
        'multi_output_mode': 'native_single_fit_independent_output_archives',
        'multi_output_stats': multi_output_stats,
        'evolution_controls': multi_output_stats.get('evolution_controls', {}),
        'machine_precision_early_stop': multi_output_stats.get(
            'machine_precision_early_stop', {}),
        'machine_precision_early_stop_enabled': bool(multi_output_stats.get(
            'machine_precision_early_stop_enabled', False)),
        'machine_precision_early_stop_triggered': bool(multi_output_stats.get(
            'machine_precision_early_stop_triggered', False)),
        'niterations_requested': int(multi_output_stats.get(
            'niterations_requested', config.get('niterations', 100))),
        'niterations_completed': int(multi_output_stats.get(
            'niterations_completed', config.get('niterations', 100))),
        'initial_guesses_enabled': bool(config.get('initial_guesses_enable', False)),
        'initial_guesses': list(config.get('_resolved_initial_guesses', [])),
        'fraction_replaced_guesses': float(config.get('fraction_replaced_guesses', 0.05)),
        'periodic_guess_reinjection_enabled': bool(
            config.get('initial_guesses_enable', False) and
            float(config.get('fraction_replaced_guesses', 0.05)) > 0.0),
        'guess_injection_policy': ('official_periodic_population_replacement'
            if (config.get('initial_guesses_enable', False) and
                float(config.get('fraction_replaced_guesses', 0.05)) > 0.0)
            else 'disabled'),
        'initial_guess_scope': str(config.get('initial_guess_scope', 'shared_all_unresolved_outputs')),
        'outputs': outputs,
        'total_time_seconds': time.time() - t_total,
    }
    (work_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
