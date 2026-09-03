#!/usr/bin/env python3
"""MATLAB adapter for the unchanged official EQL-Div Theano implementation.

The bundled upstream files under ``official_eql/src`` implement the network,
loss, division curriculum, regularization phases, optimizer updates, active
unit count, and Vint-S model selection.  This adapter only:
  * translates MATLAB CSV data into the upstream pickle format,
  * launches independent upstream candidates,
  * calls the upstream model-selection function,
  * reloads the selected upstream state and exports predictions/metadata.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import gzip
import hashlib
import io
import json
import os
import pickle
import shutil
import secrets
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from kan_paper_adapter_common import json_safe, metrics, read_csv_matrix, write_json  # noqa: E402

EXPECTED_SHA256 = {
    "mlfg_final.py": "170c200842d135a5f0ffb11df6b6a1d2b14c073d3027be68df006eb2a1e648f0",
    "utils.py": "ac513891b83829b9e15d8bc7bef5803ef94819f1470523b36e0931133e480d9d",
    "model_selection_val_sparsity.py": "1e4b194286260199eda735b98e71b63edca5fdce4ec4a66a52dcc83df221e142",
}
EXPECTED_EQ11_DATA_SHA256 = {
    "div-n-10k-1.dat.gz": "c7380344c895cda092c30f499df959faad2cbd26a1e30aa520517c3408ddb20c",
    "div-n-5k-1-test.dat.gz": "e1e15ee2cf09aa0bb6333c790d831bb7ddd4839b83568ce2592c1751309aaf1b",
    "div-n-5k-1-2-test.dat.gz": "aafc7c166c83f32a1f9f712b6e36c537d89883a4880dc9e268f7c59ac7b67124",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verify_official_source(src_root: Path) -> Dict[str, str]:
    actual = {}
    for name, expected in EXPECTED_SHA256.items():
        path = src_root / name
        if not path.is_file():
            raise FileNotFoundError("Bundled official EQL source is missing: %s" % path)
        actual[name] = _sha256(path)
        if actual[name] != expected:
            raise RuntimeError(
                "Official EQL source integrity check failed for %s. Expected %s, got %s."
                % (name, expected, actual[name])
            )
    return actual


def _dump_dataset(path: Path, xtr, ytr, xv, yv, xt, yt) -> None:
    data = (
        (np.asarray(xtr, dtype=float), np.asarray(ytr, dtype=float)),
        (np.asarray(xv, dtype=float), np.asarray(yv, dtype=float)),
        (np.asarray(xt, dtype=float), np.asarray(yt, dtype=float)),
    )
    with gzip.open(str(path), "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_gzip_pickle(path: Path):
    with gzip.open(str(path), "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def _write_matrix(path: Path, a: np.ndarray) -> None:
    a = np.asarray(a)
    if a.size == 0:
        path.write_text("", encoding="utf-8")
    else:
        np.savetxt(str(path), a, delimiter=",", fmt="%.17g")


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_input_unit_box(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an affine map that sends the training box to approximately [-1,1]."""
    x = np.asarray(x, dtype=float)
    lo = np.min(x, axis=0, keepdims=True)
    hi = np.max(x, axis=0, keepdims=True)
    center = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    scale[~np.isfinite(scale) | (np.abs(scale) < 1e-12)] = 1.0
    return center, scale


def _fit_output_standardizer(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    center = np.mean(y, axis=0, keepdims=True)
    scale = np.std(y, axis=0, keepdims=True)
    scale[~np.isfinite(scale) | (np.abs(scale) < 1e-12)] = 1.0
    return center, scale


def _affine_apply(a: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) - center) / scale


def _affine_reverse(a: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float) * scale + center




def _remove_cache_tree(cache_root: Path, attempts: int = 8, delay_seconds: float = 0.25) -> None:
    """Best-effort removal of one run-private Theano cache tree.

    Windows can retain short-lived file handles after a compiler/Python child
    exits, so removal is retried briefly.  Candidate logs, JSON metadata, and
    trained states are stored in the run work directory and are not touched.
    """
    cache_root = Path(cache_root)
    for attempt in range(max(1, int(attempts))):
        try:
            if cache_root.exists():
                shutil.rmtree(str(cache_root))
            break
        except OSError:
            if attempt + 1 >= attempts:
                print(
                    "[Official EQL sweep] warning: could not fully remove Theano cache: %s" % cache_root,
                    flush=True,
                )
                return
            time.sleep(float(delay_seconds))


def _theano_path(path: Path) -> str:
    """Return a Windows-safe path for THEANO_FLAGS (forward slashes)."""
    return Path(path).resolve().as_posix()


def _available_logical_cpu_count() -> int:
    """Return the logical CPU count available to this Python process."""
    # On POSIX, sched_getaffinity respects container/job affinity limits.
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            n = len(get_affinity(0))
            if n > 0:
                return int(n)
        except (OSError, TypeError, ValueError):
            pass
    n = os.cpu_count()
    return max(1, int(n) if n is not None else 1)


def _parse_candidate_worker_request(raw) -> int:
    """Parse worker control; 0/negative/'auto' means automatic."""
    if raw is None:
        return 0
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("", "auto", "max", "all"):
            return 0
        try:
            return int(float(text))
        except ValueError as exc:
            raise ValueError(
                "candidate_workers must be a positive integer or 0/'auto'; got %r" % raw
            ) from exc
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "candidate_workers must be a positive integer or 0/'auto'; got %r" % raw
        ) from exc


def _resolve_complete_minibatch_size(requested: int, n_train: int, n_val: int) -> Tuple[int, bool]:
    """Choose the largest batch <= requested that exactly divides train and val.

    The unchanged official EQL loop uses integer batch counts and drops any
    incomplete minibatch.  Adapting the batch size preserves every MATLAB
    train/validation sample instead of silently discarding rows or aborting a
    sample-efficiency sweep such as Ntrain=250 with the paper default 20.
    """
    try:
        requested = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size must be a positive integer; got %r" % (requested,)) from exc
    n_train = int(n_train)
    n_val = int(n_val)
    if requested < 1:
        raise ValueError("batch_size must be at least 1; got %d" % requested)
    if n_train < 1 or n_val < 1:
        raise ValueError("EQL train and validation sets must both be non-empty.")
    upper = min(requested, n_train, n_val)
    for batch in range(upper, 0, -1):
        if n_train % batch == 0 and n_val % batch == 0:
            return batch, batch != requested
    # batch=1 always divides positive counts, so this line is defensive only.
    raise RuntimeError("Could not resolve a complete EQL minibatch size.")


def _resolve_candidate_workers(requested: int, candidate_count: int, logical_cpu_count: int) -> Tuple[int, str]:
    """Resolve the candidate-level worker pool without oversubscribing CPUs."""
    candidate_count = max(0, int(candidate_count))
    logical_cpu_count = max(1, int(logical_cpu_count))
    if candidate_count == 0:
        return 0, "auto" if requested <= 0 else "manual"
    if requested <= 0:
        return min(candidate_count, logical_cpu_count), "auto"
    return min(candidate_count, max(1, int(requested))), "manual"


def _run_candidate(python_exe: str, runner: Path, cfg_path: Path, log_path: Path) -> Tuple[int, str]:
    cmd = [python_exe, str(runner), "--config", str(cfg_path)]
    env = os.environ.copy()
    # The upstream job scripts request one CPU per candidate.  Preserve that
    # behavior so candidate-level parallelism does not create nested BLAS/OMP
    # oversubscription.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False, env=env)
    return int(proc.returncode), str(log_path)


def _build_official_scan(path: Path, candidates: List[Dict], outer_seed: int, dataset_label: str) -> None:
    keys = [
        "k", "iter", "layers", "epochs", "nodes", "lr", "L1", "L2", "shortcut",
        "batchsize", "regstart", "regend", "id", "dataset", "gradient", "numactive",
        "bestnumactive", "bestepoch", "runtime", "extrapol2", "valerror", "valerrorbest", "testerror",
    ]
    lines = ["#C " + " ".join(keys), "# extra datasets: no OOD labels are used for selection"]
    for c in candidates:
        row = [
            99999999999,
            outer_seed,
            c["official_hidden_layers"],
            c["epochs"],
            c["units_per_type"],
            c["learning_rate"],
            c["lambda_l1"],
            c["lambda_l2"],
            False,
            c["batch_size"],
            c["reg_start"],
            c["reg_end"],
            c["candidate_id"],
            dataset_label,
            c["gradient"],
            c["official_num_active"],
            c["official_best_num_active"],
            c["official_best_epoch"],
            c["official_runtime_minutes"],
            0.0,  # informational field only; OOD labels are deliberately not supplied
            c["official_final_val_mse"],
            c["official_best_val_mse"],
            c["official_test_mse"],
        ]
        lines.append("\t".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _official_select(src_root: Path, scan_path: Path) -> Tuple[Dict, str]:
    if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: np.asarray(a).item()  # type: ignore[attr-defined]
    sys.path.insert(0, str(src_root))
    import model_selection_val_sparsity as official_selection  # noqa: E402

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        selected = official_selection.select_instance(str(scan_path))
    return selected, buf.getvalue()


def _official_eq11_data(official_root: Path):
    data_root = official_root / "data"
    for name, expected in EXPECTED_EQ11_DATA_SHA256.items():
        path = data_root / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("Official Eq. (11) data integrity check failed: %s" % path)
    train_data = _load_gzip_pickle(data_root / "div-n-10k-1.dat.gz")
    id_data = _load_gzip_pickle(data_root / "div-n-5k-1-test.dat.gz")
    ood_data = _load_gzip_pickle(data_root / "div-n-5k-1-2-test.dat.gz")
    xid, yid = id_data[0]
    xood, yood = ood_data[0]
    return train_data, np.asarray(xid), np.asarray(yid), np.asarray(xood), np.asarray(yood)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--parent-pid")
    ap.add_argument("--control-file")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    t_total = time.perf_counter()
    outdir = Path(cfg["work_dir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    official_root = Path(cfg.get("official_root", HERE / "official_eql")).resolve()
    src_root = official_root / "src"
    source_hashes = _verify_official_source(src_root)

    use_eq11 = bool(cfg.get("use_bundled_official_eq11_data", False))
    paths = cfg["paths"]
    if use_eq11:
        train_data, xt_raw, yt_raw, xo_raw, yo_raw = _official_eq11_data(official_root)
        dataset_path = official_root / "data" / "div-n-10k-1.dat.gz"
        (xtr_raw, ytr_raw), (xv_raw, yv_raw), _ = train_data
        # The acceptance demo must use the exact upstream bytes and scale.
        x_center = np.zeros((1, xtr_raw.shape[1]), dtype=float)
        x_scale = np.ones((1, xtr_raw.shape[1]), dtype=float)
        y_center = np.zeros((1, ytr_raw.shape[1]), dtype=float)
        y_scale = np.ones((1, ytr_raw.shape[1]), dtype=float)
        xtr, ytr, xv, yv, xt, yt, xo, yo = (
            np.asarray(xtr_raw), np.asarray(ytr_raw), np.asarray(xv_raw), np.asarray(yv_raw),
            np.asarray(xt_raw), np.asarray(yt_raw), np.asarray(xo_raw), np.asarray(yo_raw)
        )
        normalize_inputs = False
        normalize_outputs = False
        dataset_label = "official_eql_division_dataset"
    else:
        xtr_raw = read_csv_matrix(paths["X_train"])
        ytr_raw = read_csv_matrix(paths["Y_train"])
        xv_raw = read_csv_matrix(paths["X_val"])
        yv_raw = read_csv_matrix(paths["Y_val"])
        xt_raw = read_csv_matrix(paths["X_test"])
        yt_raw = read_csv_matrix(paths["Y_test"])
        xo_raw = read_csv_matrix(paths["X_ood"])
        yo_raw = read_csv_matrix(paths["Y_ood"])
        if xtr_raw.ndim != 2 or ytr_raw.ndim != 2 or xtr_raw.shape[0] != ytr_raw.shape[0]:
            raise ValueError("X_train/Y_train must be matrices with equal row counts.")
        normalize_inputs = bool(cfg.get("normalize_inputs", True))
        normalize_outputs = bool(cfg.get("normalize_outputs", True))
        if normalize_inputs:
            x_center, x_scale = _fit_input_unit_box(xtr_raw)
        else:
            x_center = np.zeros((1, xtr_raw.shape[1]), dtype=float)
            x_scale = np.ones((1, xtr_raw.shape[1]), dtype=float)
        if normalize_outputs:
            y_center, y_scale = _fit_output_standardizer(ytr_raw)
        else:
            y_center = np.zeros((1, ytr_raw.shape[1]), dtype=float)
            y_scale = np.ones((1, ytr_raw.shape[1]), dtype=float)
        xtr = _affine_apply(xtr_raw, x_center, x_scale)
        xv = _affine_apply(xv_raw, x_center, x_scale)
        xt = _affine_apply(xt_raw, x_center, x_scale)
        xo = _affine_apply(xo_raw, x_center, x_scale) if xo_raw.size else xo_raw
        ytr = _affine_apply(ytr_raw, y_center, y_scale)
        yv = _affine_apply(yv_raw, y_center, y_scale)
        yt = _affine_apply(yt_raw, y_center, y_scale)
        yo = _affine_apply(yo_raw, y_center, y_scale) if yo_raw.size else yo_raw
        dataset_path = outdir / "official_dataset.dat.gz"
        _dump_dataset(dataset_path, xtr, ytr, xv, yv, xt, yt)
        dataset_label = "matlab_shared_split_affine_normalized"

    if xtr.shape[0] < 1 or xtr.shape[1] < 1 or ytr.shape[1] < 1:
        raise ValueError("EQL requires non-empty two-dimensional training matrices.")
    if any(a.shape[0] < 1 for a in (xv, yv, xt, yt)):
        raise ValueError("EQL requires non-empty validation and ID-test data.")
    requested_batch_size = int(cfg["batch_size"])
    effective_batch_size, batch_size_adjusted = _resolve_complete_minibatch_size(
        requested_batch_size, xtr.shape[0], xv.shape[0]
    )
    cfg["batch_size"] = int(effective_batch_size)
    if batch_size_adjusted:
        print(
            "[Official EQL sweep] adjusted batch_size %d -> %d so train=%d and val=%d "
            "use complete minibatches without dropping samples."
            % (requested_batch_size, effective_batch_size, xtr.shape[0], xv.shape[0]),
            flush=True,
        )

    eval_test_path = outdir / "X_eval_test.csv"
    eval_ood_path = outdir / "X_eval_ood.csv"
    _write_matrix(eval_test_path, xt)
    _write_matrix(eval_ood_path, xo)

    depth_list = [int(v) for v in cfg["depth_list"]]
    lambda_list = [float(v) for v in cfg["lambda_list"]]
    steps_per_hidden = int(cfg["steps_per_hidden_layer"])
    requested_candidate_workers = _parse_candidate_worker_request(cfg.get("candidate_workers", 0))
    python_exe = sys.executable
    candidate_runner = HERE / "eql_official_candidate_runner.py"
    candidate_dir = outdir / "candidates"
    candidate_dir.mkdir(exist_ok=True)

    # Keep all disposable Theano compiler artifacts inside a deliberately
    # short project-local path.  Old Theano/Windows compiler toolchains are
    # sensitive to long base_compiledir paths, so use cache/rXXXX/cYY rather
    # than embedding the work-directory UUID, depth, or lambda in the path.
    cache_parent = HERE / "cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    for _ in range(256):
        short_run_id = "r" + secrets.token_hex(2)
        cache_root = cache_parent / short_run_id
        try:
            cache_root.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Could not allocate a unique short EQL cache directory under %s" % cache_parent)
    cache_cleanup_enabled = not bool(cfg.get("keep_theano_cache", False))
    if cache_cleanup_enabled:
        atexit.register(_remove_cache_tree, cache_root)
    print(
        "[Official EQL sweep] disposable Theano cache root=%s; auto-delete=%d; mode=exact-compiledir"
        % (cache_root, int(cache_cleanup_enabled)),
        flush=True,
    )

    base_seed = int(cfg.get("seed", 1))
    candidate_specs = []
    index = 0
    for paper_depth in depth_list:
        if paper_depth < 2:
            raise ValueError("Paper depth L must be at least 2; official hidden layers are L-1.")
        hidden_layers = paper_depth - 1
        epochs = hidden_layers * steps_per_hidden
        reg_start = int(round(epochs / 4.0))
        reg_end = int(round(19.0 * epochs / 20.0))
        for lam in lambda_list:
            candidate_id = (base_seed - 1) * 100000 + index
            stem = "candidate_%04d_L%d_lam_%0.12g" % (index + 1, paper_depth, lam)
            cdir = candidate_dir / stem
            cdir.mkdir(exist_ok=True)
            ccfg = {
                "candidate_id": candidate_id,
                "paper_depth_L": paper_depth,
                "official_hidden_layers": hidden_layers,
                "lambda_l1": lam,
                "lambda_l2": float(cfg.get("lambda_l2", 0.0)),
                "epochs": epochs,
                "reg_start": reg_start,
                "reg_end": reg_end,
                "learning_rate": float(cfg["learning_rate"]),
                "gradient": str(cfg.get("gradient", "adam")),
                "batch_size": int(cfg["batch_size"]),
                "units_per_type": int(cfg["units_per_unary_type"]),
                "physical_y_scale": np.asarray(y_scale, dtype=float).reshape(1, -1).tolist(),
                "checkpoint_selection_mode": "final_state",
                "penalty_every": int(cfg.get("penalty_every", 50)),
                "validate_every": int(cfg.get("validate_every", 10)),
                "verbose": bool(cfg.get("official_verbose", False)),
                "theano_flags": str(cfg.get("theano_flags", "")),
                # Every candidate receives its own Theano compilation-cache root.
                # This removes shared-cache locking/races when all unseen depth/lambda
                # graphs are launched concurrently on Windows.
                "candidate_compiledir": _theano_path(cache_root / ("c%02d" % (index + 1))),
                "official_src_root": str(src_root),
                "dataset_path": str(dataset_path),
                "state_path": str(cdir / "selected_state.pkl"),
                "best_state_path": str(cdir / "best_state.pkl"),
                "final_state_path": str(cdir / "final_state.pkl"),
                "result_path": str(cdir / "candidate_result.json"),
            }
            cfg_path = cdir / "candidate_config.json"
            cfg_path.write_text(json.dumps(ccfg, indent=2), encoding="utf-8")
            candidate_specs.append((index, cfg_path, cdir / "candidate_stdout.txt", Path(ccfg["result_path"])))
            index += 1

    logical_cpu_count = _available_logical_cpu_count()
    candidate_workers, worker_policy = _resolve_candidate_workers(
        requested_candidate_workers, len(candidate_specs), logical_cpu_count
    )
    requested_label = "auto" if requested_candidate_workers <= 0 else str(requested_candidate_workers)
    print(
        "[Official EQL sweep] candidates=%d, worker_request=%s, logical_cpus=%d, "
        "resolved_workers=%d, policy=%s, upstream=%s"
        % (
            len(candidate_specs), requested_label, logical_cpu_count,
            candidate_workers, worker_policy, official_root,
        ),
        flush=True,
    )

    completed: Dict[int, Dict] = {}
    def launch(spec):
        idx, cfg_path, log_path, result_path = spec
        status, _ = _run_candidate(python_exe, candidate_runner, cfg_path, log_path)
        if result_path.is_file():
            r = _read_json(result_path)
        else:
            r = {"status": "failed", "error": "candidate_result.json was not created", "time_seconds": None}
        r["subprocess_status"] = status
        r["stdout_path"] = str(log_path)
        return idx, r

    def report_completion(done_count, r):
        print(
            "[Official EQL sweep] %d/%d L=%s lambda=%s status=%s "
            "physical_val_mse=%s checkpoint=%s active=%s time=%s"
            % (
                done_count, len(candidate_specs), r.get("paper_depth_L"), r.get("lambda_l1"), r.get("status"),
                r.get("framework_val_mse"), r.get("selected_checkpoint"),
                r.get("selected_num_active", r.get("official_num_active")), r.get("time_seconds")
            ), flush=True,
        )

    if candidate_workers == 1:
        for k, spec in enumerate(candidate_specs, 1):
            idx, r = launch(spec)
            completed[idx] = r
            report_completion(k, r)
    else:
        # Launch every depth/lambda candidate immediately. Each subprocess has
        # a candidate-private exact Theano compiledir (set by the candidate
        # runner), so unseen graphs do not contend for a shared Windows cache.
        done_count = 0
        print(
            "[Official EQL sweep] direct candidate parallelism: %d candidate(s), %d workers, no serial warm-up"
            % (len(candidate_specs), candidate_workers),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=candidate_workers) as pool:
            pending = {pool.submit(launch, spec): spec for spec in candidate_specs}
            heartbeat_t0 = time.perf_counter()
            while pending:
                finished, not_done = wait(set(pending), timeout=30.0, return_when=FIRST_COMPLETED)
                if not finished:
                    elapsed = time.perf_counter() - heartbeat_t0
                    print(
                        "[Official EQL sweep] heartbeat: %d/%d complete; %d running/pending; elapsed %.1f s"
                        % (done_count, len(candidate_specs), len(not_done), elapsed),
                        flush=True,
                    )
                    continue
                for fut in finished:
                    spec = pending.pop(fut)
                    try:
                        idx, r = fut.result()
                    except BaseException as exc:
                        idx = spec[0]
                        r = {
                            "status": "failed",
                            "error": "%s: %s" % (type(exc).__name__, exc),
                            "paper_depth_L": _read_json(spec[1]).get("paper_depth_L"),
                            "lambda_l1": _read_json(spec[1]).get("lambda_l1"),
                            "time_seconds": None,
                        }
                    completed[idx] = r
                    done_count += 1
                    report_completion(done_count, r)

    candidates = [completed[i] for i in range(len(candidate_specs))]
    successful = [c for c in candidates if c.get("status") == "ok"]
    if not successful:
        first_error = next((str(c.get("error")) for c in candidates if c.get("error")), "unknown candidate failure")
        raise RuntimeError(
            "All official EQL candidates failed. First failure: %s. "
            "See candidate_stdout.txt files in %s" % (first_error, candidate_dir)
        )

    scan_path = outdir / "official_vint_s_scan.dat"
    _build_official_scan(scan_path, successful, base_seed, dataset_label)
    selected_info, selection_stdout = _official_select(src_root, scan_path)
    selected_id = int(selected_info["id"])
    selected = next(c for c in successful if int(c["candidate_id"]) == selected_id)
    for c in candidates:
        c["selected"] = bool(c.get("status") == "ok" and int(c["candidate_id"]) == selected_id)
    selected["selection_score"] = float(selected_info["score"])
    selected["selection_active_normalized"] = float(selected_info["num_active"])
    selected["selection_validation_normalized"] = float(selected_info["valerror"])

    eval_cfg = {
        "official_src_root": str(src_root),
        "dataset_path": str(dataset_path),
        "state_path": selected["state_path"],
        "candidate_id": selected_id,
        "units_per_type": int(cfg["units_per_unary_type"]),
        "official_hidden_layers": int(selected["official_hidden_layers"]),
        "gradient": str(cfg.get("gradient", "adam")),
        "theano_flags": str(cfg.get("theano_flags", "")),
        "candidate_compiledir": _theano_path(cache_root / "s01"),
        "xtest_path": str(eval_test_path),
        "xood_path": str(eval_ood_path),
        "yhat_train_path": str(outdir / "Yhat_train.csv"),
        "yhat_val_path": str(outdir / "Yhat_val.csv"),
        "yhat_test_path": str(outdir / "Yhat_test.csv"),
        "yhat_ood_path": str(outdir / "Yhat_ood.csv"),
    }
    eval_cfg_path = outdir / "selected_evaluation_config.json"
    eval_cfg_path.write_text(json.dumps(eval_cfg, indent=2), encoding="utf-8")
    eval_log = outdir / "selected_evaluation_stdout.txt"
    status, _ = _run_candidate(python_exe, HERE / "eql_official_evaluate_runner.py", eval_cfg_path, eval_log)
    if status != 0:
        raise RuntimeError("Official EQL selected-state evaluation failed. See %s" % eval_log)

    def read_pred(name):
        p = outdir / name
        if not p.exists() or p.stat().st_size == 0:
            return np.empty((0, ytr.shape[1]))
        return np.loadtxt(str(p), delimiter=",", ndmin=2)

    yp_tr_model = read_pred("Yhat_train.csv")
    yp_v_model = read_pred("Yhat_val.csv")
    yp_t_model = read_pred("Yhat_test.csv")
    yp_o_model = read_pred("Yhat_ood.csv")
    yp_tr = _affine_reverse(yp_tr_model, y_center, y_scale)
    yp_v = _affine_reverse(yp_v_model, y_center, y_scale)
    yp_t = _affine_reverse(yp_t_model, y_center, y_scale)
    yp_o = _affine_reverse(yp_o_model, y_center, y_scale) if yp_o_model.size else yp_o_model
    # MATLAB expects physical-scale predictions in the common output files.
    _write_matrix(outdir / "Yhat_train.csv", yp_tr)
    _write_matrix(outdir / "Yhat_val.csv", yp_v)
    _write_matrix(outdir / "Yhat_test.csv", yp_t)
    _write_matrix(outdir / "Yhat_ood.csv", yp_o)
    result = {
        "method": "Official EQL-Div upstream sweep",
        "protocol": "official_eql_div_theano_vint_s",
        "source_repository": "martius-lab/EQL",
        "source_commit": "cd93824ccd330b814ddd334dd25e8b0fd5eb3f77",
        "official_source_sha256": source_hashes,
        "official_source_unmodified": True,
        "official_python_executable": sys.executable,
        "selection_metric": "Upstream model_selection_val_sparsity.select_instance (Vint-S)",
        "reported_mse_scale": "original_physical_output_units",
        "official_model_selection_stdout": selection_stdout,
        "total_time_seconds": float(time.perf_counter() - t_total),
        "selected_candidate_time_seconds": selected.get("time_seconds"),
        "selected_index": int(candidates.index(selected) + 1),
        "selected_candidate_id": selected_id,
        "selected_depth": int(selected["paper_depth_L"]),
        "selected_functional_layer_count": int(selected["official_hidden_layers"]),
        "selected_lambda": float(selected["lambda_l1"]),
        "selected_score": float(selected_info["score"]),
        "selected_checkpoint": str(selected.get("selected_checkpoint", "final_state")),
        "checkpoint_selection_metric": str(selected.get("checkpoint_selection_metric", "forced_upstream_final_state")),
        "selected_connected_unit_count": int(selected.get("selected_num_active", selected["official_num_active"])),
        "selected_connected_units_by_layer": [],
        "selected_connected_units_by_type": {},
        "selected_connected_active_weight_count": int(selected["active_weight_count"]),
        "selected_active_weight_count": int(selected["active_weight_count"]),
        "selected_active_bias_count": int(selected["active_bias_count"]),
        "selected_active_parameter_count": int(selected["active_parameter_count"]),
        "parameter_count": int(selected["parameter_count"]),
        "units_per_unary_type": int(cfg["units_per_unary_type"]),
        "multiplication_units": int(cfg["units_per_unary_type"]),
        "unary_functions": ["identity", "sin", "cos"],
        "binary_functions": ["multiply"],
        "output_function": "official regularized division",
        "train_metrics": metrics(ytr_raw, yp_tr),
        "val_metrics": metrics(yv_raw, yp_v),
        "test_metrics": metrics(yt_raw, yp_t),
        "ood_metrics": metrics(yo_raw, yp_o),
        "selected_penalty_diagnostics": {},
        "candidate_count": len(candidates),
        "successful_candidate_count": len(successful),
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "batch_size_adjusted": bool(batch_size_adjusted),
        "candidates": candidates,
        "official_settings": {
            "paper_depth_list": depth_list,
            "lambda_list": lambda_list,
            "steps_per_hidden_layer": steps_per_hidden,
            "learning_rate": float(cfg["learning_rate"]),
            "gradient": str(cfg.get("gradient", "adam")),
            "lambda_l2": float(cfg.get("lambda_l2", 0.0)),
            "batch_size": int(effective_batch_size),
            "requested_batch_size": int(requested_batch_size),
            "effective_batch_size": int(effective_batch_size),
            "batch_size_policy": "largest common divisor not exceeding requested; no samples dropped",
            "batch_size_adjusted": bool(batch_size_adjusted),
            "penalty_every": int(cfg.get("penalty_every", 50)),
            "validate_every": int(cfg.get("validate_every", 10)),
            "candidate_workers": candidate_workers,
            "candidate_workers_requested": requested_label,
            "candidate_worker_policy": worker_policy,
            "logical_cpu_count": logical_cpu_count,
            "uses_input_normalization": normalize_inputs,
            "input_normalization": "training-box affine map to [-1,1]" if normalize_inputs else "none",
            "uses_output_normalization": normalize_outputs,
            "output_normalization": "training mean/std" if normalize_outputs else "none",
            "output_division_bias_initialization": "upstream ones",
            "active_unit_threshold": 0.1,
            "hard_weight_threshold": 0.001,
            "test_division_threshold": 0.0001,
            "upstream_extrapolation_output_bound": 100,
        },
        "normalization": {
            "x_center": x_center.tolist(), "x_scale": x_scale.tolist(),
            "y_center": y_center.tolist(), "y_scale": y_scale.tolist(),
        },
        "data_mode": "bundled official Eq. (11) dataset (no normalization)" if use_eq11 else "MATLAB shared split adapted to official EQL input/output scale",
        "uses_ood_labels_for_selection": False,
        "scan_file": str(scan_path),
    }
    write_json(str(outdir / "result.json"), json_safe(result))
    if cache_cleanup_enabled:
        _remove_cache_tree(cache_root)
    print(
        "[Official EQL sweep] selected candidate_id=%d, L=%d, lambda=%.3e, active=%d, physical_val_mse=%.6e"
        % (selected_id, result["selected_depth"], result["selected_lambda"], result["selected_connected_unit_count"], result["val_metrics"]["mse"]),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
