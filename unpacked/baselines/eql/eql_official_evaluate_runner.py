#!/usr/bin/env python3
"""Load one official EQL state and produce predictions for MATLAB."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def _load_dataset(path):
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def _write(path, a):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(a)
    if a.size == 0:
        p.write_text("", encoding="utf-8")
    else:
        np.savetxt(str(p), a, delimiter=",", fmt="%.17g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: np.asarray(a).item()  # type: ignore[attr-defined]
    flags = str(cfg.get("theano_flags", "")).strip()
    candidate_compiledir = str(cfg.get("candidate_compiledir", "")).strip()
    if candidate_compiledir:
        compiledir_path = Path(candidate_compiledir).resolve()
        compiledir_path.mkdir(parents=True, exist_ok=True)
        lowered = flags.lower()
        if "compiledir=" not in lowered and "base_compiledir=" not in lowered:
            # Match the candidate runner: exact ``compiledir`` avoids Theano's
            # long auto-generated platform suffix under ``base_compiledir``.
            private_flag = "compiledir=%s" % compiledir_path.as_posix()
            flags = private_flag if not flags else flags + "," + private_flag
    if flags:
        os.environ["THEANO_FLAGS"] = flags
    src_root = Path(cfg["official_src_root"]).resolve()
    sys.path.insert(0, str(src_root))
    import mlfg_final as official_mlfg  # noqa: E402

    datasets = _load_dataset(cfg["dataset_path"])
    with open(cfg["state_path"], "rb") as fh:
        state = pickle.load(fh, encoding="latin1")
    (xtr, ytr), (xv, yv), (xt, yt) = datasets
    rng = np.random.RandomState(int(cfg["candidate_id"]))
    classifier = official_mlfg.MLFG(
        rng=rng,
        n_in=int(xtr.shape[1]),
        n_per_base=int(cfg["units_per_type"]),
        n_out=int(ytr.shape[1]),
        n_layer=int(cfg["official_hidden_layers"]),
        gradient=str(cfg["gradient"]),
        basefuncs1=[0, 1, 2],
        basefuncs2=[0],
        with_shortcuts=False,
    )
    classifier.set_state(state)
    xtest_path = Path(cfg.get("xtest_path", ""))
    xtest = np.loadtxt(str(xtest_path), delimiter=",", ndmin=2) if xtest_path.exists() and xtest_path.stat().st_size else xt
    xood_path = Path(cfg["xood_path"])
    xood = np.loadtxt(str(xood_path), delimiter=",", ndmin=2) if xood_path.exists() and xood_path.stat().st_size else np.empty((0, xtr.shape[1]))
    _write(cfg["yhat_train_path"], classifier.evaluate(xtr))
    _write(cfg["yhat_val_path"], classifier.evaluate(xv))
    _write(cfg["yhat_test_path"], classifier.evaluate(xtest))
    _write(cfg["yhat_ood_path"], classifier.evaluate(xood) if xood.size else np.empty((0, ytr.shape[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
