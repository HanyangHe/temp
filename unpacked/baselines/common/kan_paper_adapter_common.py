from __future__ import annotations
import csv, json, math, os, random, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

def read_csv_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.empty((0,0), dtype=float)
    a = np.loadtxt(str(p), delimiter=',', ndmin=2)
    return np.asarray(a, dtype=float)

def write_csv_matrix(path: str, a: np.ndarray) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(a)
    if a.size == 0:
        p.write_text('', encoding='utf-8'); return
    np.savetxt(str(p), a, delimiter=',', fmt='%.17g')

def write_json(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def finite_float(x: Any, default: float = float('nan')) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def metrics(y: np.ndarray, yh: np.ndarray) -> Dict[str,float]:
    if y.size == 0 or yh.size == 0:
        return {k: float('nan') for k in ('mse','rmse','mae','nrmse','nrmseRange','nmae')}
    e = np.asarray(yh)-np.asarray(y)
    mse = float(np.mean(e*e)); rmse = math.sqrt(mse); mae = float(np.mean(np.abs(e)))
    ys = float(np.std(y)); yr = float(np.max(y)-np.min(y)); yma = float(np.mean(np.abs(y)))
    return dict(mse=mse, rmse=rmse, mae=mae,
                nrmse=rmse/ys if ys>0 else float('nan'),
                nrmseRange=rmse/yr if yr>0 else float('nan'),
                nmae=mae/yma if yma>0 else float('nan'))

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict): return {str(k): json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return json_safe(obj.tolist())
    if isinstance(obj, (np.floating,float)):
        v=float(obj); return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,int)): return int(obj)
    if isinstance(obj, (np.bool_,bool)): return bool(obj)
    return obj

def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    mu=np.mean(x,axis=0,keepdims=True); sd=np.std(x,axis=0,keepdims=True)
    sd=np.where(sd>1e-12,sd,1.0)
    return mu,sd

def standardize(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x-mu)/sd

def seed_all(seed: int, torch_module=None) -> None:
    random.seed(seed); np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available(): torch_module.cuda.manual_seed_all(seed)

def add_pykan_path(root: str) -> None:
    root=os.path.abspath(root)
    if not os.path.isdir(os.path.join(root,'kan')):
        raise FileNotFoundError(f'pykan root must contain kan/: {root}')
    if root not in sys.path: sys.path.insert(0,root)
