#!/usr/bin/env python3
"""KAN-paper-style MLP architecture/activation sweep on a MATLAB data split."""
from __future__ import annotations
import argparse, importlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
COMMON=HERE.parent/'common'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from kan_paper_adapter_common import (add_pykan_path, fit_standardizer, json_safe,
    metrics, read_csv_matrix, seed_all, standardize, write_csv_matrix, write_json)


class _QuietTqdm:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def set_description(self, *_args, **_kwargs):
        return None

def _activation(name, torch):
    n=name.strip().lower()
    if n=='tanh': return torch.nn.Tanh()
    if n=='relu': return torch.nn.ReLU()
    if n in ('silu','swish'): return torch.nn.SiLU()
    raise ValueError(f'Unsupported MLP activation: {name}')

def _predict(model, x, torch, y_mu, y_sd):
    with torch.no_grad():
        yp=model(torch.as_tensor(x, dtype=torch.get_default_dtype(), device=model.device)).detach().cpu().numpy()
    return yp*y_sd+y_mu

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True); ap.add_argument('--parent-pid'); ap.add_argument('--control-file')
    a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    t_total=time.perf_counter()
    paths=cfg['paths']; outdir=Path(cfg['work_dir']); outdir.mkdir(parents=True,exist_ok=True)
    Xtr=read_csv_matrix(paths['X_train']); Ytr=read_csv_matrix(paths['Y_train'])
    Xv=read_csv_matrix(paths['X_val']); Yv=read_csv_matrix(paths['Y_val'])
    Xt=read_csv_matrix(paths['X_test']); Yt=read_csv_matrix(paths['Y_test'])
    Xo=read_csv_matrix(paths['X_ood']); Yo=read_csv_matrix(paths['Y_ood'])
    if Ytr.shape[0]!=Xtr.shape[0]: raise ValueError('X_train/Y_train row mismatch')
    if cfg.get('normalize_inputs',True):
        xmu,xsd=fit_standardizer(Xtr)
    else: xmu=np.zeros((1,Xtr.shape[1])); xsd=np.ones((1,Xtr.shape[1]))
    if cfg.get('normalize_outputs',True):
        ymu,ysd=fit_standardizer(Ytr)
    else: ymu=np.zeros((1,Ytr.shape[1])); ysd=np.ones((1,Ytr.shape[1]))
    Xtrn=standardize(Xtr,xmu,xsd); Xvn=standardize(Xv,xmu,xsd); Xtn=standardize(Xt,xmu,xsd)
    Xon=standardize(Xo,xmu,xsd) if Xo.size else Xo
    Ytrn=standardize(Ytr,ymu,ysd); Yvn=standardize(Yv,ymu,ysd)

    add_pykan_path(cfg['pykan_root'])
    import torch
    torch.set_default_dtype(torch.float64 if str(cfg.get('dtype','float64')).lower()=='float64' else torch.float32)
    nt=int(cfg.get('torch_num_threads',0) or 0)
    if nt>0: torch.set_num_threads(nt)
    mlp_mod=importlib.import_module('kan.MLP')
    mlp_mod.tqdm=_QuietTqdm
    MLP=mlp_mod.MLP
    device=str(cfg.get('device','cpu'))
    seed=int(cfg.get('seed',1)); width=int(cfg.get('width',20))
    depths=[int(v) for v in cfg.get('depth_list',[2,3,4,5,6])]
    acts=[str(v).lower() for v in cfg.get('activation_list',['tanh','relu','silu'])]
    steps=int(cfg.get('steps',500)); opt=str(cfg.get('optimizer','LBFGS')); lr=float(cfg.get('learning_rate',1.0))
    dataset={
      'train_input':torch.as_tensor(Xtrn,dtype=torch.get_default_dtype(),device=device),
      'train_label':torch.as_tensor(Ytrn,dtype=torch.get_default_dtype(),device=device),
      'test_input':torch.as_tensor(Xvn,dtype=torch.get_default_dtype(),device=device),
      'test_label':torch.as_tensor(Yvn,dtype=torch.get_default_dtype(),device=device),
    }
    candidates=[]; selected=None; selected_model=None
    for depth in depths:
      # pykan depth = number of affine layers; therefore depth D has D-1 hidden layers.
      shape=[Xtr.shape[1]]+[width]*(max(depth-1,0))+[Ytr.shape[1]]
      for ia,act in enumerate(acts):
        nan_metrics={k:float('nan') for k in ('mse','rmse','mae','nrmse','nrmseRange','nmae')}
        c=dict(depth=depth, hidden_layer_count=max(depth-1,0), width=width, activation=act,
               shape=shape, seed=seed, status='failed', time_seconds=float('nan'),
               parameter_count=None, train_metrics=dict(nan_metrics), val_metrics=dict(nan_metrics),
               error='', traceback='')
        tc=time.perf_counter()
        try:
          seed_all(seed,torch)
          model=MLP(width=shape,seed=seed,device=device,save_act=False)
          model.act_fun=_activation(act,torch)
          model.fit(dataset,opt=opt,steps=steps,log=max(steps+1,1),lamb=0.,lr=lr,batch=-1,device=device)
          yp_tr=_predict(model,Xtrn,torch,ymu,ysd); yp_v=_predict(model,Xvn,torch,ymu,ysd)
          vm=metrics(Yv,yp_v)
          c.update(status='ok',time_seconds=time.perf_counter()-tc,
                   parameter_count=sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
                   train_metrics=metrics(Ytr,yp_tr),val_metrics=vm)
          key=(vm['mse'],c['parameter_count'],depth,ia)
          if np.isfinite(key[0]) and (selected is None or key<selected[0]):
            selected=(key,c); selected_model=model
        except Exception as e:
          c.update(time_seconds=time.perf_counter()-tc,error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(limit=8))
        candidates.append(c)
        print(f"[MLP sweep] depth={depth} width={width} act={act} status={c['status']} val_mse={c.get('val_metrics',{}).get('mse',float('nan')):.6e} time={c['time_seconds']:.3f}s",flush=True)
    if selected is None or selected_model is None: raise RuntimeError('All MLP sweep candidates failed.')
    sc=selected[1]
    yp_tr=_predict(selected_model,Xtrn,torch,ymu,ysd); yp_v=_predict(selected_model,Xvn,torch,ymu,ysd)
    yp_t=_predict(selected_model,Xtn,torch,ymu,ysd); yp_o=_predict(selected_model,Xon,torch,ymu,ysd) if Xo.size else np.empty((0,Ytr.shape[1]))
    write_csv_matrix(str(outdir/'Yhat_train.csv'),yp_tr); write_csv_matrix(str(outdir/'Yhat_val.csv'),yp_v)
    write_csv_matrix(str(outdir/'Yhat_test.csv'),yp_t); write_csv_matrix(str(outdir/'Yhat_ood.csv'),yp_o)
    result=dict(method='KAN-paper MLP sweep',protocol='kan_feynman_mlp_sweep',selection_metric='validation_mse',
      total_time_seconds=time.perf_counter()-t_total,selected_candidate_time_seconds=sc['time_seconds'],
      selected_depth=sc['depth'],selected_hidden_layer_count=sc['hidden_layer_count'],selected_width=sc['width'],
      selected_activation=sc['activation'],selected_shape=sc['shape'],parameter_count=sc['parameter_count'],
      train_metrics=metrics(Ytr,yp_tr),val_metrics=metrics(Yv,yp_v),test_metrics=metrics(Yt,yp_t),
      ood_metrics=metrics(Yo,yp_o),candidate_count=len(candidates),candidates=candidates,
      normalization=dict(x_mean=xmu,x_std=xsd,y_mean=ymu,y_std=ysd))
    write_json(str(outdir/'result.json'),json_safe(result))
    print(f"[MLP sweep] selected depth={sc['depth']} hidden={sc['hidden_layer_count']} width={width} activation={sc['activation']} val_mse={result['val_metrics']['mse']:.6e}",flush=True)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
