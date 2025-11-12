#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARA-Select — BRM + SARM (frame-only, GLOBAL + MACRO-SCENE)
===========================================================

Author: ywyou
E-mail: ywyou96@gmail.com
Date: 2025-09-16
Version: 4.0 (macro-scene grouping & routing; leak-proof; pretty logs)

目的
----
在帧级特征（不使用原子力）上训练：
1) BRM 基线：直接拟合帧级“每原子能量残差” label_dE_pa
2) SARM-total：汇总所有 *_min 小样本场景，基于 BRM 预测做全局残差学习（输入特征包含 pred_BRM_pa）
3) SARM-scene（宏场景）：把 amorphous_1_min、amorphous_2_min 等合并成一个宏场景 amorphous_min，
   再对每个宏场景做残差学习；推理 ALL 时按 amorphous_* → amorphous_min 路由。
4) 导出训练/测试对比与 ALL 对比（BRM vs SARM-total / BRM vs SARM-scene），均使用每原子量。

用法（JSON 驱动，支持“单文件汇总”或“多文件列表”）
---------------------------------------------------
# in_train.json（单文件示例）
{
  "outdir": "./SARA_features",
  "random_seed": 42,
  "feature_policy": "safe",
  "scene_suffix_min": "_min",
  "macro_min_samples": 50,                     // 宏场景最小样本数，不足则回退 SARM-total（默认 50）
  "brm": {
    "frames": ["./SARA_features/BRM/brm_frame.parquet"],
    "split": { "train_ratio": 0.8, "test_ratio": 0.2, "train_all": false }
  },
  "sarm": {
    "frames": ["./SARA_features/SARM/sarm_frame.parquet"],
    "test_ratio": 0.2
  },
  "all": {
    "frames": ["./SARA_features/ALL/all_frame.parquet"]
  }
}

# 运行
python train_sarm_residual.py --in ./in_train.json

重要列假设
---------
- BRM/SARM：必须含 scene, sid；且 label_dE_pa（或 label_dE + natoms 以便换算）
- ALL：必须含 scene, sid；不要求标签
- 特征建议：包含 SOAP 聚合、A3/A6/RDF 等帧级特征；SARM 阶段自动注入 pred_BRM_pa 作为特征

特性
----
- 自动“宏场景”分组：
  - SARM（含 _min）：^(.*?)(?:_[0-9]+)(_min)$ → scene_macro = <base>_min，否则保持原样
  - ALL（不含 _min）：^(.*?)(?:_[0-9]+)$      → scene_macro_all = <base>，否则保持原样
- 宏场景样本不足（默认 <50）时，ALL 上该宏路由自动回退 SARM-total（提示 [FALLBACK]）
- 防泄露：剔除 label_* / E_dp* / residual / scene / sid / natoms 等列；但保留 pred_BRM_pa 作为一个特征
- 屏幕输出：清晰美观 + 进度条 + 数据维度、样本数、R²/MAE/RMSE
- 导出文件（均为 CSV，列顺序与需求一致）：
  outdir/SARM_ADAPT/predictions/
    - BRM_train_test.csv
    - SARM_total_train_test.csv
    - SARM_scene_train_test_concat.csv           （拼接，新增 model_key=宏场景）
    - ALL_compare_BRM_vs_SARM_total.csv          （BRM vs SARM-total）
    - ALL_compare_BRM_vs_SARM_scene_concat.csv   （BRM vs SARM-scene 拼接，含 model_key）

依赖
----
conda install -c conda-forge pandas numpy scikit-learn tqdm joblib colorama pyarrow fastparquet
"""

import os, sys, re, json, argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump as joblib_dump, load as joblib_load

# ---------- 彩色美化 ----------
try:
    from colorama import init as _cinit, Fore as F, Style as S
    _cinit()
    RED, GRN, YEL, BLU, MAG, CYN = F.RED, F.GREEN, F.YELLOW, F.BLUE, F.MAGENTA, F.CYAN
    BLD, RST = S.BRIGHT, S.RESET_ALL
except Exception:
    RED=GRN=YEL=BLU=MAG=CYN=""; BLD=RST=""

def banner(msg: str):
    line = "═" * max(30, len(msg)+2)
    print(f"\n{BLU}{BLD}╔{line}╗\n║ {msg} ║\n╚{line}╝{RST}")

def section(msg: str):
    print(f"\n{MAG}{BLD}── {msg} ─────────────────────────────────────{RST}")

def kv(k,v): return f"{BLD}{k}{RST}={v}"

def ensure_dir(d: str): os.makedirs(d, exist_ok=True)

def read_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def load_many(paths: List[str], tag: str) -> pd.DataFrame:
    rows=[]
    for p in tqdm(paths, desc=f"load[{tag}]", ncols=100):
        df=read_table(p)
        df["_src_file"]=p
        rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()

# ---------- 标签/分列 ----------
def attach_label_pa(df: pd.DataFrame) -> pd.DataFrame:
    if "label_dE_pa" in df.columns:
        return df
    if ("label_dE" in df.columns) and ("natoms" in df.columns):
        out=df.copy()
        out["label_dE_pa"]=out["label_dE"]/out["natoms"].clip(lower=1)
        return out
    return df

# ---------- 宏场景解析 ----------
_re_sarm_macro = re.compile(r"^(.*?)(?:_([0-9]+))(_min)$")     # SARM：带 _min
_re_all_macro  = re.compile(r"^(.*?)(?:_([0-9]+))$")           # ALL ：不带 _min

def scene_macro_sarm(scene: str) -> str:
    s=str(scene)
    m=_re_sarm_macro.match(s)
    if m:
        base, _num, suf = m.groups()
        return f"{base}{suf}"   # <base>_min
    return s

def scene_macro_all(scene: str) -> str:
    s=str(scene)
    m=_re_all_macro.match(s)
    if m:
        base, _num = m.groups()
        return base
    return s

def scene_base_name(s: str, suffix_min: str) -> str:
    s=str(s)
    return s[:-len(suffix_min)] if (suffix_min and s.endswith(suffix_min)) else s

# ---------- 特征选择（防泄露） ----------
def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s)

LEAK_PREFIX = ("label_", "E_dp")
LEAK_EXACT  = {"scene","sid","natoms","label_dE_pa","_src_file","scene_base","residual", "scene_macro", "scene_macro_all"}

def pick_features(df: pd.DataFrame, policy: str="safe") -> List[str]:
    cols=[]
    for c in df.columns:
        if not is_numeric_series(df[c]): continue
        if policy=="safe":
            if c in LEAK_EXACT: continue
            if any(c.startswith(p) for p in LEAK_PREFIX): continue
        cols.append(c)
    # 去零方差
    keep=[]
    for c in cols:
        s=df[c].dropna()
        if s.empty: continue
        if s.min()==s.max(): continue
        keep.append(c)
    return keep

# ---------- 评价指标 ----------
def metrics(y_true, y_pred) -> Dict[str,float]:
    y_true=np.asarray(y_true, float); y_pred=np.asarray(y_pred, float)
    m=np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return dict(R2=np.nan, MAE=np.nan, RMSE=np.nan, N=0)
    r2=r2_score(y_true[m], y_pred[m])
    mae=mean_absolute_error(y_true[m], y_pred[m])
    rmse=float(np.sqrt(np.mean((y_true[m]-y_pred[m])**2)))
    return dict(R2=r2, MAE=mae, RMSE=rmse, N=int(m.sum()))

# ---------- 划分 ----------
def split_train_test_index(n: int, train_ratio: float, test_ratio: float, train_all: bool, seed: int):
    rng=np.random.default_rng(seed)
    idx=np.arange(n)
    rng.shuffle(idx)
    if train_all:
        n_test=max(1, int(round(test_ratio*n))) if test_ratio>0 else 0
        test_idx=set(idx[:n_test].tolist())
        train_idx=set(idx.tolist())   # 全部
    else:
        n_train=int(round(train_ratio*n))
        n_test =int(round(test_ratio*n))
        train_idx=set(idx[:n_train].tolist())
        test_idx =set(idx[n_train:n_train+n_test].tolist())
    return train_idx, test_idx

# ---------- 模型 ----------
def fit_regressor(X: np.ndarray, y: np.ndarray, seed: int):
    model=HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=None,
        learning_rate=0.1,
        max_iter=400,
        random_state=seed
    )
    model.fit(X, y)
    return model

# ---------- BRM ----------
def build_brm(df_brm: pd.DataFrame, feat_policy: str, split_cfg: dict, out_models_dir: str, seed: int):
    section("BRM baseline training")
    df=attach_label_pa(df_brm.copy())
    df=df[np.isfinite(df.get("label_dE_pa", np.nan))]
    if df.empty:
        raise RuntimeError("BRM 数据无 label_dE_pa（或无法换算），无法训练。")
    n0=len(df)
    feat_cols=pick_features(df, policy=feat_policy)
    X=df[feat_cols].to_numpy()
    y=df["label_dE_pa"].to_numpy()
    print(f"{CYN}数据集{RST} | rows={n0}  feat_dim={len(feat_cols)}")
    tr_idx, te_idx = split_train_test_index(
        n0,
        float(split_cfg.get("train_ratio",0.8)),
        float(split_cfg.get("test_ratio",0.2)),
        bool(split_cfg.get("train_all", False)),
        seed
    )
    print(f"{CYN}拆分{RST} | train={len(tr_idx)}  test={len(te_idx)}  train_all={bool(split_cfg.get('train_all', False))}")
    model=fit_regressor(X, y, seed)
    yhat=model.predict(X)
    m_tr=metrics(y[list(tr_idx)], yhat[list(tr_idx)])
    m_te=metrics(y[list(te_idx)], yhat[list(te_idx)])
    print(f"{GRN}指标{RST} | train: R2={m_tr['R2']:.4f} MAE={m_tr['MAE']:.6f} RMSE={m_tr['RMSE']:.6f}")
    print(f"{GRN}指标{RST} | test : R2={m_te['R2']:.4f} MAE={m_te['MAE']:.6f} RMSE={m_te['RMSE']:.6f}")
    ensure_dir(out_models_dir)
    joblib_dump(dict(model=model, feat_cols=feat_cols), os.path.join(out_models_dir, "frame_model.joblib"))
    print(f"{GRN}[OK]{RST} 保存 BRM 模型 → {os.path.join(out_models_dir,'frame_model.joblib')}")
    # 导出 BRM_train_test.csv
    df_out=df[["scene","sid"]].copy()
    df_out["train_true_pa"]=np.nan; df_out["train_pred_pa"]=np.nan
    df_out["test_true_pa"]=np.nan;  df_out["test_pred_pa"]=np.nan
    idx_arr=np.arange(n0)
    m_train=np.array([i in tr_idx for i in idx_arr])
    m_test =np.array([i in te_idx for i in idx_arr])
    df_out.loc[m_train,"train_true_pa"]=y[m_train]
    df_out.loc[m_train,"train_pred_pa"]=yhat[m_train]
    df_out.loc[m_test,"test_true_pa"]=y[m_test]
    df_out.loc[m_test,"test_pred_pa"]=yhat[m_test]
    return dict(model=model, feat_cols=feat_cols, df_train_test=df_out, metrics_train=m_tr, metrics_test=m_te)

def predict_with_brm(df: pd.DataFrame, brm_pack: dict) -> np.ndarray:
    # 缺列容忍：reindex 保留列顺序；HistGB 支持 NaN
    X=df.reindex(columns=brm_pack["feat_cols"], fill_value=np.nan)
    return brm_pack["model"].predict(X.to_numpy())

# ---------- SARM-total ----------
def build_sarm_total(df_sarm: pd.DataFrame, brm_pack: dict, feat_policy: str,
                     test_ratio: float, out_models_dir: str, seed: int):
    section("SARM-total (global residual) training")
    df=attach_label_pa(df_sarm.copy())
    if "label_dE_pa" not in df.columns:
        raise RuntimeError("SARM 数据无 label，可训练 SARM-total。")
    # 注入 BRM 预测
    df["pred_BRM_pa"]=predict_with_brm(df, brm_pack)
    n0=len(df)
    feat_cols=pick_features(df, policy=feat_policy)
    if "pred_BRM_pa" not in feat_cols:
        feat_cols=feat_cols+["pred_BRM_pa"]
    X=df[feat_cols].to_numpy()
    y=df["label_dE_pa"].to_numpy()
    print(f"{CYN}SARM-total 数据{RST} | rows={n0}  feat_dim={len(feat_cols)}  (含 pred_BRM_pa)")
    # 全样本训练 + 子集测试
    tr_idx, te_idx = split_train_test_index(n0, train_ratio=1.0, test_ratio=test_ratio, train_all=True, seed=seed)
    print(f"{CYN}拆分{RST} | train={len(tr_idx)} (ALL)  test={len(te_idx)} (subset)")
    model=fit_regressor(X, y, seed)
    yhat = model.predict(X)
    m_tr=metrics(y[list(tr_idx)], yhat[list(tr_idx)])
    m_te=metrics(y[list(te_idx)], yhat[list(te_idx)])
    print(f"{GRN}指标{RST} | train: R2={m_tr['R2']:.4f} MAE={m_tr['MAE']:.6f} RMSE={m_tr['RMSE']:.6f}")
    print(f"{GRN}指标{RST} | test : R2={m_te['R2']:.4f} MAE={m_te['MAE']:.6f} RMSE={m_te['RMSE']:.6f}")
    ensure_dir(out_models_dir)
    joblib_dump(dict(model=model, feat_cols=feat_cols), os.path.join(out_models_dir, "frame_model.joblib"))
    print(f"{GRN}[OK]{RST} 保存 SARM-total 模型 → {os.path.join(out_models_dir,'frame_model.joblib')}")
    # 导出 SARM_total_train_test.csv
    df_out=df[["scene","sid"]].copy()
    df_out["train_true_pa"]=y
    df_out["train_pred_pa"]=yhat
    df_out["test_true_pa"]=np.nan; df_out["test_pred_pa"]=np.nan
    idx_arr=np.arange(n0)
    m_test=np.array([i in te_idx for i in idx_arr])
    df_out.loc[m_test,"test_true_pa"]=y[m_test]
    df_out.loc[m_test,"test_pred_pa"]=yhat[m_test]
    return dict(model=model, feat_cols=feat_cols, df_train_test=df_out)

# ---------- SARM-scene（宏场景） ----------
def build_sarm_scene_macro(df_sarm: pd.DataFrame, brm_pack: dict, feat_policy: str,
                           test_ratio: float, out_models_dir: str, seed: int,
                           macro_min_samples: int) -> Dict[str, str]:
    section("SARM-scene (macro-scene residual) training")
    ensure_dir(out_models_dir)
    df=attach_label_pa(df_sarm.copy())
    if "label_dE_pa" not in df.columns:
        raise RuntimeError("SARM 数据无 label，无法训练 SARM-scene（宏）。")
    # 注入 BRM 预测 + 宏场景
    df["pred_BRM_pa"]=predict_with_brm(df, brm_pack)
    df["scene_macro"]=df["scene"].astype(str).map(scene_macro_sarm)

    rows_out=[]
    model_keys={}
    for macro in sorted(df["scene_macro"].unique().tolist()):
        sub=df[df["scene_macro"]==macro].copy()
        n0=len(sub)
        feat_cols=pick_features(sub, policy=feat_policy)
        if "pred_BRM_pa" not in feat_cols:
            feat_cols=feat_cols+["pred_BRM_pa"]
        X=sub[feat_cols].to_numpy()
        y=sub["label_dE_pa"].to_numpy()
        # 样本过少 → 跳过训练（回退 total）
        if n0 < int(macro_min_samples):
            print(f"  {YEL}[FALLBACK]{RST} {macro} | rows={n0} < {macro_min_samples}，该宏场景不训练，ALL 将回退 SARM-total")
            continue
        tr_idx, te_idx = split_train_test_index(n0, train_ratio=1.0, test_ratio=test_ratio, train_all=True, seed=seed)
        model=fit_regressor(X, y, seed)
        yhat=model.predict(X)
        m_tr=metrics(y[list(tr_idx)], yhat[list(tr_idx)])
        m_te=metrics(y[list(te_idx)], yhat[list(te_idx)])
        print(f"  {CYN}{macro}{RST} | rows={n0} feat_dim={len(feat_cols)}  train: R2={m_tr['R2']:.4f}  test: R2={m_te['R2']:.4f}")
        # 存模型
        macro_dir=os.path.join(out_models_dir, macro)
        ensure_dir(macro_dir)
        joblib_dump(dict(model=model, feat_cols=feat_cols), os.path.join(macro_dir, "frame_model.joblib"))
        model_keys[macro]=os.path.join(macro_dir,"frame_model.joblib")
        # 收集对比行（保留原始 scene/sid，新增 model_key=macro）
        tmp=sub[["scene","sid"]].copy()
        tmp["train_true_pa"]=y
        tmp["train_pred_pa"]=yhat
        tmp["test_true_pa"]=np.nan; tmp["test_pred_pa"]=np.nan
        idx_arr=np.arange(n0)
        m_test=np.array([i in te_idx for i in idx_arr])
        tmp.loc[m_test,"test_true_pa"]=y[m_test]
        tmp.loc[m_test,"test_pred_pa"]=yhat[m_test]
        tmp["model_key"]=macro
        rows_out.append(tmp)

    df_concat=pd.concat(rows_out, ignore_index=True, sort=False) if rows_out else pd.DataFrame(columns=["scene","sid","train_true_pa","train_pred_pa","test_true_pa","test_pred_pa","model_key"])
    return dict(df_train_test_concat=df_concat, model_keys=model_keys)

# ---------- 应用到 ALL ----------
def apply_all_predictions(df_all: pd.DataFrame,
                          brm_pack: dict,
                          sarm_total_pack: dict,
                          sarm_scene_models_dir: str,
                          macro_min_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    section("Apply to ALL: BRM vs SARM-total / SARM-scene")
    df=df_all.copy()
    # BRM
    yhat_brm=predict_with_brm(df, brm_pack)
    all_brm=pd.DataFrame(dict(scene=df["scene"], sid=df["sid"], pred_BRM_pa=yhat_brm))
    # SARM-total
    feat_cols_total=sarm_total_pack["feat_cols"]
    Xt=df.reindex(columns=feat_cols_total, fill_value=np.nan).copy()
    if "pred_BRM_pa" not in Xt.columns:
        Xt["pred_BRM_pa"]=yhat_brm
    for c in feat_cols_total:
        if c not in Xt.columns:
            Xt[c]=0.0
    yhat_total=sarm_total_pack["model"].predict(Xt.to_numpy())
    out_total=all_brm.merge(pd.DataFrame(dict(scene=df["scene"], sid=df["sid"], pred_SARM_total_pa=yhat_total)), on=["scene","sid"], how="left")
    print(f"{GRN}[OK]{RST} ALL: BRM vs SARM-total  rows={len(out_total)}")

    # SARM-scene（宏路由）
    # 预加载已有宏模型
    macro2pack={}
    if os.path.isdir(sarm_scene_models_dir):
        for macro in sorted(os.listdir(sarm_scene_models_dir)):
            model_path=os.path.join(sarm_scene_models_dir, macro, "frame_model.joblib")
            if os.path.isfile(model_path):
                macro2pack[macro]=joblib_load(model_path)

    df["scene_macro_all"]=df["scene"].astype(str).map(scene_macro_all)
    parts=[]
    for macro in sorted(df["scene_macro_all"].unique().tolist()):
        sub=df[df["scene_macro_all"]==macro].copy()
        if sub.empty: continue
        # 期望的宏模型键：macro_min（若 SARM 使用 _min 后缀）
        macro_sarm_key=f"{macro}_min"
        if macro_sarm_key in macro2pack:
            pack=macro2pack[macro_sarm_key]
            feat_cols=pack["feat_cols"]
            Xs=sub.reindex(columns=feat_cols, fill_value=np.nan).copy()
            if "pred_BRM_pa" not in Xs.columns:
                # 用 BRM 预测填充
                Xs["pred_BRM_pa"]=predict_with_brm(sub, brm_pack)
            for c in feat_cols:
                if c not in Xs.columns:
                    Xs[c]=0.0
            yhat=pack["model"].predict(Xs.to_numpy())
            parts.append(pd.DataFrame(dict(scene=sub["scene"], sid=sub["sid"], pred_SARM_scene_pa=yhat, model_key=macro_sarm_key)))
        else:
            # 回退到 SARM-total（该宏场景没有训练）
            print(f"  {YEL}[FALLBACK]{RST} ALL: {macro} 无 SARM-scene 模型 → 使用 SARM-total")
            parts.append(pd.DataFrame(dict(scene=sub["scene"], sid=sub["sid"],
                                           pred_SARM_scene_pa=yhat_total[sub.index],
                                           model_key="SARM_total_fallback")))
    out_scene = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=["scene","sid","pred_SARM_scene_pa","model_key"])
    out_scene = all_brm.merge(out_scene, on=["scene","sid"], how="left")  # 带上 pred_BRM_pa
    print(f"{GRN}[OK]{RST} ALL: BRM vs SARM-scene  rows={len(out_scene)} (concat)")
    return out_total, out_scene

# ---------- 主流程 ----------
def main():
    ap=argparse.ArgumentParser(description="Train BRM + SARM (frame-only, JSON-driven, macro-scene grouping)")
    ap.add_argument("--in", dest="in_path", required=True, help="JSON 配置文件路径（参见 in_train.json 示例）")
    args=ap.parse_args()

    cfg=json.load(open(args.in_path,"r",encoding="utf-8"))
    out_root=str(cfg.get("outdir","./SARA_features"))
    out_models=os.path.join(out_root,"SARM_ADAPT","models")
    out_preds =os.path.join(out_root,"SARM_ADAPT","predictions")
    ensure_dir(out_models); ensure_dir(out_preds)

    random_seed=int(cfg.get("random_seed",42))
    feat_policy=str(cfg.get("feature_policy","safe"))
    scene_suffix_min=str(cfg.get("scene_suffix_min","_min"))
    macro_min_samples=int(cfg.get("macro_min_samples",50))

    banner("SARM Residual Trainer (GLOBAL + MACRO-SCENE)")
    print("配置 |", kv("outdir", out_root), kv("seed", random_seed),
          kv("feat_policy", feat_policy), kv("suffix_min", scene_suffix_min),
          kv("macro_min_samples", macro_min_samples))

    # 1) 读取 BRM / SARM / ALL
    brm_files = list(cfg.get("brm",{}).get("frames",[]))
    sarm_files= list(cfg.get("sarm",{}).get("frames",[]))
    all_files = list(cfg.get("all",{}).get("frames",[]))
    if not brm_files: sys.exit(f"{RED}ERROR{RST}: brm.frames 为空")
    if not sarm_files: sys.exit(f"{RED}ERROR{RST}: sarm.frames 为空")
    if not all_files:  sys.exit(f"{RED}ERROR{RST}: all.frames 为空")

    section("Load datasets")
    F_brm = load_many(brm_files, "BRM")
    F_sarm= load_many(sarm_files,"SARM")
    F_all = load_many(all_files, "ALL")

    # scene_base（用于排序对齐）
    def base(s): return scene_base_name(str(s), scene_suffix_min)
    for df,tag in [(F_brm,"BRM"),(F_sarm,"SARM"),(F_all,"ALL")]:
        if "scene" not in df.columns or "sid" not in df.columns:
            sys.exit(f"{RED}ERROR{RST}: {tag} 缺少 scene/sid 列")
        df["scene"]=df["scene"].astype(str)
        df["scene_base"]=df["scene"].map(base)

    # 2) 训练 BRM
    brm_split=cfg.get("brm",{}).get("split", {"train_ratio":0.8,"test_ratio":0.2,"train_all":False})
    brm_pack = build_brm(F_brm, feat_policy, brm_split, os.path.join(out_models,"BRM"), random_seed)
    # 导出 BRM_train_test.csv
    brm_csv = brm_pack["df_train_test"][["scene","sid","train_true_pa","train_pred_pa","test_true_pa","test_pred_pa"]]
    brm_csv.to_csv(os.path.join(out_preds, "BRM_train_test.csv"), index=False)
    print(f"{GRN}[OK]{RST} 导出 → {os.path.join(out_preds,'BRM_train_test.csv')}")

    # 3) 训练 SARM-total（全局）
    sarm_total_pack = build_sarm_total(F_sarm, brm_pack, feat_policy,
                                       float(cfg.get("sarm",{}).get("test_ratio",0.2)),
                                       os.path.join(out_models,"SARM_total"), random_seed)
    sarm_total_csv = sarm_total_pack["df_train_test"][["scene","sid","train_true_pa","train_pred_pa","test_true_pa","test_pred_pa"]].copy()
    sarm_total_csv["scene_base"]=sarm_total_csv["scene"].astype(str).map(base)
    sarm_total_csv.sort_values(by=["scene_base","sid"], inplace=True)
    sarm_total_csv.drop(columns=["scene_base"], inplace=True)
    sarm_total_csv.to_csv(os.path.join(out_preds, "SARM_total_train_test.csv"), index=False)
    print(f"{GRN}[OK]{RST} 导出 → {os.path.join(out_preds,'SARM_total_train_test.csv')}")

    # 4) 训练 SARM-scene（宏场景）
    sarm_scene_pack = build_sarm_scene_macro(F_sarm, brm_pack, feat_policy,
                                             float(cfg.get("sarm",{}).get("test_ratio",0.2)),
                                             os.path.join(out_models,"SARM_scene"),
                                             random_seed, macro_min_samples)
    sarm_scene_csv = sarm_scene_pack["df_train_test_concat"][["scene","sid","train_true_pa","train_pred_pa","test_true_pa","test_pred_pa","model_key"]].copy()
    # 对齐顺序（参考 SARM_total）
    sarm_scene_csv["scene_base"]=sarm_scene_csv["scene"].astype(str).map(base)
    ref_order = sarm_total_pack["df_train_test"][["scene","sid"]].copy()
    ref_order["scene_base"]=ref_order["scene"].astype(str).map(base)
    ref_order["__ord"]=np.arange(len(ref_order))
    sarm_scene_csv = sarm_scene_csv.merge(ref_order[["scene_base","sid","__ord"]], on=["scene_base","sid"], how="left")
    missing=np.sum(~np.isfinite(sarm_scene_csv["__ord"]))
    if missing>0:
        print(f"{YEL}[WARN]{RST} SARM-scene 中有 {missing} 行无法与 SARM-total 顺序对齐，已保留相对顺序。")
        sarm_scene_csv["__ord_fill"]=sarm_scene_csv["__ord"].fillna(1e12)  # 放到最后
        sarm_scene_csv.sort_values(by=["__ord_fill","scene_base","sid"], inplace=True)
        sarm_scene_csv.drop(columns=["__ord_fill"], inplace=True)
    else:
        sarm_scene_csv.sort_values(by="__ord", inplace=True)
    sarm_scene_csv.drop(columns=["scene_base","__ord"], inplace=True)
    sarm_scene_csv.to_csv(os.path.join(out_preds, "SARM_scene_train_test_concat.csv"), index=False)
    print(f"{GRN}[OK]{RST} 导出 → {os.path.join(out_preds,'SARM_scene_train_test_concat.csv')}")

    # 5) ALL 对比：BRM vs SARM-total / SARM-scene（宏路由）
    F_all = F_all.copy()
    F_all["pred_BRM_pa"]=predict_with_brm(F_all, brm_pack)
    all_total, all_scene = apply_all_predictions(F_all, brm_pack, sarm_total_pack,
                                                 os.path.join(out_models,"SARM_scene"),
                                                 macro_min_samples)
    # 顺序对齐（参照 SARM_total 的 ALL 顺序）
    all_total["scene_base"]=all_total["scene"].astype(str).map(base)
    ref_all = all_total[["scene","sid","scene_base"]].copy()
    ref_all["__ord"]=np.arange(len(ref_all))
    all_scene["scene_base"]=all_scene["scene"].astype(str).map(base)
    all_scene = all_scene.merge(ref_all[["scene_base","sid","__ord"]], on=["scene_base","sid"], how="left")
    all_scene.sort_values(by=["__ord","scene_base","sid"], inplace=True)
    all_scene = all_scene[["scene","sid","pred_BRM_pa","pred_SARM_scene_pa","model_key"]]
    all_total = all_total[["scene","sid","pred_BRM_pa","pred_SARM_total_pa"]]
    # 导出
    all_total.to_csv(os.path.join(out_preds, "ALL_compare_BRM_vs_SARM_total.csv"), index=False)
    all_scene.to_csv(os.path.join(out_preds, "ALL_compare_BRM_vs_SARM_scene_concat.csv"), index=False)
    print(f"{GRN}[OK]{RST} 导出 → {os.path.join(out_preds,'ALL_compare_BRM_vs_SARM_total.csv')}")
    print(f"{GRN}[OK]{RST} 导出 → {os.path.join(out_preds,'ALL_compare_BRM_vs_SARM_scene_concat.csv')}")

    # 总结
    section("SUMMARY")
    print("模型 |", kv("BRM", os.path.join(out_models,"BRM")),
          kv("SARM_total", os.path.join(out_models,"SARM_total")),
          kv("SARM_scene", os.path.join(out_models,"SARM_scene")))
    print("对比 |", os.path.join(out_preds,"BRM_train_test.csv"))
    print("     |", os.path.join(out_preds,"SARM_total_train_test.csv"))
    print("     |", os.path.join(out_preds,"SARM_scene_train_test_concat.csv"))
    print("     |", os.path.join(out_preds,"ALL_compare_BRM_vs_SARM_total.csv"))
    print("     |", os.path.join(out_preds,"ALL_compare_BRM_vs_SARM_scene_concat.csv"))
    print(f"{GRN}[DONE]{RST} 训练完成")

if __name__=="__main__":
    main()

