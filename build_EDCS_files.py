#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDCS Builder (E, D, C, S generators)
====================================

Author: ywyou
E-mail: ywyou96@gmail.com
Date  : 2025-09-20

目标
----
从已生成的帧级特征与预测结果中，构建四类文件：
- E：不确定度代理（UE），来自 ALL 对 SARM-total 的 |pred|（per-atom）。按“大场景”（macro_scene）做归一化。
- D：多样性。用 (SOAP 聚合 + 配位直方图 等) 做 PCA-2D。输出：
      1) D.csv：initial 的 2D 坐标；all 的 2D 坐标 + 到 initial 最近邻距离（raw 和分场景归一）。
      2) D_new.csv：基于 all 的“场景内贪心最远点”打分（从 1 均匀衰减到 0）。
- C：覆盖。横轴为 Li/Cl 或 Zr/Cl（同时输出 linear 与 log10 两套），纵轴为密度 ρ。
      对 ALL（分大场景）与 initial（全局）做 5×5 分箱，给出各格的计数与覆盖分。
- S：短程风险（S2）。对 ALL 每帧每原子对（6 种）取 top-10 最小距离（r_search=4Å）。
      用区间 [1.5Å, 2.0Å] 线性插值打分：r≤1.5→1，r≥2.0→0。输出原子对 60 条/帧（不足则 <60）。
      同时给帧级聚合分（S_max/S_mean/violations）。

注意
----
- “大场景（macro_scene）”：将 amorphous_1、amorphous_2 … 归为 amorphous；
  规则由配置中的 `macro_scene` 定义（strip 后缀 + 正则分组 + 回退策略）。
- S 的 POSCAR 读取更鲁棒：既支持 scene/sid/POSCAR，也支持 sid 本身就是 POSCAR 文件（scene/sid）。

输入（来自 JSON 配置）
--------------------
- paths.brm_frame  : initial 的帧级特征 parquet（BRM）
- paths.all_frame  : ALL 的帧级特征 parquet
- paths.all_compare: ALL 的 BRM vs SARM-total 对比 CSV（含 pred_SARM_total_pa 等）
- paths.initial_pool_root: initial-pool 的几何根目录（计算 S 用）
- paths.all_pool_root    : all-pool 的几何根目录（计算 S 用）

关键列（columns）：
- scene, sid
- density
- counts: {Li, Zr, Cl} → 分别在帧表里的列名（用于 C 的比值）
- ue_col: 用作 E 的 |pred| 列名（通常 'pred_SARM_total_pa'）

超参（可在 JSON 中调节）
----------------------
- pca:
    feature_prefixes: 用于 D 的特征前缀（默认覆盖 soap_* 聚合 + coord_hist_ + rdf_ + q*）
    standardize     : 是否标准化（建议 True）
    n_components    : 2
- S:
    r_search     : 4.0 Å
    risk_low     : 1.5 Å
    risk_high    : 2.0 Å
    topk_per_pair: 10
    elements     : ["Li","Zr","Cl"]
- C:
    xbins, ybins : 5, 5
    输出同时包含：x_scale ∈ {linear, log} × binning ∈ {equal, quantile}
- E:
    ue_threshold_meV: 5.0  # 仅用于 keep 过滤，归一化不受此阈值影响

输出
----
outdir/EDCS/
  E.csv            # UE（raw、keep、按大场景归一）
  D.csv            # initial 的 pc1/pc2；all 的 pc1/pc2 + D_raw + D_norm_macro
  D_new.csv        # 在 D.csv 基础上增加 D_greedy（场景内贪心最远点线性衰减得分）
  C.csv            # 每帧的 (x, ρ) 所在栅格计数：initial 全局计数 + ALL（分大场景）计数 + C_score
  S_pairs.csv      # 每帧每原子对的 top-10 最小距离（共最多 60 条）
  S_frame.csv      # 每帧聚合：S_max/S_mean/violations

用法
----
python build_EDCS_files.py --config ./edcs_config.json

参数解释与建议
--------------
- “大场景归一化”：E、D、D_new 的归一化在 macro_scene 内完成，避免子编号导致稀释。
- D 的 PCA 维度：由 initial+all 共同的特征列决定；若特征短缺（比如未生成 SOAP 聚合），D 仍可用剩余列。
- C 的“覆盖分”直觉：initial 在某格越拥挤，该格的 C_score 越小；ALL 更偏到 initial 稀疏处 → C_score 高。
- S 的效率：一次结构只构造一次邻域表；对每对元素单独维护一个 10 容量的小根堆加速 top-k。

"""
from __future__ import annotations
import os, re, json, math, argparse, warnings
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# 彩色输出
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    COK = Fore.GREEN + "[OK]" + Style.RESET_ALL
    CWN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
    CER = Fore.RED + "[ERROR]" + Style.RESET_ALL
    CINFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL
except Exception:
    COK = "[OK]"; CWN = "[WARN]"; CER = "[ERROR]"; CINFO = "[INFO]"

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# -------------------- 小工具 --------------------
def ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def banner(msg: str):
    line = "═" * max(10, len(msg) + 2)
    print(f"\n╔{line}╗\n║ {msg} ║\n╚{line}╝")

def section(msg: str):
    print(f"\n── {msg} ─────────────────────────────────────")

def macro_scene_of(name: str, cfg_macro: dict) -> str:
    s = str(name)
    for suf in cfg_macro.get("_strip_suffix", []):
        if s.endswith(suf): s = s[: -len(suf)]
    for patt, label in cfg_macro.get("regex_groups", []):
        if re.match(patt, s): return label
    if re.search(r"_\d+$", s): s = re.sub(r"_\d+$", "", s)
    if "_" in s: s = s.split("_", 1)[0]
    return s

def read_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

# -------------------- E --------------------
def build_E(cfg: dict, F_all: pd.DataFrame, macro_map: dict, outdir: str) -> pd.DataFrame:
    compare = read_df(cfg["paths"]["all_compare"])
    scene_col, sid_col = cfg["columns"]["scene"], cfg["columns"]["sid"]
    ue_col = cfg["columns"]["ue_col"]
    if ue_col not in compare.columns:
        raise KeyError(f"{CER} UE 列 {ue_col} 不在 {cfg['paths']['all_compare']}")
    df = compare[[scene_col, sid_col, ue_col]].copy().rename(columns={ue_col: "E_raw"})
    df["macro_scene"] = [macro_map.get(s, s) for s in df[scene_col].astype(str)]
    thr_eV = float(cfg["E"].get("ue_threshold_meV", 5.0)) / 1000.0
    df["E_keep"] = (df["E_raw"].abs() >= thr_eV).astype(int)
    def _norm(x):
        x = x.abs(); lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else x*0.0
    df["E_norm_macro"] = df.groupby("macro_scene", group_keys=False)["E_raw"].apply(_norm)
    df = df[["macro_scene", scene_col, sid_col, "E_raw", "E_keep", "E_norm_macro"]]
    out = os.path.join(outdir, "E.csv"); df.to_csv(out, index=False)
    print(f"{COK} 写出 E → {out}")
    return df

# -------------------- D --------------------
def pick_D_features(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    cols = []
    for p in prefixes: cols += [c for c in df.columns if c.startswith(p)]
    return sorted({c for c in cols if pd.api.types.is_numeric_dtype(df[c])})

def build_D(cfg: dict, F_brm: pd.DataFrame, F_all: pd.DataFrame, macro_map: dict, outdir: str):
    scene_col, sid_col = cfg["columns"]["scene"], cfg["columns"]["sid"]
    pca_cfg = cfg["pca"]
    prefixes = pca_cfg.get("feature_prefixes", [
        "soap_mean_", "soap_p90_", "soap_p95_", "soap_max_", "soap_topk_",
        "coord_hist_", "rdf_", "q4_", "q6_"
    ])
    feat_brm = pick_D_features(F_brm, prefixes)
    feat_all = pick_D_features(F_all, prefixes)
    feats = sorted(set(feat_brm).intersection(feat_all))
    if len(feats) < 5:
        warnings.warn(f"{CWN} D 可用特征过少（{len(feats)}），请检查 SOAP 聚合/直方图是否存在")
    X_init = F_brm[feats].to_numpy(float)
    X_all  = F_all[feats].to_numpy(float)
    if pca_cfg.get("standardize", True):
        sc = StandardScaler().fit(X_init)
        X_init = sc.transform(X_init); X_all = sc.transform(X_all)
    pca = PCA(n_components=int(pca_cfg.get("n_components", 2)))
    Z_init = pca.fit_transform(X_init); Z_all = pca.transform(X_all)
    nn = NearestNeighbors(n_neighbors=1).fit(Z_init)
    d_all, _ = nn.kneighbors(Z_all); d_raw = d_all.reshape(-1)
    df_init = pd.DataFrame({
        "source":"initial",
        "macro_scene":[macro_map.get(s,s) for s in F_brm[scene_col].astype(str)],
        scene_col:F_brm[scene_col].astype(str).values,
        sid_col:F_brm[sid_col].values,
        "pc1":Z_init[:,0], "pc2":Z_init[:,1]
    })
    df_all = pd.DataFrame({
        "source":"all",
        "macro_scene":[macro_map.get(s,s) for s in F_all[scene_col].astype(str)],
        scene_col:F_all[scene_col].astype(str).values,
        sid_col:F_all[sid_col].values,
        "pc1":Z_all[:,0], "pc2":Z_all[:,1],
        "D_raw":d_raw
    })
    def _norm_grp(g):
        lo, hi = float(g.min()), float(g.max())
        return (g-lo)/(hi-lo) if hi>lo else g*0.0
    df_all["D_norm_macro"] = df_all.groupby("macro_scene", group_keys=False)["D_raw"].apply(_norm_grp)
    D = pd.concat([df_init, df_all], ignore_index=True)
    out = os.path.join(outdir, "D.csv"); D.to_csv(out, index=False)
    print(f"{COK} 写出 D → {out}")
    # 贪心最远点（场景内）
    D_new = df_all.copy(); D_new["D_greedy"] = 0.0
    for ms, G in D_new.groupby("macro_scene"):
        base = df_init[["pc1","pc2"]].to_numpy()
        P = G[["pc1","pc2"]].to_numpy(); idxs = list(G.index)
        nn = NearestNeighbors(n_neighbors=1).fit(base)
        d,_ = nn.kneighbors(P); d = d.reshape(-1)
        order=[]
        for _ in range(len(P)):
            j = int(np.argmax(d)); order.append(j)
            base = np.vstack([base, P[j:j+1]])
            nn = NearestNeighbors(n_neighbors=1).fit(base)
            d,_ = nn.kneighbors(P); d = d.reshape(-1)
        score = np.linspace(1.0, 0.0, len(P)) if len(P)>1 else np.array([1.0])
        for rank, j in enumerate(order):
            D_new.loc[idxs[j], "D_greedy"] = float(score[rank])
    out2 = os.path.join(outdir, "D_new.csv"); D_new.to_csv(out2, index=False)
    print(f"{COK} 写出 D_new（贪心最远点） → {out2}")
    return D, D_new

# -------------------- C --------------------
def _safe_ratio(num: float, den: float) -> float:
    if den == 0: return float("inf") if num>0 else 0.0
    return num/den

def build_C(cfg: dict, F_brm: pd.DataFrame, F_all: pd.DataFrame, macro_map: dict, outdir: str) -> pd.DataFrame:
    scene_col, sid_col = cfg["columns"]["scene"], cfg["columns"]["sid"]
    dens_col = cfg["columns"]["density"]
    cLi, cZr, cCl = cfg["columns"]["counts"]["Li"], cfg["columns"]["counts"]["Zr"], cfg["columns"]["counts"]["Cl"]
    def _prep(df):
        g = pd.DataFrame({
            scene_col: df[scene_col].astype(str).values,
            sid_col: df[sid_col].values,
            "rho": pd.to_numeric(df[dens_col], errors="coerce").values,
            "x_li_cl_linear":[_safe_ratio(df[cLi].iloc[i], df[cCl].iloc[i]) for i in range(len(df))],
            "x_zr_cl_linear":[_safe_ratio(df[cZr].iloc[i], df[cCl].iloc[i]) for i in range(len(df))]
        })
        for k in ["x_li_cl_linear","x_zr_cl_linear"]:
            arr = g[k].astype(float).values
            with np.errstate(divide="ignore", invalid="ignore"):
                logv = np.log10(arr)
            logv[~np.isfinite(logv)] = np.nan
            g[k.replace("linear","log")] = logv
        return g
    G_init = _prep(F_brm); G_all = _prep(F_all)
    G_all["macro_scene"] = [macro_map.get(s, s) for s in F_all[scene_col].astype(str)]
    out_rows=[]
    def _fit_bins(vals, nbins, mode):
        v = vals[np.isfinite(vals)]
        if len(v)==0: return np.linspace(0,1,nbins+1)
        if mode=="equal":
            lo, hi = float(np.min(v)), float(np.max(v))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo: lo, hi = 0.0, 1.0
            return np.linspace(lo, hi, nbins+1)
        else:
            qs = np.linspace(0,1,nbins+1)
            return np.quantile(v, qs)
    def _digitize(v, edges):
        idx = np.digitize(v, edges[1:-1], right=False)
        idx[(v<edges[0]) | (v>edges[-1]) | ~np.isfinite(v)] = -1
        return idx
    for which in ["li_cl","zr_cl"]:
        for x_scale in ["linear","log"]:
            xcol = f"x_{which}_{x_scale}"
            for binmode in ["equal","quantile"]:
                xedges = _fit_bins(G_all[xcol].to_numpy(float), int(cfg["C"]["xbins"]), binmode)
                yedges = _fit_bins(G_all["rho"].to_numpy(float), int(cfg["C"]["ybins"]), binmode)
                xi = _digitize(G_init[xcol].to_numpy(float), xedges)
                yi = _digitize(G_init["rho"].to_numpy(float), yedges)
                key_init={}
                for a,b in zip(xi,yi):
                    if a<0 or b<0: continue
                    key_init[(int(a),int(b))] = key_init.get((int(a),int(b)),0)+1
                max_init = max(key_init.values()) if key_init else 1
                xa = _digitize(G_all[xcol].to_numpy(float), xedges)
                ya = _digitize(G_all["rho"].to_numpy(float), yedges)
                key_all_by_ms: Dict[str, Dict[Tuple[int,int], int]] = {}
                for ms,(a,b) in zip(G_all["macro_scene"], zip(xa,ya)):
                    if a<0 or b<0: continue
                    d = key_all_by_ms.setdefault(ms,{})
                    d[(int(a),int(b))] = d.get((int(a),int(b)),0)+1
                for i in range(len(G_all)):
                    ms = G_all["macro_scene"].iloc[i]
                    a,b = int(xa[i]), int(ya[i])
                    if a<0 or b<0:
                        init_cnt=0; all_cnt=0; c_score=1.0
                    else:
                        init_cnt = key_init.get((a,b),0)
                        all_cnt  = key_all_by_ms.get(ms,{}).get((a,b),0)
                        c_score  = 1.0 - (init_cnt/max_init if max_init>0 else 0.0)
                    out_rows.append(dict(
                        macro_scene=ms, scene=F_all[scene_col].iloc[i], sid=F_all[sid_col].iloc[i],
                        x_which="Li/Cl" if which=="li_cl" else "Zr/Cl",
                        x_scale=x_scale, binning=binmode,
                        x_value = float(G_all[xcol].iloc[i]) if np.isfinite(G_all[xcol].iloc[i]) else np.nan,
                        rho = float(G_all["rho"].iloc[i]) if np.isfinite(G_all["rho"].iloc[i]) else np.nan,
                        bin_x=a, bin_y=b, init_count=init_cnt, all_count=all_cnt, C_score=c_score
                    ))
    Cdf = pd.DataFrame(out_rows)
    out = os.path.join(outdir, "C.csv"); Cdf.to_csv(out, index=False)
    print(f"{COK} 写出 C → {out}")
    return Cdf

# -------------------- S --------------------
def _locate_structure(root: str, scene: str, sid) -> Optional[str]:
    sid_str = str(sid)
    cand = [
        os.path.join(root, scene, sid_str, "POSCAR"),
        os.path.join(root, scene, sid_str),
        os.path.join(root, scene, f"{sid_str}.vasp"),
        os.path.join(root, scene, f"POSCAR_{sid_str}"),
        os.path.join(root, scene, "POSCAR"),
        os.path.join(root, scene, "CONTCAR"),
    ]
    scene_dir = os.path.join(root, scene)
    if os.path.isdir(scene_dir):
        for nm in [sid_str, f"{sid_str}.vasp", f"{sid_str}.POSCAR"]:
            cand.append(os.path.join(scene_dir, nm))
        for nm in os.listdir(scene_dir):
            if nm.upper().startswith("POSCAR") or nm.upper().startswith("CONTCAR"):
                cand.append(os.path.join(scene_dir, nm))
    for p in cand:
        if os.path.isfile(p): return p
    return None

def _risk_score(d: float, lo: float, hi: float) -> float:
    if d <= lo: return 1.0
    if d >= hi: return 0.0
    return (hi - d) / (hi - lo)

def _spec_name(s: str) -> str:
    return re.sub(r"[^A-Za-z].*$","", str(s))

def build_S(cfg: dict, F_all: pd.DataFrame, macro_map: dict, outdir: str):
    scene_col, sid_col = cfg["columns"]["scene"], cfg["columns"]["sid"]
    root = cfg["paths"]["all_pool_root"]
    r_search = float(cfg["S"].get("r_search", 4.0))
    lo = float(cfg["S"].get("risk_low", 1.5))
    hi = float(cfg["S"].get("risk_high", 2.0))
    topk = int(cfg["S"].get("topk_per_pair", 10))
    elems = sorted(set(cfg["S"].get("elements", ["Li","Zr","Cl"])))
    # 统一“规范化原子对”以及遍历顺序
    PAIRS = [(a,b) for i,a in enumerate(elems) for b in elems[i:]]  # ('Cl','Cl'),('Cl','Li'),('Cl','Zr'),...
    rows_pairs, rows_frame = [], []
    for i in tqdm(range(len(F_all)), desc="S (pairs)"):
        sc = str(F_all[scene_col].iloc[i])
        ms = macro_map.get(sc, sc)
        sid = F_all[sid_col].iloc[i]
        poscar = _locate_structure(root, sc.replace("_min",""), sid)
        if not poscar:
            print(f"{CWN} S 跳过 {sc}/{sid}: 未找到结构"); continue
        try:
            if os.path.splitext(poscar)[1]=="" and os.path.basename(poscar).isdigit():
                st = Poscar.from_file(poscar).structure
            else:
                st = Structure.from_file(poscar)
        except Exception as e:
            print(f"{CWN} S 跳过 {sc}/{sid}: 读取失败 {e}"); continue
        neigh = st.get_all_neighbors(r_search, include_index=True)
        heap = defaultdict(list)  # key = canonical pair ('Cl','Li') etc.
        n = len(st)
        for ia in range(n):
            ea = _spec_name(st[ia].species_string)
            if ea not in elems: continue
            for nb in neigh[ia]:
                ib = nb.index
                if ib <= ia: continue
                eb = _spec_name(st[ib].species_string)
                if eb not in elems: continue
                pair = tuple(sorted((ea, eb)))
                d = float(st.get_distance(ia, ib))
                h = heap[pair]
                if len(h) < topk:
                    # 存 -d 构造“最大堆”以保留最小 topk
                    import heapq
                    heapq.heappush(h, -d)
                else:
                    import heapq
                    heapq.heappushpop(h, -d)
        # 输出
        frame_scores = []
        for pair in PAIRS:
            ds = sorted([-x for x in heap.get(pair, [])])  # 最小→最大
            if not ds: continue
            dmax = max(ds)
            for rk, d in enumerate(ds, 1):
                s = _risk_score(d, lo, hi)
                rows_pairs.append(dict(
                    macro_scene=ms, scene=sc, sid=sid,
                    pair=f"{pair[0]}-{pair[1]}", rank=rk, d=d,
                    d_norm_pair=(d/dmax if dmax>0 else np.nan), s_score=s
                ))
                frame_scores.append(s)
        if frame_scores:
            rows_frame.append(dict(
                macro_scene=ms, scene=sc, sid=sid,
                S_max=float(np.max(frame_scores)),
                S_mean=float(np.mean(frame_scores)),
                violations=int(np.sum(np.array(frame_scores)>=1.0-1e-12))
            ))
    S_pairs = pd.DataFrame(rows_pairs); S_frame = pd.DataFrame(rows_frame)
    out1 = os.path.join(outdir, "S_pairs.csv"); S_pairs.to_csv(out1, index=False)
    out2 = os.path.join(outdir, "S_frame.csv"); S_frame.to_csv(out2, index=False)
    print(f"{COK} 写出 S_pairs → {out1}")
    print(f"{COK} 写出 S_frame → {out2}")
    return S_pairs, S_frame

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser(description="Build E/D/C/S files with macro-scene normalization")
    ap.add_argument("--config", required=True, help="edcs_config.json")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    outdir = os.path.join(cfg["outdir"], "EDCS"); ensure_dir(outdir)
    banner("EDCS Builder")
    section("Load data")
    F_brm = read_df(cfg["paths"]["brm_frame"])
    F_all = read_df(cfg["paths"]["all_frame"])
    scene_col = cfg["columns"]["scene"]
    macro_map = {}
    for sc in pd.concat([F_brm[scene_col], F_all[scene_col]]).astype(str).unique():
        macro_map[sc] = macro_scene_of(sc, cfg.get("macro_scene", {}))
    print(f"{COK} initial rows={len(F_brm)}  all rows={len(F_all)}  macro_scenes={sorted(set(macro_map.values()))}")
    section("Build E"); _E = build_E(cfg, F_all, macro_map, outdir)
    section("Build D / D_new"); _D,_Dnew = build_D(cfg, F_brm, F_all, macro_map, outdir)
    section("Build C"); _C = build_C(cfg, F_brm, F_all, macro_map, outdir)
    section("Build S (short-range risk)"); _Sp,_Sf = build_S(cfg, F_all, macro_map, outdir)
    section("SUMMARY")
    print(f"{COK} E  : {os.path.join(outdir,'E.csv')}")
    print(f"{COK} D  : {os.path.join(outdir,'D.csv')}")
    print(f"{COK} D+ : {os.path.join(outdir,'D_new.csv')}")
    print(f"{COK} C  : {os.path.join(outdir,'C.csv')}")
    print(f"{COK} S* : {os.path.join(outdir,'S_pairs.csv')}  &  {os.path.join(outdir,'S_frame.csv')}")
    print(COK, "完成")

if __name__ == "__main__":
    main()

