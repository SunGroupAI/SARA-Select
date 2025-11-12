#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARA-Select Feature Builder (frame-only, JSON-driven IN, scene-cache)
====================================================================

Author: ywyou
E-mail: ywyou96@gmail.com
Date: 2025-09-16
Version: 2.2 (JSON参数三态合并修复；SOAP维度可视；帧级-only；指纹缓存)

目标
----
从 three-stage pools（initial/min/all）构建用于 BRM/SARM/ALL 的帧级特征：
- B2: 组成/体积/密度/原子计数
- B3: 短程统计（最小间距、短程配对计数）
- B4: 配位直方（按元素 + 分箱）
- B5: RDF（元素对 × bins）
- B7: q4/q6 的分位统计（全局 + 分元素），可 --disable-q 关闭
- SOAP 聚合：对每帧原子 SOAP 向量做 mean/p90/p95/max/topk 聚合（不降维）

注意：不再生成“力”相关特征/标签。若提供 energy（.e.out），则写入：
- label_dE = E_DFT - E_DP   （帧级能量残差）
- E_dp_atom = E_DP / natoms  （帧级 DP 单原子能量，便于 per-atom 标定）

输入结构（与既有规范一致）
------------------------
project_root/
  initial-pool/ sceneA/1[/POSCAR] 2 3 ...
  initial-energy/ sceneA.e.out (col1=E_DFT, col2=E_DP)
  min-pool/      # SARM 用
  min-energy/
  all-pool/      # ALL 无 energy

两种“IN”入口
-----------
A) JSON（推荐）：
  python prepare_features.py build-from-in --in ./in_features.json
  * 所有参数都可写在 JSON 中；命令行仅需 --in
  * 若命令行也提供了同名参数，将覆盖 JSON（CLI 优先）

B) IN 目录（向后兼容）：
  IN_DIR/
    initial-pool      # 文件内写路径（相对/绝对）
    initial-energy
    min-pool
    min-energy
    all-pool
  用法：python prepare_features.py build-from-in --in ./IN_DIR --type-map Li,Zr,Cl --outdir ./SARA_features
  其他参数走命令行或默认

输出
----
outdir/<stage>/
  parts/<stage>_frame_<scene>.parquet
  <stage>_frame.parquet
  logs/errors_<stage>.log
  cache/scene_fingerprints/<scene>.json

依赖（建议单独 conda 环境）
--------------------------
conda install -c conda-forge numpy scipy pandas pyarrow fastparquet tqdm colorama
conda install -c conda-forge pymatgen ase dscribe
"""

import os, sys, re, json, math, time, shutil, argparse, inspect, traceback, random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar

# ────────── 彩色输出（可选） ──────────
try:
    from colorama import init as _cinit, Fore as F, Style as S
    _cinit()
    RED, GRN, YEL, BLU, MAG, CYN = F.RED, F.GREEN, F.YELLOW, F.BLUE, F.MAGENTA, F.CYAN
    BLD, RST = S.BRIGHT, S.RESET_ALL
except Exception:
    RED=GRN=YEL=BLU=MAG=CYN=""; BLD=RST=""

def banner(msg:str):
    line="═"*max(28,len(msg)+2)
    print(f"\n{BLU}{BLD}╔{line}╗\n║ {msg} ║\n╚{line}╝{RST}", flush=True)
def section(msg:str):
    print(f"\n{MAG}{BLD}── {msg} ─────────────────────────────────────{RST}", flush=True)
def kv(k,v): return f"{BLD}{k}{RST}={v}"
def ensure_dir(d:str): os.makedirs(d, exist_ok=True)
NUMERIC_RE = re.compile(r"^\d+$")

# ────────── 依赖探测 ──────────
_HAS_DSCRIBE=_HAS_ASE=_HAS_PMG_AAA=False
try:
    from dscribe.descriptors import SOAP
    import dscribe
    _HAS_DSCRIBE=True
except Exception:
    _HAS_DSCRIBE=False
try:
    from ase import Atoms
    from ase.data import atomic_numbers as ASE_Z
    _HAS_ASE=True
except Exception:
    _HAS_ASE=False
try:
    from pymatgen.io.ase import AseAtomsAdaptor
    _HAS_PMG_AAA=True
except Exception:
    _HAS_PMG_AAA=False

# ────────── I/O 工具 ──────────
def safe_write_df(df: pd.DataFrame, out_path: str, kind: str):
    try:
        df.to_parquet(out_path, index=False)
        print(f"{GRN}[OK]{RST} {kind:<22} → {out_path}")
    except Exception:
        alt=re.sub(r"\.parquet$", ".csv", out_path)
        df.to_csv(alt, index=False)
        print(f"{YEL}[WARN]{RST} {kind:<22} parquet 不可用，改写 CSV → {alt}")

def append_errors(logdir: str, stage: str, lines: List[str]):
    if not lines: return
    ensure_dir(logdir)
    path=os.path.join(logdir, f"errors_{stage}.log")
    with open(path,"a",encoding="utf-8") as f:
        for ln in lines: f.write(ln.rstrip()+"\n")

def write_run_meta(logdir: str, stage: str, meta: dict):
    ensure_dir(logdir)
    path=os.path.join(logdir, f"run_{stage}.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False)+"\n")

# ────────── 结构文件读取 ──────────
def list_scenes(root: str) -> List[str]:
    if not os.path.isdir(root): raise FileNotFoundError(root)
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def list_poscars(scene_dir: str) -> List[Tuple[int,str]]:
    out=[]
    for d in os.listdir(scene_dir):
        dpath=os.path.join(scene_dir,d)
        if os.path.isdir(dpath) and NUMERIC_RE.match(d):
            p=os.path.join(dpath,"POSCAR")
            if os.path.isfile(p): out.append((int(d),p))
    for f in os.listdir(scene_dir):
        fpath=os.path.join(scene_dir,f)
        if os.path.isfile(fpath) and NUMERIC_RE.match(f):
            out.append((int(f), fpath))
    out.sort(key=lambda x:x[0]); return out

def load_structure(path: str) -> Structure:
    name=os.path.basename(path)
    try:
        if os.path.splitext(name)[1] or name.upper()=="POSCAR":
            return Structure.from_file(path)
        return Poscar.from_file(path).structure
    except Exception:
        return Poscar.from_file(path).structure

# ────────── SOAP 工具 ──────────
def make_soap(species: List[str], rcut: float, nmax: int, lmax: int, sigma: float):
    if not (_HAS_DSCRIBE and _HAS_ASE):
        raise RuntimeError("缺少 dscribe 或 ase：`conda install -c conda-forge dscribe ase`")
    sig=inspect.signature(SOAP)
    params=sig.parameters
    k_r='r_cut' if 'r_cut' in params else ('rcut' if 'rcut' in params else None)
    k_n='n_max' if 'n_max' in params else ('nmax' if 'nmax' in params else None)
    k_l='l_max' if 'l_max' in params else ('lmax' if 'lmax' in params else None)
    kw={}
    if k_r and k_n and k_l:
        kw[k_r]=rcut; kw[k_n]=nmax; kw[k_l]=lmax
        if 'sigma' in params: kw['sigma']=sigma
        if 'species' in params: kw['species']=list(species)
        if 'periodic' in params: kw['periodic']=True
        if 'sparse' in params: kw['sparse']=False
        if 'average' in params: kw['average']="off"
        return SOAP(**kw)
    return SOAP(rcut, nmax, lmax, sigma)

def pmg_to_ase_atoms(st: Structure):
    if _HAS_PMG_AAA:
        return AseAtomsAdaptor.get_atoms(st)
    if not _HAS_ASE:
        raise RuntimeError("缺少 ase，无法构造 Atoms")
    numbers=[ASE_Z[str(site.specie)] for site in st.sites]
    positions=st.cart_coords
    cell=st.lattice.matrix
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=(True,True,True))

def calc_soap_matrix(st: Structure, soap_obj) -> np.ndarray:
    atoms=pmg_to_ase_atoms(st)
    X=soap_obj.create(atoms)  # (n_atoms, D)
    X=np.asarray(X,float)
    X[~np.isfinite(X)] = np.nan
    return X

# ────────── 帧级特征 ──────────
def neighbors(st: Structure, rc: float):
    try: return st.get_all_neighbors(rc, include_index=True)
    except Exception: return [[] for _ in range(len(st))]

def B2_comp(st: Structure, type_map: List[str]) -> Dict[str,float]:
    n=len(st); vol=float(st.lattice.volume)
    dens=(n/vol if vol>0 else np.nan)
    row=dict(natoms=n, volume=vol, density=dens, vol_per_atom=(vol/n if n>0 else np.nan))
    counts={sym:0 for sym in type_map}
    for sp in st.species: counts[str(sp)]=counts.get(str(sp),0)+1
    tot=max(1,sum(counts.values()))
    for sym,c in counts.items():
        row[f"count_{sym}"]=c; row[f"frac_{sym}"]=c/tot
    return row

def B3_short(st: Structure, type_map: List[str], short_thr: float) -> Dict[str,float]:
    n=len(st); neigh=neighbors(st, short_thr)
    dmin=dmin_same=dmin_diff=np.inf
    counts={f"short_pairs_{a}-{b}":0 for a in type_map for b in type_map}
    for i in range(n):
        si=st[i].species_string
        for nb in neigh[i]:
            j=nb.index
            if j<=i: continue
            sj=st[j].species_string
            try: d=st.get_distance(i,j)
            except Exception: d=float(np.linalg.norm(st[j].coords-st[i].coords))
            if d<dmin: dmin=d
            if si==sj: dmin_same=min(dmin_same,d)
            else:      dmin_diff=min(dmin_diff,d)
            key=f"short_pairs_{si}-{sj}"; rkey=f"short_pairs_{sj}-{si}"
            if key in counts: counts[key]+=1
            else: counts[rkey]+=1
    return dict(dmin=np.nan if dmin==np.inf else dmin,
                dmin_same=np.nan if dmin_same==np.inf else dmin_same,
                dmin_diff=np.nan if dmin_diff==np.inf else dmin_diff,
                **counts)

def B4_coord_hist(st: Structure, type_map: List[str], rc: float, bins: List[Tuple[int,int]]) -> Dict[str,float]:
    neigh=neighbors(st, rc)
    cn=np.array([len(li) for li in neigh], int)
    hist={}; syms=[str(sp) for sp in st.species]
    for sym in type_map:
        idx=[i for i,s in enumerate(syms) if s==sym]
        vals=cn[idx] if idx else np.array([], int)
        total=max(1,len(vals))
        for lo,hi in bins:
            if hi<0: c=int(np.sum(vals>=lo)); name=f"coord_hist_{sym}_{lo}+"
            else:    c=int(np.sum((vals>=lo)&(vals<=hi))); name=f"coord_hist_{sym}_{lo}-{hi}"
            hist[name]=c/total
    return hist

def B5_rdf(st: Structure, type_map: List[str], rmin: float, rmax: float, nbins: int) -> Dict[str,float]:
    bins=np.linspace(rmin, rmax, nbins+1)
    pairs=[(a,b) for i,a in enumerate(type_map) for j,b in enumerate(type_map) if j>=i]
    acc={(a,b): np.zeros(nbins, float) for (a,b) in pairs}
    neigh=neighbors(st, rmax)
    n=len(st)
    for i in range(n):
        si=st[i].species_string
        for nb in neigh[i]:
            j=nb.index
            if j<=i: continue
            sj=st[j].species_string
            try: d=st.get_distance(i,j)
            except Exception: d=float(np.linalg.norm(st[j].coords-st[i].coords))
            if not (rmin<=d<rmax): continue
            k=np.searchsorted(bins, d)-1
            a,b=(si,sj) if type_map.index(si)<=type_map.index(sj) else (sj,si)
            acc[(a,b)][k]+=1.0
    row={}
    for (a,b),h in acc.items():
        s=h.sum(); norm=(h/s) if s>0 else np.zeros_like(h)
        for t,val in enumerate(norm):
            row[f"rdf_{a}-{b}_bin{t}"]=float(val)
    return row

def A6_ql(st: Structure, rc: float, l_list=(4,6)) -> np.ndarray:
    try: from scipy.special import sph_harm
    except Exception: return np.full((len(st), len(l_list)), np.nan, float)
    neigh=neighbors(st, rc)
    n=len(st); out=np.zeros((n, len(l_list)), float)
    for i in range(n):
        try:
            ri=st[i].coords
            vs=[nb.coords - ri for nb in neigh[i]]
            if not vs: out[i,:]=np.nan; continue
            vs=np.asarray(vs,float)
            r=np.linalg.norm(vs,axis=1); m=r>1e-12
            if not np.any(m): out[i,:]=np.nan; continue
            vs=vs[m]; r=r[m]
            x,y,z=vs[:,0],vs[:,1],vs[:,2]
            theta=np.arccos(np.clip(z/r,-1.0,1.0)); phi=np.arctan2(y,x)
            for li,l in enumerate(l_list):
                ms=np.arange(-l,l+1)
                Y=np.stack([np.atleast_1d(sph_harm(m,l,phi,theta)) for m in ms], axis=1)
                Ym=np.mean(Y,axis=0)
                ql=np.sqrt(4*np.pi/(2*l+1) * np.sum(np.abs(Ym)**2))
                out[i,li]=float(np.real(ql))
        except Exception:
            out[i,:]=np.nan
    return out

# ────────── 工具：分箱解析、能量读取、指纹 ──────────
def parse_bins(coord_bins: str)->List[Tuple[int,int]]:
    out=[]
    for token in str(coord_bins).split(","):
        t=token.strip()
        if not t: continue
        if "+" in t:
            lo=int(t.replace("+","")); out.append((lo,-1))
        elif "-" in t:
            lo,hi=t.split("-"); out.append((int(lo),int(hi)))
    return out if out else [(0,-1)]

def read_energy_file(efile: str)->np.ndarray:
    try:
        arr=np.loadtxt(efile, skiprows=1)
    except Exception:
        df=pd.read_csv(efile, sep=r"\s+", header=None, comment="#", engine="python")
        arr=df.to_numpy()
    return np.atleast_2d(arr)  # col1=E_DFT, col2=E_DP

def path_sig(p:str)->str:
    try: st=os.stat(p); return f"{os.path.abspath(p)}|{st.st_mtime_ns}|{st.st_size}"
    except Exception: return f"{os.path.abspath(p)}|NA|NA"

def scene_fingerprint(scene_dir:str, energy_file:Optional[str])->dict:
    pairs=list_poscars(scene_dir)
    items=[[int(sid), path_sig(p)] for sid,p in pairs]
    return dict(scene=os.path.basename(scene_dir),
                n=len(pairs),
                energy=path_sig(energy_file) if (energy_file and os.path.exists(energy_file)) else "NA",
                poscars=items)

def same_fingerprint(a:dict,b:dict)->bool:
    return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)

# ────────── 每帧任务 ──────────
def frame_task(args):
    (scene, sid, path, type_map, coord_cutoffs, coord_bins,
     rdf_rmin, rdf_rmax, rdf_nbins, short_thr, q_cutoff, disable_q,
     soap_enable, soap_params, soap_agg_topk) = args
    try:
        st=load_structure(path); n=len(st)
        # q4/q6
        q_local = (np.full((n,2), np.nan, float) if disable_q else A6_ql(st, q_cutoff, (4,6)))
        # SOAP → 帧聚合
        Bsoap={}
        if soap_enable:
            soap=make_soap(type_map, **soap_params)
            S=calc_soap_matrix(st, soap)  # (n,D)
            if S is not None and S.size>0:
                k=max(3,int(soap_agg_topk))
                meanv=np.nanmean(S, axis=0)
                p90v =np.nanquantile(S,0.90,axis=0)
                p95v =np.nanquantile(S,0.95,axis=0)
                maxv =np.nanmax(S, axis=0)
                sort_idx=np.argsort(S, axis=0)
                top_idx =sort_idx[-k:,:] if S.shape[0]>=k else sort_idx
                topv   =np.take_along_axis(S, top_idx, axis=0).mean(axis=0)
                for d in range(meanv.shape[0]):
                    Bsoap[f"soap_mean_{d}"]=float(meanv[d])
                    Bsoap[f"soap_p90_{d}"] =float(p90v[d])
                    Bsoap[f"soap_p95_{d}"] =float(p95v[d])
                    Bsoap[f"soap_max_{d}"] =float(maxv[d])
                    Bsoap[f"soap_topk_{d}"]=float(topv[d])

        # 组装帧级
        B2=B2_comp(st, type_map)
        B3=B3_short(st, type_map, short_thr)
        bins=parse_bins(coord_bins); rc_hist=max(coord_cutoffs) if coord_cutoffs else 3.5
        B4=B4_coord_hist(st, type_map, rc_hist, bins)
        B5=B5_rdf(st, type_map, rdf_rmin, rdf_rmax, rdf_nbins)

        # q 统计
        syms=[str(sp) for sp in st.species]
        qstats={}
        for j,nm in enumerate(("q4","q6")):
            arr=q_local[:,j]
            qstats[f"{nm}_p50"]=float(np.nanmedian(arr))
            qstats[f"{nm}_p90"]=float(np.nanquantile(arr,0.90))
            qstats[f"{nm}_p95"]=float(np.nanquantile(arr,0.95))
            for sym in type_map:
                idx=[i for i,s in enumerate(syms) if s==sym]
                vals=arr[idx] if idx else np.array([])
                qstats[f"{nm}_{sym}_p50"]=float(np.nanmedian(vals)) if vals.size else np.nan
                qstats[f"{nm}_{sym}_90"]=float(np.nanquantile(vals,0.90)) if vals.size else np.nan
                qstats[f"{nm}_{sym}_95"]=float(np.nanquantile(vals,0.95)) if vals.size else np.nan

        row=dict(scene=scene, sid=int(sid), **B2, **B3, **B4, **B5, **qstats, **Bsoap)
        return ("ok", row)
    except Exception as e:
        return ("err", scene, sid, f"{type(e).__name__}: {e}", traceback.format_exc())

# ────────── 核心构建 ──────────
def build_stage(stage_name: str,
                pool_root: str,
                energy_root: Optional[str],
                outdir: str,
                type_map: List[str],
                coord_cutoffs: List[float],
                coord_bins: str,
                rdf_rmin: float, rdf_rmax: float, rdf_nbins: int,
                short_thr: float,
                q_cutoff: float, disable_q: bool,
                soap_enable: bool,
                soap_rcut: float, soap_nmax: int, soap_lmax: int, soap_sigma: float,
                soap_agg_topk: int,
                workers: int, mp_start: str,
                resume: bool, clean_stage: bool,
                require_soap: bool):

    banner(f"BUILD {stage_name} (frame-only, JSON-ready)")
    print("配置 |", kv("pool", pool_root), kv("energy", energy_root if energy_root else "N/A"), kv("type_map", ",".join(type_map)))
    print("A3   |", kv("cutoffs", coord_cutoffs), kv("bins", coord_bins))
    print("RDF  |", kv("rmin", rdf_rmin), kv("rmax", rdf_rmax), kv("nbins", rdf_nbins))
    print("Q    |", kv("q_cutoff", q_cutoff), kv("disable_q", disable_q))
    print("SOAP |", kv("enable", soap_enable), kv("rcut", soap_rcut), kv("nmax", soap_nmax), kv("lmax", soap_lmax), kv("sigma", soap_sigma))
    print("I/O  |", kv("outdir", outdir), kv("stage", stage_name))

    stage_dir=os.path.join(outdir, stage_name)
    parts_dir=os.path.join(stage_dir, "parts")
    log_dir  =os.path.join(stage_dir, "logs")
    cache_dir=os.path.join(stage_dir, "cache", "scene_fingerprints")
    if clean_stage and os.path.isdir(stage_dir):
        section(f"CLEAN {stage_dir}")
        shutil.rmtree(stage_dir, ignore_errors=True)
    ensure_dir(parts_dir); ensure_dir(log_dir); ensure_dir(cache_dir)

    out_frame=os.path.join(stage_dir, f"{stage_name.lower()}_frame.parquet")

    scenes=list_scenes(pool_root)

    # SOAP 预检（打印预计维度）
    if soap_enable:
        if not (_HAS_DSCRIBE and _HAS_ASE):
            sys.exit(f"{RED}ERROR{RST}: 启用了 SOAP，但 dscribe 或 ase 不可用")
        sample=[]
        for sc in scenes:
            pairs=list_poscars(os.path.join(pool_root, sc))
            if pairs: sample.append((sc, pairs[0][1]))
        if sample:
            sc0,p0=random.choice(sample)
            try:
                soap=make_soap(type_map, soap_rcut, soap_nmax, soap_lmax, soap_sigma)
                X0=calc_soap_matrix(load_structure(p0), soap)
                sig=inspect.signature(SOAP)
                print(f"[SOAP PROBE] dscribe={getattr(dscribe,'__version__','?')} signature={sig}")
                print(f"[SOAP PROBE] scene={sc0} atoms×dim={X0.shape[0]}×{X0.shape[1]}  agg_dim≈{5*X0.shape[1]}")
            except Exception as e:
                print(f"{RED}[SOAP PROBE FAIL]{RST} {type(e).__name__}: {e}")
                if require_soap: sys.exit("由于 --require-soap，直接退出")

    # 能量载入
    scene2E={}
    if energy_root:
        for sc in scenes:
            ef=os.path.join(energy_root, f"{sc}.e.out")
            if not os.path.isfile(ef):
                print(f"{YEL}[WARN]{RST} 缺少能量文件：{ef}")
                continue
            scene2E[sc]=read_energy_file(ef)

    # 逐 scene
    partial_frame_paths=[]
    import multiprocessing as mp
    ctx=mp.get_context(mp_start)
    for scene in scenes:
        section(f"SCENE {scene}")
        scene_dir=os.path.join(pool_root, scene)
        pairs=list_poscars(scene_dir)
        if not pairs:
            print(f"{YEL}[WARN]{RST} 无结构，跳过 {scene}"); continue

        # 指纹缓存
        ef=os.path.join(energy_root, f"{scene}.e.out") if energy_root else None
        cur_fp=scene_fingerprint(scene_dir, ef)
        fp_file=os.path.join(cache_dir, f"{scene}.json")
        partF=os.path.join(parts_dir, f"{stage_name}_frame_{scene}.parquet")
        if resume and os.path.exists(fp_file):
            try:
                old=json.load(open(fp_file,"r",encoding="utf-8"))
                if same_fingerprint(cur_fp, old) and os.path.exists(partF):
                    print(f"{CYN}[RESUME]{RST} 指纹未变，跳过：{scene}")
                    partial_frame_paths.append(partF)
                    continue
            except Exception:
                pass

        # energy 行数校验
        if energy_root and scene in scene2E:
            E_scene=scene2E[scene]
            if E_scene.shape[0]!=len(pairs):
                print(f"{RED}[ERR]{RST} {scene}.e.out 行数({E_scene.shape[0]}) != 帧数({len(pairs)})，跳过")
                continue

        # 任务
        soap_params=dict(rcut=soap_rcut, nmax=soap_nmax, lmax=soap_lmax, sigma=soap_sigma)
        tasks=[]
        for sid,p in pairs:
            tasks.append((scene, sid, p, type_map, [float(x) for x in coord_cutoffs], coord_bins,
                          float(rdf_rmin), float(rdf_rmax), int(rdf_nbins),
                          float(short_thr), float(q_cutoff), bool(disable_q),
                          bool(soap_enable), soap_params, int(32)))

        if workers<=1:
            results=list(tqdm(map(frame_task, tasks), total=len(tasks), desc=f"{stage_name}:{scene}"))
        else:
            with ctx.Pool(processes=workers) as pool:
                results=list(tqdm(pool.imap_unordered(frame_task, tasks,
                                  chunksize=max(8, len(tasks)//(workers*4) or 1)),
                                  total=len(tasks), desc=f"{stage_name}:{scene}"))

        rowsF=[]; err_lines=[]
        for res in results:
            status,*payload=res
            if status=="ok":
                rowsF.append(payload[0])
            else:
                _sc,_sid,msg,tb=payload
                err_lines.append(f"[{stage_name}] {scene}/{_sid} ERROR: {msg}\n{tb}")

        dfF=pd.DataFrame(rowsF)

        # 标签：有 energy 时写入 label_dE 与 E_dp_atom
        if energy_root and scene in scene2E and not dfF.empty:
            sid_list=[sid for sid,_ in pairs]
            E_scene=scene2E[scene]
            sid2dE={sid: float(E_scene[i,0]-E_scene[i,1]) for i,sid in enumerate(sid_list)}
            sid2Edp={sid: float(E_scene[i,1]) for i,sid in enumerate(sid_list)}
            dfF["label_dE"]=dfF["sid"].map(sid2dE)
            # E_dp_atom = E_DP / natoms
            dfF["E_dp_atom"]=dfF.apply(lambda r: (sid2Edp.get(int(r["sid"]), np.nan)/max(1, int(r["natoms"]))), axis=1)

        safe_write_df(dfF, partF, f"frame[{scene}]")
        partial_frame_paths.append(partF)
        if err_lines:
            append_errors(log_dir, stage_name, err_lines)
            print(f"{YEL}[WARN]{RST} {scene}: {len(err_lines)} 帧异常，详见 logs/errors_{stage_name}.log")

        # 保存指纹
        try:
            json.dump(cur_fp, open(fp_file,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 汇总
    def concat_parts(paths: List[str]) -> pd.DataFrame:
        dfs=[]
        for p in paths:
            if p.endswith(".parquet") and os.path.exists(p):
                try: dfs.append(pd.read_parquet(p)); continue
                except Exception: pass
            alt=re.sub(r"\.parquet$", ".csv", p)
            if os.path.exists(alt): dfs.append(pd.read_csv(alt))
        return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()

    dfF_all=concat_parts(partial_frame_paths)
    if dfF_all.empty:
        print(f"{RED}[ERROR]{RST} 未收集到任何帧级数据，请检查输入与日志。")
    safe_write_df(dfF_all, out_frame, f"frame(all:{stage_name})")

    # 维度报告
    n_cols=dfF_all.shape[1]
    n_soap=sum(c.startswith(("soap_mean_","soap_p90_","soap_p95_","soap_max_","soap_topk_")) for c in dfF_all.columns)
    section("SUMMARY")
    print(kv("frames_total", dfF_all.shape[0]),
          kv("feat_dim", n_cols),
          kv("soap_agg_cols", n_soap),
          kv("scenes", len(scenes)),
          kv("out", out_frame))
    if soap_enable and n_soap==0:
        print(f"{YEL}[WARN]{RST} SOAP 启用但未生成列，请检查 dsribe/ase 或结构转换。")
    print(f"{GRN}[DONE]{RST} {stage_name} 完成")

# ────────── IN 读取 ──────────
def read_in_dir(in_path:str)->List[Tuple[str,str,Optional[str]]]:
    def _read_one(fn):
        p=os.path.join(in_path, fn)
        if not os.path.isfile(p): return None
        return open(p,"r",encoding="utf-8").read().strip()
    mp=[]
    ip=_read_one("initial-pool"); ie=_read_one("initial-energy")
    if ip: mp.append(("BRM", ip, ie))
    mp_=_read_one("min-pool"); me=_read_one("min-energy")
    if mp_: mp.append(("SARM", mp_, me))
    ap=_read_one("all-pool")
    if ap: mp.append(("ALL", ap, None))
    return mp

def read_in_json(in_path:str)->Tuple[dict, List[Tuple[str,str,Optional[str]]]]:
    obj=json.load(open(in_path,"r",encoding="utf-8"))
    stages=[]
    for it in obj.get("stages", []):
        name=str(it.get("name","")).upper()
        if name not in ("BRM","SARM","ALL"): continue
        pool=it.get("pool"); energy=it.get("energy")
        if not pool: continue
        stages.append((name, pool, energy))
    return obj, stages

# ────────── 参数合并（JSON ←→ CLI） ──────────
def parse_args():
    p=argparse.ArgumentParser(description="SARA-Select: 构建帧级特征（JSON 一把梭；CLI 可覆盖）")
    sub=p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--outdir", type=str, required=False)
        sp.add_argument("--type-map", type=str, required=False)
        sp.add_argument("--coord-cutoffs", type=str, default=None)
        sp.add_argument("--coord-bins", type=str, default=None)
        sp.add_argument("--rdf-rmin", type=float, default=None)
        sp.add_argument("--rdf-rmax", type=float, default=None)
        sp.add_argument("--rdf-nbins", type=int, default=None)
        sp.add_argument("--short-threshold", type=float, default=None)
        sp.add_argument("--q-cutoff", type=float, default=None)

        # 三态布尔：None 表示“未从 CLI 指定”，避免覆盖 JSON
        sp.add_argument("--disable-q", action=argparse.BooleanOptionalAction, default=None)
        sp.add_argument("--soap-enable", action=argparse.BooleanOptionalAction, default=None)
        sp.add_argument("--require-soap", action=argparse.BooleanOptionalAction, default=None)
        sp.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
        sp.add_argument("--clean-stage", action=argparse.BooleanOptionalAction, default=None)

        sp.add_argument("--soap-rcut", type=float, default=None)
        sp.add_argument("--soap-nmax", type=int, default=None)
        sp.add_argument("--soap-lmax", type=int, default=None)
        sp.add_argument("--soap-sigma", type=float, default=None)
        sp.add_argument("--soap-agg-topk", type=int, default=None)
        sp.add_argument("--workers", type=int, default=None)
        sp.add_argument("--mp-start", type=str, default=None, choices=["spawn","forkserver","fork"])

    spA=sub.add_parser("build", help="构建单一 stage（BRM/SARM/ALL）")
    spA.add_argument("--stage", type=str, required=True, choices=["BRM","SARM","ALL"])
    spA.add_argument("--pool", type=str, required=True)
    spA.add_argument("--energy", type=str, help="BRM/SARM 需提供；ALL 无")
    common(spA)

    spB=sub.add_parser("build-from-in", help="从 IN 目录 或 JSON 批量构建")
    spB.add_argument("--in", dest="in_path", type=str, required=True)
    common(spB)

    return p.parse_args()

def pick(v_cli, v_json, v_def):
    if v_cli not in (None, "", []): return v_cli
    if v_json not in (None, "", []): return v_json
    return v_def

def pick_bool(v_cli, v_json, v_def: bool) -> bool:
    if v_cli is True or v_cli is False: return bool(v_cli)
    if v_json is True or v_json is False: return bool(v_json)
    return bool(v_def)

def main():
    args=parse_args()

    if args.cmd=="build":
        # 单阶段：CLI 主导（便于临时测试）
        if not args.type_map: sys.exit(f"{RED}ERROR{RST}: --type-map 不能为空")
        if not args.outdir:   sys.exit(f"{RED}ERROR{RST}: --outdir 不能为空")
        type_map=[x.strip() for x in args.type_map.split(",") if x.strip()]
        coord_cutoffs=[float(x) for x in (args.coord_cutoffs or "3.5").split(",") if x.strip()]
        build_stage(args.stage, args.pool, args.energy, args.outdir, type_map,
                    coord_cutoffs, args.coord_bins or "0-2,3-4,5-6,7+",
                    args.rdf_rmin or 0.8, args.rdf_rmax or 5.0, args.rdf_nbins or 12,
                    args.short_threshold or 2.0, args.q_cutoff or 4.5,
                    pick_bool(args.disable_q, None, False),
                    pick_bool(args.soap_enable, None, False),
                    args.soap_rcut or 5.0, args.soap_nmax or 4, args.soap_lmax or 6, args.soap_sigma or 0.5,
                    args.soap_agg_topk or 32,
                    args.workers or 1, args.mp_start or "spawn",
                    pick_bool(args.resume, None, True), pick_bool(args.clean_stage, None, False),
                    pick_bool(args.require_soap, None, False))
        return

    # build-from-in
    in_path=args.in_path
    json_cfg=None
    if os.path.isdir(in_path):
        stages=read_in_dir(in_path)
        if not args.type_map or not args.outdir:
            sys.exit(f"{RED}ERROR{RST}: IN 目录模式需要 CLI 提供 --type-map 与 --outdir")
        type_map=[x.strip() for x in args.type_map.split(",") if x.strip()]
        outdir=args.outdir
        coord_cutoffs=[float(x) for x in (args.coord_cutoffs or "3.5").split(",") if x.strip()]
        coord_bins=args.coord_bins or "0-2,3-4,5-6,7+"
        rdf_rmin=args.rdf_rmin or 0.8; rdf_rmax=args.rdf_rmax or 5.0; rdf_nbins=args.rdf_nbins or 12
        short_thr=args.short_threshold or 2.0
        q_cutoff=args.q_cutoff or 4.5
        disable_q=pick_bool(args.disable_q, None, False)
        soap_enable=pick_bool(args.soap_enable, None, False)
        soap_rcut=args.soap_rcut or 5.0; soap_nmax=args.soap_nmax or 4; soap_lmax=args.soap_lmax or 6; soap_sigma=args.soap_sigma or 0.5
        soap_agg_topk=args.soap_agg_topk or 32
        workers=args.workers or 1; mp_start=args.mp_start or "spawn"
        resume=pick_bool(args.resume, None, True)
        clean_stage=pick_bool(args.clean_stage, None, False)
        require_soap=pick_bool(args.require_soap, None, False)
    else:
        json_cfg, stages = read_in_json(in_path)
        if not stages:
            sys.exit(f"{RED}ERROR{RST}: JSON 中未提供 stages")
        outdir = pick(args.outdir, json_cfg.get("outdir"), "./SARA_features")
        tm_json = json_cfg.get("type_map")
        type_map=[x.strip() for x in pick(args.type_map, (",".join(tm_json) if tm_json else None), "").split(",") if x.strip()]
        if not type_map: sys.exit(f"{RED}ERROR{RST}: 未提供 type_map（JSON 或 CLI 任一即可）")

        coord_cutoffs = pick(args.coord_cutoffs, json_cfg.get("coord_cutoffs"), "3.5")
        coord_cutoffs = [float(x) for x in coord_cutoffs.split(",")] if isinstance(coord_cutoffs, str) else list(map(float, coord_cutoffs))
        coord_bins    = pick(args.coord_bins, json_cfg.get("coord_bins"), "0-2,3-4,5-6,7+")
        rdf = json_cfg.get("rdf", {})
        rdf_rmin = float(pick(args.rdf_rmin, rdf.get("rmin"), 0.8))
        rdf_rmax = float(pick(args.rdf_rmax, rdf.get("rmax"), 5.0))
        rdf_nbins= int(pick(args.rdf_nbins, rdf.get("nbins"), 12))
        short_thr= float(pick(args.short_threshold, json_cfg.get("short_threshold"), 2.0))
        q_cutoff = float(pick(args.q_cutoff, json_cfg.get("q_cutoff"), 4.5))
        disable_q= pick_bool(args.disable_q, json_cfg.get("disable_q"), False)
        soap_cfg = json_cfg.get("soap", {})
        soap_enable= pick_bool(args.soap_enable,  soap_cfg.get("enable"), False)
        soap_rcut  = float(pick(args.soap_rcut,  soap_cfg.get("rcut"), 6.0))
        soap_nmax  = int(pick(args.soap_nmax,  soap_cfg.get("nmax"), 4))
        soap_lmax  = int(pick(args.soap_lmax,  soap_cfg.get("lmax"), 6))
        soap_sigma = float(pick(args.soap_sigma, soap_cfg.get("sigma"), 0.5))
        soap_agg_topk = int(pick(args.soap_agg_topk, json_cfg.get("soap_agg_topk"), 32))
        workers    = int(pick(args.workers, json_cfg.get("workers"), 16))
        mp_start   = pick(args.mp_start, json_cfg.get("mp_start"), "spawn")
        resume     = pick_bool(args.resume,       json_cfg.get("resume"), True)
        clean_stage= pick_bool(args.clean_stage,  json_cfg.get("clean_stage"), False)
        require_soap=pick_bool(args.require_soap, json_cfg.get("require_soap"), True)

    banner("BUILD-FROM-IN")
    print("IN   |", kv("path", in_path), kv("stages", len(stages)))
    print("CONF |", kv("outdir", outdir), kv("type_map", ",".join(type_map)))
    print("PAR  |", kv("soap_enable", soap_enable), kv("workers", workers), kv("resume", resume))

    # 逐阶段执行
    for name,pool,energy in stages:
        build_stage(name, pool, energy, outdir, type_map,
                    coord_cutoffs, coord_bins, rdf_rmin, rdf_rmax, rdf_nbins,
                    short_thr, q_cutoff, disable_q,
                    soap_enable, soap_rcut, soap_nmax, soap_lmax, soap_sigma,
                    soap_agg_topk, workers, mp_start, resume, clean_stage,
                    require_soap)

if __name__=="__main__":
    main()

