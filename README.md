# SARA-Select
# README (English)

## 1. Project Overview

This repository contains the supplemental code used in the manuscript to build and analyze a scene-adaptive, data-efficient training workflow for interatomic potentials on Li–Zr–Cl (LZC)–type systems. The three scripts form a pipeline:

1. **`prepare_features.py`** – read structure pools from different “scenes” (BRM / SARM / ALL), compute frame-level descriptors, and export tidy tables for learning.
2. **`train_sarm_residual.py`** – train a baseline BRM model and scene-aware residual models (SARM) on the exported features, and compare their predictions on the full pool. 
3. **`build_EDCS_files.py`** – postprocess the features and predictions to build four auxiliary views (E/D/C/S) for uncertainty, diversity, coverage, and short-range risk. 

Together they let a reviewer or reader reproduce the data-engineering part of the paper: feature construction → scene-adaptive residual learning → EDCS analysis.

---

## 2. Repository Contents

* `prepare_features.py`
  Build frame-level feature tables from BRM/SARM/ALL structure pools. Input is either a JSON config or a directory of pool/energy files. Output is a set of parquet/CSV files per stage.
* `train_sarm_residual.py`
  Train BRM (global baseline) and SARM (global + macro-scene residuals) on the feature tables, and export prediction/diagnostic CSVs. 
* `build_EDCS_files.py`
  Build E (uncertainty proxy), D (PCA-based diversity), C (composition–density coverage), and S (short-range risk) files from the trained results. 
* (optional) example JSON configs: you are expected to provide your own paths according to your data layout.

---

## 3. Requirements

Use a clean Python 3.9–3.11 environment.

**Recommended packages**

```bash
conda install -c conda-forge \
  numpy pandas pyarrow fastparquet tqdm colorama \
  scikit-learn pymatgen ase dscribe
```

Notes:

* `pymatgen`, `ase`, `dscribe` are needed by the feature builder.
* `scikit-learn` is needed by the trainer and EDCS builder.
* `pymatgen` is also needed by `build_EDCS_files.py` to re-read POSCARs when computing short-range risks.

---

## 4. Usage

### 4.1 Step 1 — Build features

```bash
python prepare_features.py build-from-in --in in_features.json
```

Typical JSON (you should adjust paths):

```json
{
  "outdir": "./SARA_features",
  "type_map": ["Li", "Zr", "Cl"],
  "workers": 16,
  "stages": [
    {
      "name": "BRM",
      "pool": "/path/to/initial-pool",
      "energy": "/path/to/initial-energy"
    },
    {
      "name": "SARM",
      "pool": "/path/to/min-pool",
      "energy": "/path/to/min-energy"
    },
    {
      "name": "ALL",
      "pool": "/path/to/all-pool"
    }
  ]
}
```

After running, you should have something like:

```text
SARA_features/
  BRM/brm_frame.parquet
  SARM/sarm_frame.parquet
  ALL/all_frame.parquet
```

These three files are the main inputs for the next step.

---

### 4.2 Step 2 — Train BRM + SARM residual models

```bash
python train_sarm_residual.py --in in_train.json
```



Example `in_train.json`:

```json
{
  "outdir": "./SARA_features",
  "random_seed": 42,
  "feature_policy": "safe",
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
  },
  "scene_suffix_min": "_min",
  "macro_min_samples": 50
}
```

What this script does:

1. trains a **BRM** frame-level baseline on BRM data;
2. trains a **global SARM** residual model on SARM data using BRM prediction as an extra feature;
3. optionally trains **macro-scene SARM** models if a scene has enough samples;
4. applies BRM / SARM models to the **ALL** pool and exports comparison CSVs.

Main outputs (under `./SARA_features/SARM_ADAPT/`):

* `models/BRM/frame_model.joblib`
* `models/SARM_total/frame_model.joblib`
* `models/SARM_scene/<macro>/frame_model.joblib`
* `predictions/BRM_train_test.csv`
* `predictions/SARM_total_train_test.csv`
* `predictions/SARM_scene_train_test_concat.csv`
* `predictions/ALL_compare_BRM_vs_SARM_total.csv`
* `predictions/ALL_compare_BRM_vs_SARM_scene_concat.csv`

These prediction files are used by the EDCS builder.

---

### 4.3 Step 3 — Build EDCS views

```bash
python build_EDCS_files.py --config edcs_config.json
```



Example `edcs_config.json`:

```json
{
  "outdir": "./SARA_features",
  "paths": {
    "brm_frame": "./SARA_features/BRM/brm_frame.parquet",
    "all_frame": "./SARA_features/ALL/all_frame.parquet",
    "all_compare": "./SARA_features/SARM_ADAPT/predictions/ALL_compare_BRM_vs_SARM_total.csv",
    "initial_pool_root": "/path/to/initial-pool",
    "all_pool_root": "/path/to/all-pool"
  },
  "columns": {
    "scene": "scene",
    "sid": "sid",
    "density": "density",
    "counts": { "Li": "count_Li", "Zr": "count_Zr", "Cl": "count_Cl" },
    "ue_col": "pred_SARM_total_pa"
  },
  "macro_scene": {
    "_strip_suffix": ["_min"],
    "regex_groups": []
  },
  "pca": { "standardize": true, "n_components": 2 },
  "E": { "ue_threshold_meV": 5.0 },
  "C": { "xbins": 5, "ybins": 5 },
  "S": {
    "r_search": 4.0,
    "risk_low": 1.5,
    "risk_high": 2.0,
    "topk_per_pair": 10,
    "elements": ["Li", "Zr", "Cl"]
  }
}
```

This will create:

```text
SARA_features/EDCS/
  E.csv
  D.csv
  D_new.csv
  C.csv
  S_pairs.csv
  S_frame.csv
```

Interpretation:

* **E**: a simple uncertainty/priority signal per frame, normalized within each macro-scene.
* **D / D_new**: 2D PCA mapping + a greedy “furthest-point” score for diversity.
* **C**: composition–density grid coverage; helps to see whether ALL explores sparse regions.
* **S / S_frame**: short-range distance based risk scores, per pair and per frame.

---

## 5. Citation

If you use this repository for your paper or downstream work, please cite the accompanying manuscript once it is published.
