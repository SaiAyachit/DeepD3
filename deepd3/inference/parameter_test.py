"""
parameter_test.py — DeepD3 parameter sweep using official deepd3 package.

Two ways of picking the settings used for the final outputs:
  (a) "chosen" — CHOSEN_CONFIG below, picked by hand
  (b) "stable" — the setting whose spine count changes least when peak threshold or
      seed delta move one grid step (or highest F1 when gt_spines is filled in)

Outputs (in OUT_DIR):
  - deepd3_benchmark_results.csv : metrics for every stack x model x parameter combo
  - deepd3_sweep_ranked.csv      : every model x setting averaged over stacks, ranked by (b)
  - deepd3_best_per_model.csv    : top setting per model under (b)
  - deepd3_viz_chosen.png / deepd3_viz_stable.png
                                 : one random plane of one random stack showing the
                                   dendrite + spine segmentation, a model comparison and
                                   the peak x seed-delta sweep
  - deepd3_spines_chosen.csv / deepd3_spines_stable.csv
                                 : one row per spine (position, volume, brightness)
  - deepd3_spine_summary.csv     : per stack and selection: spine count, median volume,
                                   median brightness
"""
#python deepd3/inference/parameter_test.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from glob import glob
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import cc3d
import imageio as io
import tensorflow as tf
import matplotlib.pyplot as plt
import psutil

# Suppress Keras per-batch progress bars
_orig_predict = tf.keras.Model.predict
def _silent_predict(self, *args, **kwargs):
    kwargs['verbose'] = 0
    return _orig_predict(self, *args, **kwargs)
tf.keras.Model.predict = _silent_predict

from deepd3.core.analysis import Stack, _get_sorted_seeds, connected_components_3d


# ============================================================
# Configuration
# ============================================================
models = ["DeepD3_32F", "DeepD3_16F", "DeepD3_8F"]

MODEL_DIR = "deepd3/inference/pretrained_models"
model_paths = {m: f"{MODEL_DIR}/{m}.h5" for m in models}

stack_paths = sorted(glob("deepd3/inference/data/processed/*.tif"))

# Predictions are cached here so re-runs skip the (slow) network inference
CACHE_DIR = Path("deepd3/inference/predictions")
OUT_DIR = Path("deepd3/inference/results")

# Set to an int for a reproducible choice of stack / plane, None for a new one each run
RANDOM_SEED = None
# Side length (px) of the zoomed view in the visualization
VIZ_CROP = 256

# Voxel size in µm
DIMENSIONS = {'xy': 0.145, 'z': 1.0}
VOXEL_UM3 = DIMENSIONS['xy'] ** 2 * DIMENSIONS['z']

# Sweep grid
# The region of interest is peak 0.8-0.9 x delta 0.1-0.3 (peak < 0.8 fragments spines;
# 1.0 finds ~nothing). The grid extends one step past it on every side so that every
# setting in that region has 4 neighbours for the stability score (b); edge settings
# are measured but not ranked.
roi_peak_thresholds = [0.75, 0.8, 0.85, 0.9, 0.95]
# Seed delta sets how far an ROI grows from its seed: voxels join if p >= (1 - delta) * seed
roi_seed_deltas = [0.05, 0.1, 0.2, 0.3, 0.4]
# Size limits in µm³ (converted to voxels with VOXEL_UM3), so they keep their meaning
# if the voxel size changes. Bright spines look 5-9 µm long in z (axial blur), so single
# spines reach 4-8 µm³; a 4 µm³ cap drops ~14% of them. The soma (hundreds of µm³) is
# still rejected by every max value here.
min_sizes_um3 = [0.05, 0.1, 0.2]
max_sizes_um3 = [4.0, 8.0, 16.0]
# Spine depth: number of z-planes (1 µm each) an ROI spans. Fixed at 2-9 planes; kept as
# lists so they can be swept again. (At peak 0.9 / delta 0.2 with no cap, ~60% of
# spines span 2-4 planes and ~2% span more than 9.)
min_planes = [2]
max_planes = [9]

# (a) Settings picked by hand; every value must be in the sweep grid above.
# peak 0.9 / delta 0.2: counts stable across neighbouring settings, ROIs cover the
# whole spine head without merging into the shaft.
CHOSEN_CONFIG = {"model": "DeepD3_32F", "peak_threshold": 0.9, "seed_delta": 0.2,
                 "min_size_um3": 0.1, "max_size_um3": 8.0, "min_planes": 2, "max_planes": 9}

# Fixed parameters
# Not swept: a voxel only joins if p >= (1 - delta) * seed >= (1 - delta) * peak, so any
# area threshold below that (0.56 for peak 0.8, delta 0.3) has no effect.
ROI_AREA_THRESHOLD = 0.25
ROI_DISTANCE_TO_SEED = 10
DEND_THRESHOLD = 0.7
DEND_MIN_SIZE = 100
DEND_DILATION = 11
TILE_SIZE = 128
INSET_SIZE = 96

# (b) picks model, peak and delta by stability but keeps CHOSEN_CONFIG's size filters:
# stability differences between size filters are noise-level (a few spines) and would
# otherwise pick filters for noise-level reasons.
# Settings whose instability is within this of the best are reported as tied.
STABILITY_TIE_TOL = 0.01

# Brightness (standard 2P measure): background-subtracted fluorescence summed in a FIXED
# ellipsoid around the brightest voxel of each spine ROI — fixed so the value does not
# depend on how far the flood fill grew (seed delta) — divided by the median fluorescence
# of the core (p > SHAFT_CORE_THRESHOLD) of the nearest dendrite within a local window.
SPINE_HEAD_RADIUS_XY_UM = 0.6
SPINE_HEAD_RADIUS_Z_UM = 1.0
SHAFT_WINDOW_XY_UM = 3.0
SHAFT_WINDOW_Z_UM = 2.0
SHAFT_CORE_THRESHOLD = 0.9
SHAFT_MIN_VOXELS = 200
# A shaft dimmer than this fraction of the stack's median shaft is not a usable reference
SHAFT_MIN_REL = 0.25
# Voxels at or above this fraction of the dtype maximum count as saturated
SATURATION_FRAC = 0.95

# Ground truth (optional). Points are voxel coordinates (x, y, z). "region" is the
# annotated box (z0, z1, y0, y1, x0, x1) — only ROIs inside it are scored, so spines on
# unannotated dendrites are not counted as false positives. None = whole stack.
gt_spines = {}  # {stack_name: {"points_xyz": [(x, y, z), ...], "region": (z0, z1, y0, y1, x0, x1)}}
GT_MATCH_UM = 2.0


def _mem_guard(threshold_gb=24.0):
    """Raise if current process RSS exceeds threshold."""
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    if rss_gb > threshold_gb:
        raise MemoryError(f"RSS {rss_gb:.1f} GB exceeds {threshold_gb} GB — aborting to avoid OOM kill")

# ============================================================
# Helpers
# ============================================================
def xyz_to_um(points_xyz):
    """(N, 3) voxel coordinates (x, y, z) -> µm."""
    p = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    return p * [DIMENSIONS['xy'], DIMENSIONS['xy'], DIMENSIONS['z']]


def load_raw(stack_path):
    # Same loader as deepd3.core.analysis.Stack
    return np.asarray(io.mimread(stack_path, memtest=False))


# ============================================================
# Inference (cached on disk)
# ============================================================
def _prediction_params():
    return {"tile_size": TILE_SIZE, "inset_size": INSET_SIZE,
            "dend_threshold": DEND_THRESHOLD, "dend_dilation": DEND_DILATION}


def _save_atomic(fn, arr):
    # Write to a temp file first so a killed run never leaves a truncated cache file
    tmp = fn.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, fn)


def _get_prediction(model_name, stack_path):
    stem = f"{Path(stack_path).stem}__{model_name}"
    dend_fn = CACHE_DIR / f"{stem}__dend.npy"
    spine_fn = CACHE_DIR / f"{stem}__spine.npy"
    params_fn = CACHE_DIR / f"{stem}__params.json"
    if dend_fn.exists() and spine_fn.exists():
        params = _prediction_params()
        if not params_fn.exists():
            # Cache written before params were recorded — made with the current defaults
            params_fn.write_text(json.dumps(params))
        if json.loads(params_fn.read_text()) == params:
            print(f"  {model_name}: loaded cached prediction")
            return np.load(dend_fn), np.load(spine_fn)
        print(f"  {model_name}: cached prediction used other inference settings — recomputing")

    print(f"  Loading {model_name}...")
    S = Stack(stack_path, dimensions=DIMENSIONS)
    S.predictInset(model_paths[model_name], TILE_SIZE, INSET_SIZE)

    s = S.cleanSpines(DEND_THRESHOLD, DEND_DILATION)
    S.prediction[..., 1] = s

    dend_map = S.prediction[..., 0].astype(np.float32)
    spine_map = S.prediction[..., 1].astype(np.float32)
    del S

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _save_atomic(dend_fn, dend_map)
    _save_atomic(spine_fn, spine_map)
    params_fn.write_text(json.dumps(_prediction_params()))
    print(f"  {model_name}: inference done (cached)")
    return dend_map, spine_map


# ============================================================
# ROI building
# ============================================================
# The flood fill only depends on peak threshold and seed delta; min_size, max_size
# and min/max planes are pure post-filters (see ROI3D_Creator.create). So the flood
# fill runs once per (peak, delta) pair and all size/plane filters reuse its labels.
def flood_fill_labels(spine_map, peak_threshold, seed_delta):
    seeds = _get_sorted_seeds(spine_map, peak_threshold)
    labels, _ = connected_components_3d(
        spine_map, seeds, seed_delta, ROI_AREA_THRESHOLD, ROI_DISTANCE_TO_SEED,
        (DIMENSIONS['xy'], DIMENSIONS['z'])
    )
    return labels


def label_stats(labels):
    """Voxel count and number of covered z-planes per label (index 0 = background)."""
    n = int(labels.max())
    px = np.bincount(labels.ravel(), minlength=n + 1)
    planes = np.zeros(n + 1, dtype=np.int64)
    for z in range(labels.shape[0]):
        planes[np.unique(labels[z])] += 1
    return px, planes


def keep_mask(px, planes, min_um3, max_um3, min_p, max_p):
    vol = px * VOXEL_UM3
    keep = (vol >= min_um3) & (vol <= max_um3) & (planes >= min_p) & (planes <= max_p)
    keep[0] = False
    return keep


def relabel(labels, keep):
    """Drop labels not in `keep` and renumber the rest 1..K (like cleanLabels + reid)."""
    remap = np.zeros(len(keep), dtype=np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
    return remap[labels]


def gt_metrics(gt_xyz, roi_xyz):
    """Precision/recall/F1 with one-to-one matching: each annotated spine can match at
    most one ROI and vice versa, so split or merged spines are penalised."""
    gt_um, roi_um = xyz_to_um(gt_xyz), xyz_to_um(roi_xyz)
    tp = 0
    if len(gt_um) and len(roi_um):
        pairs = cKDTree(gt_um).sparse_distance_matrix(cKDTree(roi_um), GT_MATCH_UM, output_type="dict")
        if pairs:
            cost = np.full((len(gt_um), len(roi_um)), 1e6)
            for (i, j), d in pairs.items():
                cost[i, j] = d
            r, c = linear_sum_assignment(cost)
            tp = int((cost[r, c] < GT_MATCH_UM).sum())
    fp, fn = len(roi_um) - tp, len(gt_um) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


# ============================================================
# Config selection
# ============================================================
SIZE_COLS = ["min_size_um3", "max_size_um3", "min_planes", "max_planes"]
PARAM_COLS = ["peak_threshold", "seed_delta"] + SIZE_COLS


def add_instability(df):
    """Instability of each row's spine count: mean |log(n_neighbour / n)| over the 4
    settings one step away in the peak x seed-delta grid (same stack, model and size
    filter). 0 = the count does not change at all; 0.1 ≈ 10% change per step.
    NaN on the grid edge (fewer than 4 neighbours: an edge setting would only be
    compared in the directions that happen to be flat) and when the row or a neighbour
    found no spines.
    """
    pi = {p: i for i, p in enumerate(roi_peak_thresholds)}
    di = {d: i for i, d in enumerate(roi_seed_deltas)}
    inst = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["stack", "model"] + SIZE_COLS):
        counts = {(pi[p], di[d]): n for p, d, n in
                  zip(g["peak_threshold"], g["seed_delta"], g["n_spines"])}
        for idx, p, d, n in zip(g.index, g["peak_threshold"], g["seed_delta"], g["n_spines"]):
            i, j = pi[p], di[d]
            nb = [counts[k] for k in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if k in counts]
            if n > 0 and len(nb) == 4 and min(nb) > 0:
                inst[idx] = float(np.mean(np.abs(np.log(np.array(nb) / n))))
    df["instability"] = inst
    return df


def rank_configs(df):
    """Average each (model, params) over stacks and rank best-first.

    With ground truth: highest F1 (averaged over annotated stacks). Without: lowest
    instability (most stable spine count). Only settings with a valid instability on
    every stack are ranked, so one that fails on some stack cannot win on the others.
    "tied_best" marks settings within STABILITY_TIE_TOL of the best one with the same
    size filters.
    """
    agg = (df.groupby(["model"] + PARAM_COLS)
             .agg(n_spines=("n_spines", "mean"),
                  median_volume_um3=("median_volume_um3", "mean"),
                  instability=("instability", "mean"),
                  outlier_frac=("outlier_frac", "mean"),
                  f1=("f1", "mean"),
                  n_stacks_valid=("instability", "count"))
             .reset_index())
    if agg["f1"].notna().any():
        return agg[agg["f1"].notna()].sort_values("f1", ascending=False)
    n_stacks = df["stack"].nunique()
    agg = agg[(agg["n_stacks_valid"] == n_stacks) & (agg["n_spines"] > 0)].copy()
    group_min = agg.groupby(SIZE_COLS)["instability"].transform("min")
    agg["tied_best"] = agg["instability"] <= group_min + STABILITY_TIE_TOL
    return agg.sort_values(["instability", "n_spines"], ascending=[True, False])


def _same_size_filters(ranked, cfg):
    return ranked[ranked[SIZE_COLS].eq(pd.Series({k: cfg[k] for k in SIZE_COLS})).all(axis=1)]


def _check_chosen_config():
    grid = {"model": models, "peak_threshold": roi_peak_thresholds, "seed_delta": roi_seed_deltas,
            "min_size_um3": min_sizes_um3, "max_size_um3": max_sizes_um3,
            "min_planes": min_planes, "max_planes": max_planes}
    for k, allowed in grid.items():
        if CHOSEN_CONFIG[k] not in allowed:
            raise ValueError(f"CHOSEN_CONFIG[{k!r}] = {CHOSEN_CONFIG[k]} is not in the sweep grid {allowed}")
    for k, grid_vals in [("peak_threshold", roi_peak_thresholds), ("seed_delta", roi_seed_deltas)]:
        if CHOSEN_CONFIG[k] in (grid_vals[0], grid_vals[-1]):
            print(f"  Note: CHOSEN_CONFIG[{k!r}] is on the grid edge, so it gets no stability score")


# ============================================================
# Per-spine measurements
# ============================================================
def _ellipsoid_offsets(r_xy_um, r_z_um):
    """(N, 3) integer (z, y, x) offsets of the voxels inside an ellipsoid."""
    rz = int(r_z_um // DIMENSIONS['z'])
    ry = int(r_xy_um // DIMENSIONS['xy'])
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -ry:ry + 1]
    inside = ((zz * DIMENSIONS['z'] / r_z_um) ** 2
              + ((yy ** 2 + xx ** 2) * DIMENSIONS['xy'] ** 2) / r_xy_um ** 2) <= 1
    return np.stack([zz[inside], yy[inside], xx[inside]], axis=1)


def _plane_background(raw, dend):
    """Per-plane background: 25th percentile of voxels far from any dendrite."""
    bg = np.full(raw.shape[0], np.nan)
    for z in range(raw.shape[0]):
        far = dend[z] < 0.1
        if far.any():
            bg[z] = np.percentile(raw[z][far], 25)
    return np.where(np.isnan(bg), np.nanmedian(bg), bg)


def spine_table(stack_path, cfg, selection, return_labels=False):
    """One row per spine for one stack under one setting. spine_id = the spine's value in
    the label image (returned too with return_labels=True).

    Size: volume_um3 (all ROI voxels; inflated in z by axial blur), max_area_um2 (largest
    cross-section in a single plane; much less affected by z blur) and equiv_diameter_um
    (diameter of a disc with that area), z_min/z_max (planes spanned).

    brightness_shaft_um3 (primary): background-subtracted fluorescence in a fixed
    ellipsoid around the spine's brightest voxel (shaft voxels excluded), divided by
    the median fluorescence of the nearest dendrite's core, times the voxel volume.
    = how many µm³ of that dendrite's shaft would be as bright as the spine head. It
    cancels laser power / expression and does not depend on the seed delta.
    brightness_roi_shaft_um3: same but summed over the whole flood-fill ROI, so it
    grows with seed delta — use it only to compare runs with identical settings.
    qc_flags lists reasons to distrust a row; brightness is NaN when the shaft
    reference is unusable (too few core voxels, or too dim).
    """
    dend, spine = _get_prediction(cfg["model"], stack_path)
    raw_in = load_raw(stack_path)
    sat_level = SATURATION_FRAC * (np.iinfo(raw_in.dtype).max
                                   if np.issubdtype(raw_in.dtype, np.integer) else raw_in.max())
    raw = raw_in.astype(np.float32)
    del raw_in

    lab = flood_fill_labels(spine, cfg["peak_threshold"], cfg["seed_delta"])
    px, planes = label_stats(lab)
    keep = keep_mask(px, planes, *(cfg[k] for k in SIZE_COLS))
    lab = relabel(lab, keep)

    bg = _plane_background(raw, dend)
    shaft = (dend > DEND_THRESHOLD) & (lab == 0)
    # Separate dendrites, so each spine is normalised to the dendrite it sits on
    dend_cc = cc3d.connected_components(dend > DEND_THRESHOLD, connectivity=26)
    head = _ellipsoid_offsets(SPINE_HEAD_RADIUS_XY_UM, SPINE_HEAD_RADIUS_Z_UM)
    rz = int(round(SHAFT_WINDOW_Z_UM / DIMENSIONS['z']))
    ry = int(round(SHAFT_WINDOW_XY_UM / DIMENSIONS['xy']))
    scale = np.array([DIMENSIONS['z'], DIMENSIONS['xy'], DIMENSIONS['xy']])
    shape = np.array(lab.shape)

    rows = []
    for i, o in enumerate(ndimage.find_objects(lab), start=1):
        if o is None:
            continue
        start = np.array([s.start for s in o])
        m = lab[o] == i
        sub = raw[o]
        zc, yc, xc = np.array(ndimage.center_of_mass(m)) + start
        pk = np.array(np.unravel_index(np.where(m, sub, -np.inf).argmax(), m.shape)) + start

        # Whole flood-fill ROI
        roi_vals = sub[m] - bg[np.nonzero(m)[0] + start[0]]

        # Fixed spine-head region around the brightest voxel, minus shaft voxels
        pts = pk + head
        pts = pts[((pts >= 0) & (pts < shape)).all(axis=1)]
        pts = pts[~shaft[pts[:, 0], pts[:, 1], pts[:, 2]]]
        head_raw = raw[pts[:, 0], pts[:, 1], pts[:, 2]]
        head_int = float((head_raw - bg[pts[:, 0]]).sum())

        # Shaft reference: core of the dendrite component nearest to the spine
        win = tuple(slice(max(c - r, 0), c + r + 1) for c, r in zip(pk, (rz, ry, ry)))
        w0 = np.array([s.start for s in win])
        cand = np.argwhere(shaft[win])
        shaft_int, n_core, shaft_sat = np.nan, 0, False
        if len(cand):
            nearest = cand[np.argmin((((cand + w0 - pk) * scale) ** 2).sum(1))]
            comp = dend_cc[win][tuple(nearest)]
            core = shaft[win] & (dend_cc[win] == comp) & (dend[win] > SHAFT_CORE_THRESHOLD)
            n_core = int(core.sum())
            if n_core:
                core_raw = raw[win][core]
                shaft_sat = bool(np.mean(core_raw >= sat_level) > 0.1)
                if n_core >= SHAFT_MIN_VOXELS:
                    shaft_int = float(np.median(core_raw - bg[np.nonzero(core)[0] + w0[0]]))

        plane_px = m.sum(axis=(1, 2))
        max_area = float(plane_px.max() * DIMENSIONS['xy'] ** 2)
        rows.append({
            "stack": Path(stack_path).stem, "selection": selection, "model": cfg["model"],
            "spine_id": i,
            "z_px": zc, "y_px": yc, "x_px": xc,
            "z_um": zc * DIMENSIONS['z'], "y_um": yc * DIMENSIONS['xy'], "x_um": xc * DIMENSIONS['xy'],
            "volume_um3": float(m.sum() * VOXEL_UM3),
            "n_planes": int((plane_px > 0).sum()),
            "z_min": int(start[0] + np.flatnonzero(plane_px)[0]),
            "z_max": int(start[0] + np.flatnonzero(plane_px)[-1]),
            "max_area_um2": max_area,
            "equiv_diameter_um": float(2 * np.sqrt(max_area / np.pi)),
            "max_spine_prob": float(spine[o][m].max()),
            "head_mean_intensity": head_int / max(len(pts), 1),
            "head_integrated_intensity": head_int,
            "roi_integrated_intensity": float(roi_vals.sum()),
            "shaft_intensity": shaft_int,
            "n_shaft_core_voxels": n_core,
            "_saturated_head": bool((head_raw >= sat_level).any()),
            "_saturated_shaft": shaft_sat,
        })
    del dend, spine, raw, shaft, dend_cc
    if not return_labels:
        del lab
    gc.collect()

    t = _finish_spine_table(pd.DataFrame(rows))
    return (t, lab) if return_labels else t


def _finish_spine_table(t):
    if t.empty:
        return t
    median_shaft = np.nanmedian(t["shaft_intensity"]) if t["shaft_intensity"].notna().any() else np.nan
    few = t["n_shaft_core_voxels"] < SHAFT_MIN_VOXELS
    dim = ~few & (t["shaft_intensity"] < SHAFT_MIN_REL * median_shaft)
    bright = t["head_mean_intensity"] > 3 * t["shaft_intensity"]
    flags = {"few_shaft_voxels": few, "dim_shaft": dim, "saturated_head": t["_saturated_head"],
             "saturated_shaft": t["_saturated_shaft"], "head_gt_3x_shaft": bright}
    t["qc_flags"] = [";".join(k for k, v in flags.items() if v.iloc[r]) for r in range(len(t))]
    usable = ~(few | dim)
    t["brightness_shaft_um3"] = np.where(usable, t["head_integrated_intensity"] / t["shaft_intensity"] * VOXEL_UM3, np.nan)
    t["brightness_roi_shaft_um3"] = np.where(usable, t["roi_integrated_intensity"] / t["shaft_intensity"] * VOXEL_UM3, np.nan)
    return t.drop(columns=["_saturated_head", "_saturated_shaft"])


# ============================================================
# Visualization
# ============================================================
def _roi_overlay(roi_slice):
    overlay = np.zeros((*roi_slice.shape, 4))
    for label in np.unique(roi_slice):
        if label == 0:
            continue
        color = plt.cm.tab20((label - 1) % 20)
        overlay[roi_slice == label] = (*color[:3], 0.6)
    return overlay


def _show_segmentation(ax, raw, dend, roi_slice, title):
    ax.imshow(raw, cmap="gray")
    ax.contour(dend > DEND_THRESHOLD, levels=[0.5], colors="cyan", linewidths=0.6)
    ax.imshow(_roi_overlay(roi_slice))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _frame(ax):
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("red"); s.set_linewidth(3)


def visualize(viz, df, cfg, heading, output_path):
    """
    viz : dict collected during the sweep for the randomly chosen stack/plane:
          {"stack", "z", "raw", "dend": {model: 2D}, "spine": {model: 2D},
           "labels": {(model, peak, delta): (labels_2D, px, planes)}}
    cfg : dict/row with model + PARAM_COLS — the setting to show
    heading : e.g. "Chosen (a)" or "Most stable (b)"
    """
    z, raw = viz["z"], viz["raw"]
    bm = cfg["model"]
    size_args = tuple(cfg[k] for k in SIZE_COLS)
    size_txt = (f"min={cfg['min_size_um3']} max={cfg['max_size_um3']} µm³ "
                f"planes={cfg['min_planes']}-{cfg['max_planes']}")

    # Zoom: spines are only a few pixels wide, so all panels except the first show a
    # VIZ_CROP x VIZ_CROP window around the densest spine signal in this plane
    c = min(VIZ_CROP, *raw.shape)
    dens = ndimage.uniform_filter((viz["spine"][bm] >= cfg["peak_threshold"]).astype(float), c)
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    y0 = int(np.clip(cy - c // 2, 0, raw.shape[0] - c))
    x0 = int(np.clip(cx - c // 2, 0, raw.shape[1] - c))
    crop = (slice(y0, y0 + c), slice(x0, x0 + c))
    lo, hi = np.percentile(raw, [1, 99.7])
    raw_n = np.clip((raw.astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)
    raw_c = raw_n[crop]
    dend_c = {m: viz["dend"][m][crop] for m in viz["dend"]}

    def roi_slice(model, peak, delta, *size):
        lab, px, planes = viz["labels"][(model, peak, delta)]
        keep = keep_mask(px, planes, *size)
        return relabel(lab[crop], keep), int(keep.sum())

    def rows_for(model, peak, delta):
        same = df[SIZE_COLS].eq(pd.Series({k: cfg[k] for k in SIZE_COLS})).all(axis=1)
        return df[same & (df["model"] == model) & (df["peak_threshold"] == peak)
                  & (df["seed_delta"] == delta)]

    n_rows = len(roi_peak_thresholds)
    fig = plt.figure(figsize=(20, 10 + 4.2 * n_rows))
    top, mid, bottom = fig.subfigures(3, 1, height_ratios=[1, 1, 0.9 * n_rows])

    # ── Row 1: selected model, step by step ──
    axes = top.subplots(1, 4)
    best_roi, best_n = roi_slice(bm, cfg["peak_threshold"], cfg["seed_delta"], *size_args)

    axes[0].imshow(raw_n, cmap="gray")
    axes[0].add_patch(plt.Rectangle((x0, y0), c, c, fill=False, edgecolor="yellow", linewidth=1.5))
    axes[0].set_title(f"Raw (full plane, yellow = zoom {c * DIMENSIONS['xy']:.0f} µm)", fontsize=9)

    axes[1].imshow(raw_c, cmap="gray")
    axes[1].imshow(np.ma.masked_less_equal(dend_c[bm], DEND_THRESHOLD),
                   cmap="cool", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"Dendrite (p > {DEND_THRESHOLD})", fontsize=9)

    im = axes[2].imshow(viz["spine"][bm][crop], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Spine probability", fontsize=9)
    top.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

    _show_segmentation(axes[3], raw_c, dend_c[bm], best_roi,
                       f"Spine ROIs + dendrite outline\nROIs in view: {len(np.unique(best_roi)) - 1}")
    for ax in axes[:3]:
        ax.axis("off")
    top.suptitle(
        f"{heading}: {bm} | peak={cfg['peak_threshold']} delta={cfg['seed_delta']} "
        f"{size_txt} | {best_n} ROIs in stack",
        fontsize=11, fontweight="bold"
    )

    # ── Row 2: model comparison at the same parameters ──
    axes = mid.subplots(1, len(models) + 1)
    for ax, m in zip(axes, models):
        r, n = roi_slice(m, cfg["peak_threshold"], cfg["seed_delta"], *size_args)
        _show_segmentation(ax, raw_c, dend_c[m], r, f"{m}\n{n} ROIs in stack")
        if m == bm:
            _frame(ax)

    model_stats = (df[df[PARAM_COLS].eq(pd.Series({k: cfg[k] for k in PARAM_COLS})).all(axis=1)]
                   .groupby("model")[["n_spines", "instability"]].mean()
                   .reindex(models))
    ax = axes[-1]
    ax.bar(models, model_stats["n_spines"], color=["red" if m == bm else "gray" for m in models])
    ax.set_ylabel("mean ROIs per stack (all stacks)")
    for i, m in enumerate(models):
        ax.text(i, model_stats.loc[m, "n_spines"], f"instab={model_stats.loc[m, 'instability']:.2f}",
                ha="center", va="bottom", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    mid.suptitle("Model comparison (same parameters)", fontsize=11)

    # ── Rows 3+: peak threshold × seed delta sweep for the selected model ──
    axes = bottom.subplots(len(roi_peak_thresholds), len(roi_seed_deltas), squeeze=False)
    for i, peak in enumerate(roi_peak_thresholds):
        for j, delta in enumerate(roi_seed_deltas):
            ax = axes[i, j]
            r, n = roi_slice(bm, peak, delta, *size_args)
            row = rows_for(bm, peak, delta)
            _show_segmentation(ax, raw_c, dend_c[bm], r,
                               f"peak={peak} delta={delta}\n{n} ROIs | median "
                               f"{row['median_volume_um3'].mean():.2f} µm³ | "
                               f"instability={row['instability'].mean():.2f} (all stacks)")
            if peak == cfg["peak_threshold"] and delta == cfg["seed_delta"]:
                _frame(ax)
    bottom.suptitle(f"Peak × seed-delta sweep — {bm}, {size_txt} (red = {heading.lower()})",
                    fontsize=11)

    fig.suptitle(f"{viz['stack']} | z = {z}/{viz['n_planes'] - 1}",
                 fontsize=13, y=1.02)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved → {output_path}")


# ============================================================
# Benchmark loop
# ============================================================
def run_benchmark():
    if not stack_paths:
        raise FileNotFoundError("No .tif files found. Check stack_paths glob.")
    _check_chosen_config()

    rng = random.Random(RANDOM_SEED)
    viz_stack = rng.choice(stack_paths)
    viz = {"stack": Path(viz_stack).stem, "dend": {}, "spine": {}, "labels": {}}

    results = []
    n_per_stack = (len(models) * len(roi_peak_thresholds) * len(roi_seed_deltas)
                   * len(min_sizes_um3) * len(max_sizes_um3) * len(min_planes) * len(max_planes))
    print(f"Stacks: {len(stack_paths)}")
    print(f"Combos per stack: {n_per_stack}")
    print(f"Total runs: {len(stack_paths) * n_per_stack}")
    print(f"Visualization stack: {viz['stack']}\n")

    for stack_path in stack_paths:
        stack_name = Path(stack_path).stem
        stack_gt = gt_spines.get(stack_name)
        is_viz = stack_path == viz_stack
        print(f"\n{'='*60}")
        print(f"  Stack: {stack_name}")
        print(f"{'='*60}")

        for model in models:
            gc.collect()
            _mem_guard(threshold_gb=24.0)

            dend_map, spine_map = _get_prediction(model, stack_path)

            if is_viz:
                if "z" not in viz:
                    # Random plane, but only among the 25% of planes with the most spine signal
                    signal = (spine_map >= min(roi_peak_thresholds)).sum((1, 2))
                    candidates = np.flatnonzero(signal >= np.percentile(signal, 75))
                    viz["z"] = int(rng.choice(list(candidates)))
                    viz["n_planes"] = dend_map.shape[0]
                    # .copy() so the full raw stack is not kept alive by a view
                    viz["raw"] = load_raw(stack_path)[viz["z"]].copy()
                viz["dend"][model] = dend_map[viz["z"]].copy()
                viz["spine"][model] = spine_map[viz["z"]].copy()

            for peak_th, delta in product(roi_peak_thresholds, roi_seed_deltas):
                labels = flood_fill_labels(spine_map, peak_th, delta)
                px, planes = label_stats(labels)

                if stack_gt:
                    n = len(px) - 1
                    # (z, y, x) centroid per raw label -> (x, y, z); empty labels give NaN
                    zyx = np.array(ndimage.center_of_mass(labels, labels, range(1, n + 1))).reshape(-1, 3)
                    raw_centroids = zyx[:, ::-1]
                    in_region = ~np.isnan(zyx).any(axis=1)
                    if stack_gt.get("region") is not None:
                        z0, z1, y0, y1, x0, x1 = stack_gt["region"]
                        in_region &= ((zyx[:, 0] >= z0) & (zyx[:, 0] <= z1) & (zyx[:, 1] >= y0)
                                      & (zyx[:, 1] <= y1) & (zyx[:, 2] >= x0) & (zyx[:, 2] <= x1))
                if is_viz:
                    viz["labels"][(model, peak_th, delta)] = (labels[viz["z"]].copy(), px, planes)

                for min_s, max_s, min_p, max_p in product(min_sizes_um3, max_sizes_um3,
                                                          min_planes, max_planes):
                    keep = keep_mask(px, planes, min_s, max_s, min_p, max_p)
                    vols = px[keep] * VOXEL_UM3
                    n_spines = int(keep.sum())
                    if n_spines > 0:
                        median_volume = float(np.median(vols))
                        outlier_frac = float(np.mean(vols > 3 * median_volume))
                    else:
                        median_volume = outlier_frac = 0.0

                    if stack_gt:
                        roi_xyz = raw_centroids[keep[1:] & in_region]
                        precision, recall, f1 = gt_metrics(stack_gt["points_xyz"], roi_xyz)
                    else:
                        precision = recall = f1 = None

                    results.append({
                        "stack": stack_name,
                        "model": model,
                        "peak_threshold": peak_th,
                        "seed_delta": delta,
                        "min_size_um3": min_s, "max_size_um3": max_s,
                        "min_planes": min_p, "max_planes": max_p,
                        "n_spines": n_spines,
                        "median_volume_um3": median_volume,
                        "outlier_frac": outlier_frac,
                        "precision": precision, "recall": recall, "f1": f1
                    })
                del labels
            print(f"  {model}: sweep done")

            del dend_map, spine_map
            gc.collect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df["f1"] = pd.to_numeric(df["f1"])
    df = add_instability(df)
    df.to_csv(OUT_DIR / "deepd3_benchmark_results.csv", index=False)

    # ── Summary ──
    ranked = rank_configs(df)
    if ranked.empty:
        raise RuntimeError("No configuration found any spines — check predictions / thresholds.")
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.to_csv(OUT_DIR / "deepd3_sweep_ranked.csv", index=False)

    rank_name = "best F1" if ranked["f1"].notna().any() else "most stable"
    # (b): with ground truth the top F1 overall; otherwise the most stable model/peak/delta
    # among settings with the chosen size filters
    pool = ranked if rank_name == "best F1" else _same_size_filters(ranked, CHOSEN_CONFIG)
    if pool.empty:
        raise RuntimeError("No ranked setting shares CHOSEN_CONFIG's size filters.")
    # Same rule as (b), per model; first row = (b)
    best_per_model = pool.groupby("model", sort=False).head(1)
    best_per_model.to_csv(OUT_DIR / "deepd3_best_per_model.csv", index=False)
    stable_cfg = {k: (v.item() if hasattr(v, "item") else v)
                  for k, v in pool.iloc[0][["model"] + PARAM_COLS].items()}
    selections = {
        "chosen": (CHOSEN_CONFIG, "Chosen (a)"),
        "stable": (stable_cfg, f"{rank_name.capitalize()} (b)"),
    }

    print(f"\n{'='*60}")
    print(f"  Best config per model ({rank_name}):")
    print(f"{'='*60}")
    print(best_per_model.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  Top 10 configs ({rank_name}, averaged over stacks):")
    print(f"{'='*60}")
    print(ranked.head(10).to_string(index=False))

    chosen_row = ranked[ranked[["model"] + PARAM_COLS].eq(
        pd.Series({k: CHOSEN_CONFIG[k] for k in ["model"] + PARAM_COLS})).all(axis=1)]
    print(f"\n  Chosen (a) config ranks #{int(chosen_row['rank'].iloc[0]) if len(chosen_row) else 'n/a'} "
          f"of {len(ranked)} under {rank_name}")
    if "tied_best" in ranked:
        same = _same_size_filters(ranked, CHOSEN_CONFIG)
        tied = same[same["tied_best"]]
        print(f"  (b) = {selections['stable'][0]}")
        print(f"  With the chosen size filters, {len(tied)} model/peak/delta settings are within "
              f"{STABILITY_TIE_TOL} of the most stable: "
              + ", ".join(f"{r.model} {r.peak_threshold}/{r.seed_delta} ({r.instability:.3f})"
                          for r in tied.itertuples())
              + (" — chosen (a) is one of them" if len(chosen_row) and bool(chosen_row['tied_best'].iloc[0]) else ""))

    print(f"\n{'='*60}")
    print("  Generating visualizations...")
    print(f"{'='*60}")
    for key, (cfg, heading) in selections.items():
        visualize(viz, df, cfg, heading, OUT_DIR / f"deepd3_viz_{key}.png")

    print(f"\n{'='*60}")
    print("  Per-spine measurements...")
    print(f"{'='*60}")
    summaries = []
    for key, (cfg, heading) in selections.items():
        tables = []
        for stack_path in stack_paths:
            gc.collect()
            _mem_guard(threshold_gb=24.0)
            t = spine_table(stack_path, cfg, key)
            tables.append(t)
            summaries.append({
                "stack": Path(stack_path).stem, "selection": key, "model": cfg["model"],
                **{k: cfg[k] for k in PARAM_COLS},
                "n_spines": len(t),
                "median_volume_um3": t["volume_um3"].median() if len(t) else np.nan,
                "median_brightness_shaft_um3": t["brightness_shaft_um3"].median() if len(t) else np.nan,
                "frac_no_shaft_reference": t["brightness_shaft_um3"].isna().mean() if len(t) else np.nan,
                "frac_saturated": (t["qc_flags"].str.contains("saturated").mean() if len(t) else np.nan),
            })
            print(f"  {key}: {Path(stack_path).stem}: {len(t)} spines")
        pd.concat(tables, ignore_index=True).to_csv(OUT_DIR / f"deepd3_spines_{key}.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "deepd3_spine_summary.csv", index=False)
    print(summary.to_string(index=False))

    return df


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    df = run_benchmark()
