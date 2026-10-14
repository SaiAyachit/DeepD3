"""
parameter_test.py — DeepD3 parameter sweep using official deepd3 package.
"""
#python deepd3/inference/parameter_test.py   

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from glob import glob
from scipy import ndimage
from tqdm import tqdm
import flammkuchen as fl
from deepd3.core.analysis import Stack, ROI3D_Creator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF info logs

import tensorflow as tf
_orig_predict = tf.keras.Model.predict
def _silent_predict(self, *args, **kwargs):
    kwargs['verbose'] = 0
    return _orig_predict(self, *args, **kwargs)
tf.keras.Model.predict = _silent_predict   
# ============================================================
# Configuration
# ============================================================
models = ["DeepD3_32F", "DeepD3_16F", "DeepD3_8F"]
MODEL_DIR = "deepd3/inference/pretrained_models"
model_paths = {m: f"{MODEL_DIR}/{m}.h5" for m in models}   

# All your data
stack_paths = sorted(glob("deepd3/inference/data/processed/*.tif"))
# Or explicit:
# stack_paths = ["data/stack1.tif", "data/stack2.tif", "data/stack3.tif"]

# Sweep grid
roi_area_thresholds = [0.15, 0.25, 0.35, 0.5]
roi_peak_thresholds = [0.6, 0.8, 1.0]
min_sizes = [5, 10, 20]
max_sizes = [500, 1000, 2000]
min_planes = [1, 2, 3]

# Fixed parameters (DeepD3 defaults)
ROI_METHOD = "floodfill"
ROI_SEED_DELTA = 0.1
ROI_DISTANCE_TO_SEED = 10      # µm
DEND_THRESHOLD = 0.7
DEND_MIN_SIZE = 100
DEND_DILATION = 11
TILE_SIZE = 128
INSET_SIZE = 96

# Your data
gt_spines = {}  # {stack_name: [(x, y, z), ...]} — empty if no ground truth
# Example:
# gt_spines = {
#     "stack1": [(100, 200, 50), (150, 250, 52), ...],
#     "stack2": [(80, 120, 30), ...],
# }

# Voxel size in µm — SET THIS TO YOUR DATA
DIMENSIONS = {'xy': 0.145, 'z': 1} # XY = 0.145 µm pixel size , Z= 1 µm slice thickness


# ============================================================
# Helpers
# ============================================================
def dist3d_um(p1, p2, dimensions=DIMENSIONS):
    d = np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)
    d_xy = np.sqrt(d[0]**2 + d[1]**2) * dimensions['xy']
    d_z = d[2] * dimensions['z']
    return float(np.sqrt(d_xy**2 + d_z**2))


# ============================================================
# Inference (cached per model per stack)
# ============================================================
_prediction_cache = {}


def _get_prediction(model_name, stack_path):
    key = (model_name, stack_path)
    if key not in _prediction_cache:
        S = Stack(stack_path, dimensions=DIMENSIONS)
        S.predictInset(model_paths[model_name], TILE_SIZE, INSET_SIZE)

        s = S.cleanSpines(DEND_THRESHOLD, DEND_DILATION)
        S.prediction[..., 1] = s

        dend_map = S.prediction[..., 0]
        spine_map = S.prediction[..., 1]
        _prediction_cache[key] = (dend_map, spine_map)
        print(f"    {model_name} on {Path(stack_path).stem}: inference done")

    return _prediction_cache[key]


# ============================================================
# ROI building
# ============================================================
def build_rois(dend_map, spine_map, area_threshold, peak_threshold,
               min_size, max_size, min_planes):
    r = ROI3D_Creator(
        dend_map, spine_map,
        mode=ROI_METHOD,
        areaThreshold=area_threshold,
        peakThreshold=peak_threshold,
        seedDelta=ROI_SEED_DELTA,
        distanceToSeed=ROI_DISTANCE_TO_SEED,
        dimensions=DIMENSIONS
    )
    n_rois = r.create(min_size, max_size, min_planes)
    return r.rois, r.roi_map, n_rois


# ============================================================
# Benchmark loop
# ============================================================
def run_benchmark():
    results = []

    n_per_stack = (len(models) * len(roi_area_thresholds) * len(roi_peak_thresholds)
                   * len(min_sizes) * len(max_sizes) * len(min_planes))
    print(f"Stacks: {len(stack_paths)}")
    print(f"Combos per stack: {n_per_stack}")
    print(f"Total runs: {len(stack_paths) * n_per_stack}\n")

    for stack_path in stack_paths:
        stack_name = Path(stack_path).stem
        print(f"\n{'='*60}")
        print(f"  Stack: {stack_name}")
        print(f"{'='*60}")

        # Clear cache for new stack
        _prediction_cache.clear()

        for model in models:
            dend_map, spine_map = _get_prediction(model, stack_path)

            for area_th, peak_th in product(roi_area_thresholds, roi_peak_thresholds):
                for min_s, max_s, min_p in tqdm(
                    product(min_sizes, max_sizes, min_planes),
                    desc=f"  {model} area={area_th} peak={peak_th}",
                    leave=False
                ):
                    rois, roi_map, n_rois = build_rois(
                        dend_map, spine_map,
                        area_th, peak_th,
                        min_s, max_s, min_p
                    )

                    # --- Tier 1 ---
                    n_spines = n_rois
                    if n_spines > 0:
                        labels = roi_map[roi_map > 0]
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        median_size = float(np.median(counts))
                        outlier_frac = float(np.mean(counts > 3 * median_size))
                    else:
                        median_size = 0.0
                        outlier_frac = 0.0

                    # --- Tier 2 ---
                    stack_gt = gt_spines.get(stack_name, [])
                    if stack_gt:
                        centroids = ndimage.center_of_mass(
                            roi_map, roi_map, range(1, n_spines + 1)
                        )
                        roi_xyz = [(int(c[2]), int(c[1]), int(c[0])) for c in centroids]

                        tp = fp = fn = 0
                        for gt in stack_gt:
                            if any(dist3d_um(gt, xyz) < 2.0 for xyz in roi_xyz):
                                tp += 1
                            else:
                                fn += 1
                        for xyz in roi_xyz:
                            if not any(dist3d_um(xyz, gt) < 2.0 for gt in stack_gt):
                                fp += 1
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    else:
                        precision = recall = f1 = None

                    results.append({
                        "stack": stack_name,
                        "model": model,
                        "area_threshold": area_th,
                        "peak_threshold": peak_th,
                        "min_size": min_s, "max_size": max_s, "min_planes": min_p,
                        "n_spines": n_spines,
                        "median_size": median_size,
                        "outlier_frac": outlier_frac,
                        "precision": precision, "recall": recall, "f1": f1
                    })

    df = pd.DataFrame(results)
    df.to_csv("deepd3_benchmark_results.csv", index=False)

    # Summary per stack
    print(f"\n{'='*60}")
    print("  Summary (per stack, best config by outlier_frac):")
    print(f"{'='*60}")
    for s in df["stack"].unique():
        sub = df[df["stack"] == s]
        if gt_spines.get(s, []):
            best = sub.sort_values("f1", ascending=False).iloc[0]
            print(f"\n  {s}: F1={best['f1']:.3f} | "
                  f"model={best['model']} area={best['area_threshold']} "
                  f"peak={best['peak_threshold']} min_s={best['min_size']} "
                  f"max_s={best['max_size']} min_p={best['min_planes']} "
                  f"n={best['n_spines']}")
        else:
            best = sub.sort_values(["outlier_frac", "n_spines"],
                                   ascending=[True, False]).iloc[0]
            print(f"\n  {s}: outlier_frac={best['outlier_frac']:.3f} | "
                  f"model={best['model']} area={best['area_threshold']} "
                  f"peak={best['peak_threshold']} min_s={best['min_size']} "
                  f"max_s={best['max_size']} min_p={best['min_planes']} "
                  f"n={best['n_spines']}")

    return df


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    df = run_benchmark()   