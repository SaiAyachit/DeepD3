import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# --- Configuration ---
models = [ "DeepD3_32F", "DeepD3_16F", "DeepD3_8F"]
thresholds = [0.05, 0.1, 0.2, 0.3]
min_sizes = [5, 10, 20]
max_sizes = [5000, 10000, 20000]
min_planes = [1, 2, 3]

# Your data
stack_path = "path/to/your/stack.tif"
# Ground truth (optional): list of (x, y, z) for manually annotated spines
gt_spines = [] #[(x1, y1, z1), (x2, y2, z2), ...]  # empty if no ground truth available

# --- Loop ---
results = []

for model in models:
    for thr in thresholds:
        for min_s, max_s, min_p in product(min_sizes, max_sizes, min_planes):
            # Run DeepD3 inference (CLI or API)
            # → get ROI centroids: list of (x, y, z)
            rois = run_deepd3(stack_path, model=model,
                             threshold=thr, min_size=min_s,
                             max_size=max_s, min_planes=min_p)

            # --- Tier 1 metrics (always computed) ---
            n_spines = len(rois)
            roi_sizes = [r.voxel_count for r in rois]
            median_size = np.median(roi_sizes)
            outlier_frac = np.mean([s > 3*median_size for s in roi_sizes])

            # --- Tier 2 metrics (if ground truth available) ---
            if gt_spines:
                tp = fp = fn = 0
                for gt in gt_spines:
                    if any(dist3d(gt, roi.xyz) < 2.0 for roi in rois):  # 2 µm radius
                        tp += 1
                    else:
                        fn += 1
                for roi in rois:
                    if not any(dist3d(roi.xyz, gt) < 2.0 for gt in gt_spines):
                        fp += 1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            else:
                precision = recall = f1 = None

            results.append({
                "model": model, "threshold": thr,
                "min_size": min_s, "max_size": max_s, "min_planes": min_p,
                "n_spines": n_spines, "outlier_frac": outlier_frac,
                "precision": precision, "recall": recall, "f1": f1
            })

df = pd.DataFrame(results)
df.to_csv("deepd3_benchmark_results.csv")
print(df.sort_values("f1", ascending=False).head(20))