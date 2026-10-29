"""
inspect_plane.py — look at one plane of one stack with the settings found by parameter_test.py.

Saves two figures (in parameter_test.OUT_DIR):
  plane_<stack>_z<z>_<selection>_models.png : every model, each with its own best settings
  plane_<stack>_z<z>_<selection>_sweep.png  : the best model over the whole peak x seed-delta grid
With --spines it also writes the per-spine table (position, volume, brightness) for the whole
stack with the best setting.

Without --z a random plane between z = 20 and 30 is used (Z_RANGE below).

--selection stable (default): settings from deepd3_best_per_model.csv (run parameter_test.py first)
--selection chosen          : CHOSEN_CONFIG from parameter_test.py, applied to every model

Examples:
  python deepd3/inference/inspect_plane.py --stack m34_960_d3
  python deepd3/inference/inspect_plane.py --stack m34_960_d3 --z 25
  python deepd3/inference/inspect_plane.py --stack 2 --crop 300 500 256
  python deepd3/inference/inspect_plane.py --selection chosen --full --spines
"""
#python deepd3/inference/inspect_plane.py --stack m34_960_d1

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parameter_test as pt

# Without --z, a random plane is picked from this range (inclusive)
Z_RANGE = (20, 30)


def resolve_stack(arg, rng):
    if arg is None:
        return rng.choice(pt.stack_paths)
    if arg.isdigit():
        return pt.stack_paths[int(arg)]
    hits = [p for p in pt.stack_paths if arg in Path(p).stem]
    if len(hits) != 1:
        names = "\n  ".join(f"{i}: {Path(p).stem}" for i, p in enumerate(pt.stack_paths))
        raise SystemExit(f"--stack {arg!r} matches {len(hits)} stacks; use a unique part of the name "
                         f"or an index:\n  {names}")
    return hits[0]


def load_settings(selection):
    """{model: cfg} with each model's settings, and the overall best cfg."""
    if selection == "chosen":
        per_model = {m: dict(pt.CHOSEN_CONFIG, model=m) for m in pt.models}
        return per_model, dict(pt.CHOSEN_CONFIG)
    fn = pt.OUT_DIR / "deepd3_best_per_model.csv"
    if not fn.exists():
        raise SystemExit(f"{fn} not found — run parameter_test.py first, or use --selection chosen")
    rows = pd.read_csv(fn)
    per_model = {}
    for _, r in rows.iterrows():
        per_model[r["model"]] = {k: (r[k].item() if hasattr(r[k], "item") else r[k])
                                 for k in ["model"] + pt.PARAM_COLS}
    return per_model, per_model[rows.iloc[0]["model"]]


def load_prediction(model, stack_path):
    """Dendrite and spine maps, memory-mapped when cached (fast for a few planes)."""
    stem = f"{Path(stack_path).stem}__{model}"
    dend_fn = pt.CACHE_DIR / f"{stem}__dend.npy"
    spine_fn = pt.CACHE_DIR / f"{stem}__spine.npy"
    params_fn = pt.CACHE_DIR / f"{stem}__params.json"
    cached = (dend_fn.exists() and spine_fn.exists() and params_fn.exists()
              and json.loads(params_fn.read_text()) == pt._prediction_params())
    if not cached:
        pt._get_prediction(model, stack_path)  # runs inference and writes the cache
    return np.load(dend_fn, mmap_mode="r"), np.load(spine_fn, mmap_mode="r")


def z_window(z, n_planes):
    # An ROI touching plane z has its seed within ROI_DISTANCE_TO_SEED of z and extends at most
    # that far again, so a slab of ±(2 x distance + margin) contains every such ROI completely.
    half = math.ceil(2 * pt.ROI_DISTANCE_TO_SEED / pt.DIMENSIONS['z']) + 5
    return max(z - half, 0), min(z + half + 1, n_planes)


def segment_plane(spine_slab, z_local, cfg):
    """ROI labels (renumbered 1..K) on one plane for one setting."""
    labels = pt.flood_fill_labels(spine_slab, cfg["peak_threshold"], cfg["seed_delta"])
    px, planes = pt.label_stats(labels)
    keep = pt.keep_mask(px, planes, *(cfg[k] for k in pt.SIZE_COLS))
    return pt.relabel(labels[z_local], keep)


def stack_row(bench, stack, cfg):
    """This stack's row in the sweep results for a setting, if available."""
    if bench is None:
        return None
    sel = (bench["stack"] == stack) & (bench["model"] == cfg["model"])
    for k in pt.PARAM_COLS:
        sel &= np.isclose(bench[k], cfg[k])
    rows = bench[sel]
    return rows.iloc[0] if len(rows) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", help="index or unique part of the stack name (default: random)")
    ap.add_argument("--z", type=int, help=f"plane index (default: random plane in z = {Z_RANGE[0]}-{Z_RANGE[1]})")
    ap.add_argument("--selection", choices=["stable", "chosen"], default="stable")
    ap.add_argument("--crop", type=int, nargs=3, metavar=("Y", "X", "SIZE"),
                    help="zoom window (default: densest spine signal in the plane)")
    ap.add_argument("--full", action="store_true", help="show the whole plane instead of a zoom")
    ap.add_argument("--spines", action="store_true", help="also write the per-spine table for this stack")
    ap.add_argument("--show", action="store_true", help="open the figures in a window")
    ap.add_argument("--seed", type=int, help="random seed for --stack / --z defaults")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stack_path = resolve_stack(args.stack, rng)
    stem = Path(stack_path).stem
    per_model, best = load_settings(args.selection)
    best_model = best["model"]
    models = [m for m in pt.models if m in per_model]

    preds = {m: load_prediction(m, stack_path) for m in models}
    n_planes = preds[best_model][1].shape[0]
    if args.z is None:
        z = rng.randint(min(Z_RANGE[0], n_planes - 1), min(Z_RANGE[1], n_planes - 1))
    else:
        z = args.z
        if not 0 <= z < n_planes:
            raise SystemExit(f"--z must be between 0 and {n_planes - 1}")
    z0, z1 = z_window(z, n_planes)
    print(f"Stack: {stem} | plane z = {z} / {n_planes - 1} | selection: {args.selection} | "
          f"best model: {best_model}")

    raw = pt.load_raw(stack_path)[z].astype(float)
    lo, hi = np.percentile(raw, [1, 99.7])
    raw_n = np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1)

    # Zoom window
    if args.full:
        crop = (slice(0, raw.shape[0]), slice(0, raw.shape[1]))
        y0 = x0 = 0
        c = None
    else:
        if args.crop:
            y0, x0, c = args.crop
        else:
            c = min(pt.VIZ_CROP, *raw.shape)
            dens = ndimage.uniform_filter(
                (np.asarray(preds[best_model][1][z]) >= best["peak_threshold"]).astype(float), c)
            cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
            y0, x0 = cy - c // 2, cx - c // 2
        y0 = int(np.clip(y0, 0, raw.shape[0] - c))
        x0 = int(np.clip(x0, 0, raw.shape[1] - c))
        crop = (slice(y0, y0 + c), slice(x0, x0 + c))
    view = ("whole plane" if c is None else
            f"zoom y={y0} x={x0}, {c} px = {c * pt.DIMENSIONS['xy']:.0f} µm")

    bench_fn = pt.OUT_DIR / "deepd3_benchmark_results.csv"
    bench = pd.read_csv(bench_fn) if bench_fn.exists() else None

    def size_txt(cfg):
        return (f"{cfg['min_size_um3']}-{cfg['max_size_um3']} µm³, "
                f"planes {cfg['min_planes']}-{cfg['max_planes']}")

    def counts_txt(cfg, roi):
        row = stack_row(bench, stem, cfg)
        n_view = len(np.unique(roi)) - 1
        stack_part = f"{int(row['n_spines'])} in stack" if row is not None else "stack count n/a"
        return f"{n_view} ROIs in view | {stack_part}"

    pt.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = pt.OUT_DIR / f"plane_{stem}_z{z}_{args.selection}"

    # ── Figure 1: each model with its own best settings ──
    fig = plt.figure(figsize=(18, 4 + 4.6 * len(models)))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, len(models)])
    ax = top.subplots(1, 1)
    ax.imshow(raw_n, cmap="gray")
    if c is not None:
        ax.add_patch(plt.Rectangle((x0, y0), c, c, fill=False, edgecolor="yellow", linewidth=1.5))
    ax.set_title(f"Raw, whole plane (yellow = {view})", fontsize=9)
    ax.axis("off")

    axes = bottom.subplots(len(models), 4, squeeze=False)
    for i, m in enumerate(models):
        cfg = per_model[m]
        dend, spine = preds[m]
        roi = segment_plane(np.asarray(spine[z0:z1]), z - z0, cfg)[crop]
        dend_c, spine_c, raw_c = np.asarray(dend[z])[crop], np.asarray(spine[z])[crop], raw_n[crop]

        axes[i, 0].imshow(raw_c, cmap="gray")
        axes[i, 0].set_title(f"{m}\nraw", fontsize=9)
        axes[i, 1].imshow(raw_c, cmap="gray")
        axes[i, 1].imshow(np.ma.masked_less_equal(dend_c, pt.DEND_THRESHOLD),
                          cmap="cool", alpha=0.5, vmin=0, vmax=1)
        axes[i, 1].set_title(f"Dendrite (p > {pt.DEND_THRESHOLD})", fontsize=9)
        im = axes[i, 2].imshow(spine_c, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("Spine probability", fontsize=9)
        bottom.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.02)
        pt._show_segmentation(axes[i, 3], raw_c, dend_c, roi,
                              f"peak={cfg['peak_threshold']} delta={cfg['seed_delta']} | {size_txt(cfg)}\n"
                              f"{counts_txt(cfg, roi)}")
        for a in axes[i, :3]:
            a.axis("off")
        if m == best_model:
            pt._frame(axes[i, 3])
    bottom.suptitle(f"Each model with its own best settings ({args.selection}); red = best model",
                    fontsize=11)
    fig.suptitle(f"{stem} | z = {z}/{n_planes - 1}", fontsize=13, y=1.01)
    fn = f"{out_base}_models.png"
    fig.savefig(fn, dpi=120, bbox_inches="tight")
    print(f"Saved → {fn}")

    # ── Figure 2: peak x seed-delta sweep for the best model ──
    dend, spine = preds[best_model]
    slab = np.asarray(spine[z0:z1])
    dend_c, raw_c = np.asarray(dend[z])[crop], raw_n[crop]
    peaks, deltas = pt.roi_peak_thresholds, pt.roi_seed_deltas
    fig2, axes = plt.subplots(len(peaks), len(deltas), figsize=(3.8 * len(deltas), 4.0 * len(peaks)),
                              squeeze=False)
    for i, peak in enumerate(peaks):
        for j, delta in enumerate(deltas):
            cfg = dict(best, peak_threshold=peak, seed_delta=delta)
            roi = segment_plane(slab, z - z0, cfg)[crop]
            row = stack_row(bench, stem, cfg)
            inst = (f" | instab {row['instability']:.2f}"
                    if row is not None and pd.notna(row.get("instability")) else "")
            pt._show_segmentation(axes[i, j], raw_c, dend_c, roi,
                                  f"peak={peak} delta={delta}\n{counts_txt(cfg, roi)}{inst}")
            if np.isclose(peak, best["peak_threshold"]) and np.isclose(delta, best["seed_delta"]):
                pt._frame(axes[i, j])
    fig2.suptitle(f"{stem} | z = {z} | {best_model}, {size_txt(best)} | {view}\n"
                  f"peak × seed-delta sweep (red = {args.selection} setting)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    fn2 = f"{out_base}_sweep.png"
    fig2.savefig(fn2, dpi=110, bbox_inches="tight")
    print(f"Saved → {fn2}")

    if args.spines:
        t = pt.spine_table(stack_path, best, args.selection)
        fn3 = pt.OUT_DIR / f"spines_{stem}_{args.selection}.csv"
        t.to_csv(fn3, index=False)
        print(f"Saved → {fn3} ({len(t)} spines)")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
