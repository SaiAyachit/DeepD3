"""
compare_models.py — where do two models disagree on which spines exist?

Each model segments the whole stack with its own best setting (deepd3_best_per_model.csv,
or CHOSEN_CONFIG with --selection chosen). Spines are compared as whole objects: a spine
counts as found by both models when its 3D ROI overlaps any ROI of the other model after
that model's ROIs are dilated by a small tolerance (default 0.5 µm in xy, 1 plane in z), so
slightly different boundaries drawn by the two models do not count as a disagreement.
The masks drawn in the outputs are the undilated ROIs.

Outputs per stack (in results/):
  compare_<A>_vs_<B>_<stack>.png : z max projection; only-A spines green, only-B red.
                                   Yellow = a green and a red spine at different depths
                                   overlapping in the projection.
  compare_<A>_vs_<B>_<stack>.tif : 3D ImageJ/Fiji composite to scroll through z
                                   (8-bit; ch1 raw grey, ch2 only-A green, ch3 only-B red,
                                   ch4 spines found by both, blue, with --shared).
                                   Skip with --no-tiff (~0.8 GB per stack).

Examples:
  python deepd3/inference/compare_models.py                       # all stacks
  python deepd3/inference/compare_models.py --stack m34_960_d1 --shared
  python deepd3/inference/compare_models.py --model-a DeepD3_8F --model-b DeepD3_32F
  python deepd3/inference/compare_models.py --dilate-xy-um 0 --dilate-z-um 0   # touching only
"""
#python deepd3/inference/compare_models.py --stack m34_960_d1

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parameter_test as pt
from inspect_plane import load_settings, load_prediction, resolve_stack

# Matching tolerance: each model's ROIs are dilated by this much before testing overlap
MATCH_DILATION_XY_UM = 0.5
MATCH_DILATION_Z_UM = 1.0


def segment_stack(spine, cfg):
    """Kept spine ROIs of the whole stack, renumbered 1..K."""
    labels = pt.flood_fill_labels(spine, cfg["peak_threshold"], cfg["seed_delta"])
    px, planes = pt.label_stats(labels)
    keep = pt.keep_mask(px, planes, *(cfg[k] for k in pt.SIZE_COLS))
    return pt.relabel(labels, keep), int(keep.sum())


def dilate(mask, r_xy_um, r_z_um):
    """Binary dilation with an ellipsoid of the given radii (µm); no-op for zero radii."""
    rz = int(r_z_um // pt.DIMENSIONS['z'])
    ry = int(r_xy_um // pt.DIMENSIONS['xy'])
    if rz == 0 and ry == 0:
        return mask
    off = pt._ellipsoid_offsets(max(r_xy_um, 1e-9), max(r_z_um, 1e-9))
    struct = np.zeros((2 * rz + 1, 2 * ry + 1, 2 * ry + 1), dtype=bool)
    struct[off[:, 0] + rz, off[:, 1] + ry, off[:, 2] + ry] = True
    return ndimage.binary_dilation(mask, structure=struct)


def split_matched(labels, n, other_dilated):
    """Boolean per label (index 0 unused): True if the ROI touches the other model's
    dilated ROIs."""
    matched = np.zeros(n + 1, dtype=bool)
    matched[np.unique(labels[(labels > 0) & other_dilated])] = True
    matched[0] = False
    return matched


def to_uint8(raw):
    lo, hi = np.percentile(raw[::4], [1, 99.8])
    return np.clip((raw.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-9)), 0, 255).astype(np.uint8)


def write_tiff(fn, channels, luts):
    """ImageJ composite hyperstack (Z, C, Y, X) with per-channel colour tables."""
    data = np.stack(channels, axis=1)
    tifffile.imwrite(
        fn, data, imagej=True,
        resolution=(1 / pt.DIMENSIONS['xy'], 1 / pt.DIMENSIONS['xy']),
        metadata={"axes": "ZCYX", "mode": "composite", "unit": "um",
                  "spacing": pt.DIMENSIONS['z'], "LUTs": luts},
    )


def _lut(rgb):
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp if c else np.zeros(256, np.uint8) for c in rgb])


def compare_stack(stack_path, cfg_a, cfg_b, args):
    stem = Path(stack_path).stem
    a, b = cfg_a["model"], cfg_b["model"]
    short = lambda m: m.replace("DeepD3_", "")
    lab_a, n_a = segment_stack(np.asarray(load_prediction(a, stack_path)[1]), cfg_a)
    lab_b, n_b = segment_stack(np.asarray(load_prediction(b, stack_path)[1]), cfg_b)

    matched_a = split_matched(lab_a, n_a, dilate(lab_b > 0, args.dilate_xy_um, args.dilate_z_um))
    matched_b = split_matched(lab_b, n_b, dilate(lab_a > 0, args.dilate_xy_um, args.dilate_z_um))
    n_only_a = n_a - int(matched_a.sum())
    n_only_b = n_b - int(matched_b.sum())

    unmatched_a, unmatched_b = ~matched_a, ~matched_b
    unmatched_a[0] = unmatched_b[0] = False
    only_a, only_b = unmatched_a[lab_a], unmatched_b[lab_b]
    shared = (matched_a[lab_a] | matched_b[lab_b]) if args.shared else None
    del lab_a, lab_b

    raw = pt.load_raw(stack_path)
    tol = f"tolerance {args.dilate_xy_um} µm xy / {args.dilate_z_um} µm z"
    print(f"{stem}: {short(a)} {n_a} | {short(b)} {n_b} | only {short(a)} {n_only_a} "
          f"({n_only_a / max(n_a, 1):.0%}) | only {short(b)} {n_only_b} ({n_only_b / max(n_b, 1):.0%}) | {tol}")

    # ── 2D: z max projection ──
    mip = raw.max(0).astype(float)
    lo, hi = np.percentile(mip, [1, 99.7])
    mip = np.clip((mip - lo) / max(hi - lo, 1e-9), 0, 1)
    proj_a, proj_b = only_a.any(0), only_b.any(0)
    rgb = np.repeat(mip[..., None] * 0.6, 3, axis=2)
    rgb[proj_a] = rgb[proj_a] * 0.2 + np.array([0.0, 1.0, 0.0]) * 0.8
    rgb[proj_b] = rgb[proj_b] * 0.2 + np.array([1.0, 0.0, 0.0]) * 0.8
    rgb[proj_a & proj_b] = [1.0, 1.0, 0.0]

    fig, ax = plt.subplots(figsize=(12, 12.8))
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    ax.legend(handles=[Patch(color="lime", label=f"only {short(a)}: {n_only_a} spines"),
                       Patch(color="red", label=f"only {short(b)}: {n_only_b} spines"),
                       Patch(color="yellow", label="both colours overlap in projection")],
              loc="lower right", fontsize=10, framealpha=0.85)
    setting = lambda c: (f"{short(c['model'])}: peak {c['peak_threshold']}, delta {c['seed_delta']}, "
                         f"{c['min_size_um3']}-{c['max_size_um3']} µm³, planes {c['min_planes']}-{c['max_planes']}")
    ax.set_title(f"{stem} — z max projection, spines found by only one model ({tol})\n"
                 f"{short(a)} {n_a} spines, {short(b)} {n_b} spines, "
                 f"found by both: {n_a - n_only_a} ({short(a)} ROIs) / {n_b - n_only_b} ({short(b)} ROIs)\n"
                 f"{setting(cfg_a)}   |   {setting(cfg_b)}", fontsize=10)
    base = pt.OUT_DIR / f"compare_{short(a)}_vs_{short(b)}_{stem}"
    fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {base}.png")

    # ── 3D: ImageJ composite to scroll through z ──
    if not args.no_tiff:
        channels = [to_uint8(raw), only_a.astype(np.uint8) * 255, only_b.astype(np.uint8) * 255]
        luts = [_lut((1, 1, 1)), _lut((0, 1, 0)), _lut((1, 0, 0))]
        if shared is not None:
            channels.append(shared.astype(np.uint8) * 255)
            luts.append(_lut((0, 0, 1)))
        del raw, only_a, only_b, shared
        write_tiff(f"{base}.tif", channels, luts)
        names = ["raw", f"only {short(a)}", f"only {short(b)}"] + (["both"] if args.shared else [])
        print(f"  Saved → {base}.tif (channels: {', '.join(names)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", help="index or unique part of the stack name (default: all stacks)")
    ap.add_argument("--model-a", default="DeepD3_16F", help="drawn in green")
    ap.add_argument("--model-b", default="DeepD3_32F", help="drawn in red")
    ap.add_argument("--selection", choices=["stable", "chosen"], default="stable")
    ap.add_argument("--dilate-xy-um", type=float, default=MATCH_DILATION_XY_UM,
                    help="matching tolerance in xy (µm)")
    ap.add_argument("--dilate-z-um", type=float, default=MATCH_DILATION_Z_UM,
                    help="matching tolerance in z (µm)")
    ap.add_argument("--shared", action="store_true", help="add a 4th TIFF channel with spines found by both")
    ap.add_argument("--no-tiff", action="store_true", help="only save the 2D projection")
    args = ap.parse_args()

    per_model, _ = load_settings(args.selection)
    for m in (args.model_a, args.model_b):
        if m not in per_model:
            raise SystemExit(f"No settings for {m}; available: {sorted(per_model)}")
    stacks = pt.stack_paths if args.stack is None else [resolve_stack(args.stack, None)]
    pt.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stack_path in stacks:
        compare_stack(stack_path, per_model[args.model_a], per_model[args.model_b], args)


if __name__ == "__main__":
    main()
