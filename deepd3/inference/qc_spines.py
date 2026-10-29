"""
qc_spines.py — quality control of spine detection and spine size across models and settings.

Configurations: each model (default 16F and 32F) at the reference setting and at its 4
neighbours in the peak x seed-delta grid, all with the reference size filters.
Reference = --ref-model at its own best setting from deepd3_best_per_model.csv (default: the
overall best model). Run once per reference model to see each model's consistency with
itself, e.g. --ref-model DeepD3_32F; outputs carry the reference in their name.

Every spine of every configuration is measured (size, brightness — see
parameter_test.spine_table) and matched to the reference spines: a spine matches the
reference spine it overlaps most after the reference ROIs are dilated by the matching
tolerance (as in compare_models.py).

Outputs (in results/; <ref> = reference config, e.g. 16F_p0.8_d0.3):
  qc_spines_<stack>__ref-<ref>.csv : one row per spine per configuration. spine_id = the
                          spine's value in that configuration's label channel of the TIFF.
  qc_summary.csv        : one row per stack x reference x configuration (counts, matching,
                          median sizes)
  qc_<stack>__ref-<ref>.png : counts, matching and size distributions per configuration
  qc_<stack>__ref-<ref>.tif : 16-bit ImageJ/Fiji composite to scroll through z. ch1 = raw data,
                          then one channel per --tiff configuration holding spine IDs (shown
                          as solid colour; default 16F green, 32F red, both at the reference
                          setting — overlap looks yellow). Select a label channel and hover a
                          spine: the status bar value is its spine_id in the CSV.

Key columns of qc_spines_<stack>__ref-<ref>.csv:
  config_id, model, peak_threshold, seed_delta, is_reference
  spine_id, z_px/y_px/x_px (centroid; Fiji x = x_px, y = y_px, slice = z_px + 1), z_min, z_max
  volume_um3, max_area_um2, equiv_diameter_um, n_planes, brightness_shaft_um3, qc_flags
  ref_spine_id        matched reference spine (NaN = no reference spine within tolerance)
  ref_volume_um3, volume_ratio (this / reference), max_area_ratio
  iou_with_ref        overlap / union with the matched reference spine (undilated)
  n_ref_touched       reference spines this ROI overlaps (>1 = merges reference spines)
  ref_shared_by       ROIs of this configuration matched to the same reference spine
                      (>1 = splits a reference spine)

Examples:
  python deepd3/inference/qc_spines.py --stack m34_960_d1
  python deepd3/inference/qc_spines.py --stack m34_960_d1 --ref-model DeepD3_32F --no-tiff
  python deepd3/inference/qc_spines.py                                  # all stacks
  python deepd3/inference/qc_spines.py --stack 0 --tiff 16F_p0.8_d0.3 16F_p0.8_d0.2 32F_p0.8_d0.3
"""
#python deepd3/inference/qc_spines.py --stack m34_960_d1 --ref-model DeepD3_32F --no-tiff


import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parameter_test as pt
from inspect_plane import load_settings, resolve_stack
from compare_models import MATCH_DILATION_XY_UM, MATCH_DILATION_Z_UM, _lut

# Chart colours per model (colour-blind-safe categorical slots 1-2); TIFF colours below
MODEL_COLORS = {"DeepD3_16F": "#2a78d6", "DeepD3_32F": "#eb6834", "DeepD3_8F": "#1baf7a"}
TIFF_COLORS = [(0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)]


def short(model):
    return model.replace("DeepD3_", "")


def config_id(cfg):
    return f"{short(cfg['model'])}_p{cfg['peak_threshold']}_d{cfg['seed_delta']}"


def qc_configs(models, ref):
    """Each model at the reference (peak, delta) and its 4 grid neighbours."""
    peaks, deltas = pt.roi_peak_thresholds, pt.roi_seed_deltas
    i = int(np.argmin(np.abs(np.array(peaks) - ref["peak_threshold"])))
    j = int(np.argmin(np.abs(np.array(deltas) - ref["seed_delta"])))
    settings = [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    settings = [(peaks[a], deltas[b]) for a, b in settings
                if 0 <= a < len(peaks) and 0 <= b < len(deltas)]
    return [dict(ref, model=m, peak_threshold=p, seed_delta=d) for m in models for p, d in settings]


def ellipsoid_struct(r_xy_um, r_z_um):
    rz = int(r_z_um // pt.DIMENSIONS['z'])
    ry = int(r_xy_um // pt.DIMENSIONS['xy'])
    off = pt._ellipsoid_offsets(max(r_xy_um, 1e-9), max(r_z_um, 1e-9))
    struct = np.zeros((2 * rz + 1, 2 * ry + 1, 2 * ry + 1), dtype=bool)
    struct[off[:, 0] + rz, off[:, 1] + ry, off[:, 2] + ry] = True
    return struct


def _pair_counts(a, b, n_b):
    """Voxel counts of every (a_label, b_label) pair where both are > 0."""
    m = (a > 0) & (b > 0)
    keys, counts = np.unique(a[m].astype(np.int64) * (n_b + 1) + b[m], return_counts=True)
    return pd.DataFrame({"id": keys // (n_b + 1), "ref": keys % (n_b + 1), "n": counts})


def match_to_reference(t, lab, ref_lab, ref_dil, ref_t):
    """Adds the matching columns to spine table t (rows = spines of lab)."""
    n_ref = len(ref_t)
    ref_px = ref_t.set_index("spine_id")["volume_um3"] / pt.VOXEL_UM3
    dil = _pair_counts(lab, ref_dil, n_ref)
    best = dil.sort_values("n", ascending=False).drop_duplicates("id").set_index("id")["ref"]
    und = _pair_counts(lab, ref_lab, n_ref)

    t = t.copy()
    t["ref_spine_id"] = t["spine_id"].map(best)
    ov = und.set_index(["id", "ref"])["n"]
    keys = list(zip(t["spine_id"], t["ref_spine_id"].fillna(0).astype(int)))
    overlap = np.array([ov.get(k, 0) for k in keys], dtype=float)
    own_px = t["volume_um3"].to_numpy() / pt.VOXEL_UM3
    ref_vox = t["ref_spine_id"].map(ref_px).to_numpy()
    t["iou_with_ref"] = np.where(t["ref_spine_id"].notna(), overlap / (own_px + ref_vox - overlap), np.nan)
    ref_cols = ref_t.set_index("spine_id")
    t["ref_volume_um3"] = t["ref_spine_id"].map(ref_cols["volume_um3"])
    t["volume_ratio"] = t["volume_um3"] / t["ref_volume_um3"]
    t["max_area_ratio"] = t["max_area_um2"] / t["ref_spine_id"].map(ref_cols["max_area_um2"])
    t["n_ref_touched"] = t["spine_id"].map(und.groupby("id").size()).fillna(0).astype(int)
    t["ref_shared_by"] = t["ref_spine_id"].map(t["ref_spine_id"].value_counts())
    return t


def summarize(t, stack, ref_id, n_ref):
    rows = []
    for cid, g in t.groupby("config_id", sort=False):
        m = g["ref_spine_id"].notna()
        r = {"stack": stack, "reference": ref_id, "config_id": cid, "model": g["model"].iloc[0],
             "peak_threshold": g["peak_threshold"].iloc[0], "seed_delta": g["seed_delta"].iloc[0],
             "is_reference": bool(g["is_reference"].iloc[0]), "n_spines": len(g),
             "median_volume_um3": g["volume_um3"].median(),
             "median_max_area_um2": g["max_area_um2"].median(),
             "median_equiv_diameter_um": g["equiv_diameter_um"].median(),
             "median_brightness_shaft_um3": g["brightness_shaft_um3"].median()}
        if not r["is_reference"]:
            lr = np.log2(g.loc[m, "volume_ratio"])
            r.update({
                "frac_matched_to_ref": m.mean(),
                "n_unmatched": int((~m).sum()),
                "n_ref_missed": n_ref - g.loc[m, "ref_spine_id"].nunique(),
                "n_splitting_ref": int((g.loc[m, "ref_shared_by"] > 1).sum()),
                "n_merging_ref": int((g["n_ref_touched"] > 1).sum()),
                "median_volume_ratio": g.loc[m, "volume_ratio"].median(),
                "volume_ratio_iqr_log2": float(lr.quantile(0.75) - lr.quantile(0.25)) if m.any() else np.nan,
                "median_max_area_ratio": g.loc[m, "max_area_ratio"].median(),
                "median_iou_with_ref": g.loc[m, "iou_with_ref"].median(),
            })
        rows.append(r)
    return pd.DataFrame(rows)


def plot_qc(t, summary, stack, fn):
    cfgs = summary["config_id"].tolist()
    settings = list(dict.fromkeys(f"p{p}\nd{d}" for p, d in zip(summary["peak_threshold"], summary["seed_delta"])))
    models = list(dict.fromkeys(summary["model"]))
    width = 0.8 / len(models)
    xpos = {(m, s): k + (i - (len(models) - 1) / 2) * width
            for k, s in enumerate(settings) for i, m in enumerate(models)}
    key = lambda r: (r["model"], f"p{r['peak_threshold']}\nd{r['seed_delta']}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    panels = [
        ("n_spines", "Spines in stack", "bar", None),
        ("frac_matched_to_ref", "Fraction matched to reference", "bar", None),
        ("volume_um3", "Volume (µm³)", "box", "log"),
        ("max_area_um2", "Largest cross-section (µm²)", "box", "log"),
        ("volume_ratio", "Volume / matched reference spine", "box", "log"),
        ("brightness_shaft_um3", "Brightness (shaft-µm³)", "box", "log"),
    ]
    for ax, (col, title, kind, scale) in zip(axes.flat, panels):
        for _, r in summary.iterrows():
            x, c = xpos[key(r)], MODEL_COLORS.get(r["model"], "#6250d6")
            if kind == "bar":
                v = r.get(col, np.nan)
                if pd.notna(v):
                    ax.bar(x, v, width * 0.92, color=c)
                elif col == "frac_matched_to_ref":
                    ax.text(x, 0.02, "ref", ha="center", va="bottom", fontsize=7, color="#555")
            else:
                vals = t.loc[t["config_id"] == r["config_id"], col].dropna()
                vals = vals[vals > 0]
                if len(vals):
                    ax.boxplot(vals, positions=[x], widths=width * 0.8, showfliers=False,
                               patch_artist=True, medianprops=dict(color="#222", linewidth=1.5),
                               boxprops=dict(facecolor=c, edgecolor=c, alpha=0.75),
                               whiskerprops=dict(color=c), capprops=dict(color=c))
        if col == "volume_ratio":
            ax.axhline(1, color="#888", linewidth=1, linestyle="--")
        if scale:
            ax.set_yscale(scale)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(settings, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ref_cfg = summary.loc[summary["is_reference"], "config_id"].iloc[0]
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS.get(m, "#6250d6")) for m in models]
    fig.legend(handles, [short(m) for m in models], loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(models), fontsize=9, frameon=False)
    fig.suptitle(f"{stack} — spine QC across models and settings (reference: {ref_cfg})\n"
                 f"x axis = peak threshold / seed delta; boxes = 25-75%, whiskers = 1.5 IQR", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def write_label_tiff(fn, raw, labels, colors):
    channels = [raw.astype(np.uint16)]
    for lab in labels:
        if lab.max() > np.iinfo(np.uint16).max:
            raise ValueError("More than 65535 spines; cannot store IDs in a 16-bit TIFF")
        channels.append(lab.astype(np.uint16))
    lo, hi = np.percentile(raw[::4], [1, 99.8])
    ranges = [float(lo), float(hi)] + [0.0, 1.0] * len(labels)   # labels: any ID = full colour
    tifffile.imwrite(
        fn, np.stack(channels, axis=1), imagej=True,
        resolution=(1 / pt.DIMENSIONS['xy'], 1 / pt.DIMENSIONS['xy']),
        metadata={"axes": "ZCYX", "mode": "composite", "unit": "um", "spacing": pt.DIMENSIONS['z'],
                  "Ranges": tuple(ranges), "LUTs": [_lut((1, 1, 1))] + [_lut(c) for c in colors]},
    )


def qc_stack(stack_path, configs, ref_cfg, tiff_ids, args):
    stem = Path(stack_path).stem
    ref_id = config_id(ref_cfg)
    print(f"\n{stem}: reference {ref_id}")

    ref_t, ref_lab = pt.spine_table(stack_path, ref_cfg, ref_id, return_labels=True)
    struct = ellipsoid_struct(args.dilate_xy_um, args.dilate_z_um)
    ref_dil = ndimage.grey_dilation(ref_lab, footprint=struct)
    tiff_labels = {ref_id: ref_lab} if ref_id in tiff_ids else {}

    tables = []
    for cfg in configs:
        cid = config_id(cfg)
        if cid == ref_id:
            t = ref_t.copy()
        else:
            t, lab = pt.spine_table(stack_path, cfg, cid, return_labels=True)
            t = match_to_reference(t, lab, ref_lab, ref_dil, ref_t)
            if cid in tiff_ids:
                tiff_labels[cid] = lab
            del lab
            gc.collect()
        t = t.rename(columns={"selection": "config_id"})
        t.insert(3, "peak_threshold", cfg["peak_threshold"])
        t.insert(4, "seed_delta", cfg["seed_delta"])
        t.insert(5, "is_reference", cid == ref_id)
        tables.append(t)
        m = t.get("ref_spine_id")
        print(f"  {cid:18s} {len(t):5d} spines"
              + (f" | matched to reference {m.notna().mean():.0%}" if m is not None else " (reference)"))

    t = pd.concat(tables, ignore_index=True)
    cols = ["ref_spine_id", "ref_volume_um3", "volume_ratio", "max_area_ratio", "iou_with_ref",
            "n_ref_touched", "ref_shared_by"]
    t = t[[c for c in t.columns if c not in cols] + [c for c in cols if c in t.columns]]
    tag = f"{stem}__ref-{ref_id}"
    t.to_csv(pt.OUT_DIR / f"qc_spines_{tag}.csv", index=False)
    summary = summarize(t, stem, ref_id, len(ref_t))
    plot_qc(t, summary, stem, pt.OUT_DIR / f"qc_{tag}.png")
    print(f"  Saved → {pt.OUT_DIR / f'qc_spines_{tag}.csv'}\n  Saved → {pt.OUT_DIR / f'qc_{tag}.png'}")

    del ref_dil
    if tiff_labels and not args.no_tiff:
        order = [c for c in tiff_ids if c in tiff_labels]
        write_label_tiff(pt.OUT_DIR / f"qc_{tag}.tif", pt.load_raw(stack_path),
                         [tiff_labels[c] for c in order], TIFF_COLORS[:len(order)])
        print(f"  Saved → {pt.OUT_DIR / f'qc_{tag}.tif'} (channels: raw, "
              + ", ".join(f"{c} IDs" for c in order) + ")")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", help="index or unique part of the stack name (default: all stacks)")
    ap.add_argument("--models", nargs="+", default=["DeepD3_16F", "DeepD3_32F"])
    ap.add_argument("--ref-model", help="model whose best setting is the reference "
                                        "(default: the overall best model)")
    ap.add_argument("--tiff", nargs="+", metavar="CONFIG_ID",
                    help="configurations to put in the TIFF as ID channels, e.g. 16F_p0.8_d0.3 "
                         "(default: each model at the reference setting)")
    ap.add_argument("--no-tiff", action="store_true")
    ap.add_argument("--dilate-xy-um", type=float, default=MATCH_DILATION_XY_UM)
    ap.add_argument("--dilate-z-um", type=float, default=MATCH_DILATION_Z_UM)
    args = ap.parse_args()

    per_model, best = load_settings("stable")
    ref_model = args.ref_model or best["model"]
    if ref_model not in per_model:
        raise SystemExit(f"No settings for --ref-model {ref_model}; available: {sorted(per_model)}")
    ref_cfg = {k: per_model[ref_model][k] for k in ["model"] + pt.PARAM_COLS}
    models = [ref_model] + [m for m in args.models if m != ref_model]
    configs = qc_configs(models, ref_cfg)
    ids = [config_id(c) for c in configs]
    # TIFF channel colours follow --models order (16F green, 32F red), not the reference
    tiff_models = [m for m in args.models if m in models] + [m for m in models if m not in args.models]
    tiff_ids = args.tiff or [config_id(dict(ref_cfg, model=m)) for m in tiff_models]
    unknown = [c for c in tiff_ids if c not in ids]
    if unknown:
        raise SystemExit(f"--tiff {unknown} not in the QC configurations: {ids}")
    if len(tiff_ids) > len(TIFF_COLORS):
        raise SystemExit(f"At most {len(TIFF_COLORS)} TIFF configurations")
    print("QC configurations:", ", ".join(ids))

    stacks = pt.stack_paths if args.stack is None else [resolve_stack(args.stack, None)]
    pt.OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [qc_stack(s, configs, ref_cfg, tiff_ids, args) for s in stacks]

    fn = pt.OUT_DIR / "qc_summary.csv"
    new = pd.concat(summaries, ignore_index=True)
    if fn.exists():   # keep rows of (stack, reference) pairs not re-run this time
        old = pd.read_csv(fn)
        if "reference" not in old.columns:   # written before --ref-model existed
            ref_of = old[old["is_reference"].astype(bool)].set_index("stack")["config_id"]
            old.insert(1, "reference", old["stack"].map(ref_of))
        done = set(zip(new["stack"], new["reference"]))
        old = old[[k not in done for k in zip(old["stack"], old["reference"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(fn, index=False)
    print(f"\nSaved → {fn}")
    show = ["stack", "reference", "config_id", "n_spines", "frac_matched_to_ref", "n_ref_missed", "n_splitting_ref",
            "n_merging_ref", "median_volume_um3", "median_volume_ratio", "median_iou_with_ref"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(new[new["stack"].isin([Path(s).stem for s in stacks]) & (new["reference"] == config_id(ref_cfg))][show]
              .assign(stack=lambda d: d["stack"].str[:16]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
