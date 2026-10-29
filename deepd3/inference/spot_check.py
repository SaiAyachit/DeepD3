"""
spot_check.py — blinded manual check of the spines two models disagree on.

Without ground truth, the deciding question between two models is whether the spines only
one of them finds are real. This script samples spines from four groups (same whole-spine
matching and tolerance as compare_models.py, each model at its own best setting):
  only_A : found by model A only          (default A = 16F)
  only_B : found by model B only          (default B = 32F)
  both_A : A's ROIs also found by B       controls: how often agreed spines are real, one per
  both_B : B's ROIs also found by A       model (so outline shape does not reveal the group)
mixes them in random order and hides which group each sample came from, so you score them
without knowing the model.

Step 1 — sample (run from the repo root):
  python deepd3/inference/spot_check.py --stack m34_960_d1
Writes to results/spot_check/:
  spotcheck_<stack>.csv        the sheet you fill in (one row per sample, no model info)
  spotcheck_<stack>.tif        Fiji hyperstack, one frame (t) per sample = sample_id:
                               ch1 raw, ch2 outline of the sampled ROI (magenta); 96 x 96 px
                               (~14 µm) around the ROI centre, 11 planes; the ROI centre is
                               in the middle of the crop, middle plane (z = 6)
  spotcheck_<stack>_key.csv    which group/model each sample came from — do not open while scoring
  spotcheck_<stack>_counts.json group sizes, used for the estimates in step 3

Step 2 — score. Open the TIFF in Fiji, move through t (= sample_id) and z, and fill the
"verdict" column of spotcheck_<stack>.csv:
  y = a real spine (the outline sits on a spine head/neck protruding from a dendrite)
  n = not a spine (shaft, noise, out-of-focus blob, axon bouton, edge artefact)
  ? = can't tell (excluded from the estimates)
Optional "note" column, e.g. "split" (ROI covers part of a spine another ROI also covers) or
"merged" (ROI covers 2+ spines). For more context, x_px / y_px / fiji_slice locate the ROI in
the original stack (data/processed/<stack>.tif).
Keep it blind: while scoring, do NOT open the key, qc_<stack>__ref-*.tif or
compare_*_vs_*_<stack>.tif — their colours show which model found each spine.

Step 3 — results:
  python deepd3/inference/spot_check.py --score            # every scored stack + pooled
Prints and saves results/spot_check/spotcheck_scores.csv: fraction real per group with 95%
Wilson intervals, and for each model the estimated precision (fraction of its spines that are
real) and relative recall (fraction of the real spines found by either model that it finds).
The pooled row adds up each stack's estimates, so every stack counts by its number of spines,
not by its number of samples.

Options: --n 50 (per only-one-model group), --n-both 20 (per model), --seed 0,
--model-a/--model-b, --selection stable|chosen, --dilate-xy-um/--dilate-z-um, --crop-px 96,
--crop-planes 11,
--force (overwrite a sheet that already has verdicts).
"""
#python deepd3/inference/spot_check.py --stack m34_960_d1

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parameter_test as pt
from inspect_plane import load_settings, load_prediction, resolve_stack
from compare_models import (MATCH_DILATION_XY_UM, MATCH_DILATION_Z_UM, segment_stack, dilate,
                            split_matched, _lut)

SPOT_DIR = pt.OUT_DIR / "spot_check"
VERDICTS = {"y": True, "n": False}   # "?" and blanks are excluded


def short(model):
    return model.replace("DeepD3_", "")


def roi_info(lab, ids, spine):
    """Centroid, z range, volume and peak spine probability of the given label IDs."""
    if not len(ids):
        return pd.DataFrame(columns=["spine_id", "z", "y", "x", "z_min", "z_max", "volume_um3", "max_prob"])
    ids = np.asarray(ids)
    com = np.array(ndimage.center_of_mass(lab > 0, lab, ids)).reshape(-1, 3)
    objs = ndimage.find_objects(lab)
    vox = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    return pd.DataFrame({
        "spine_id": ids,
        "z": np.rint(com[:, 0]).astype(int), "y": np.rint(com[:, 1]).astype(int),
        "x": np.rint(com[:, 2]).astype(int),
        "z_min": [objs[i - 1][0].start for i in ids], "z_max": [objs[i - 1][0].stop - 1 for i in ids],
        "volume_um3": vox[ids] * pt.VOXEL_UM3,
        "max_prob": ndimage.maximum(spine, lab, ids),
    })


def crop(vol, z, y, x, nz, nxy):
    """nz x nxy x nxy block centred on (z, y, x), zero-padded at the stack edges."""
    out = np.zeros((nz, nxy, nxy), dtype=vol.dtype)
    lo = np.array([z - nz // 2, y - nxy // 2, x - nxy // 2])
    hi = lo + [nz, nxy, nxy]
    src_lo, src_hi = np.maximum(lo, 0), np.minimum(hi, vol.shape)
    if np.any(src_hi <= src_lo):
        return out
    dst = tuple(slice(a - l, b - l) for a, b, l in zip(src_lo, src_hi, lo))
    out[dst] = vol[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    return out


def outline(mask):
    """In-plane boundary of a 3D mask (each z plane eroded with a 2D cross)."""
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1)[None])


def sheet_has_verdicts(fn):
    if not fn.exists():
        return False
    v = pd.read_csv(fn, dtype=str)["verdict"].fillna("").str.strip()
    return bool((v != "").any())


def sample_stack(stack_path, cfg_a, cfg_b, args, rng):
    stem = Path(stack_path).stem
    sheet_fn = SPOT_DIR / f"spotcheck_{stem}.csv"
    if sheet_has_verdicts(sheet_fn) and not args.force:
        print(f"{stem}: {sheet_fn.name} already has verdicts — skipped (use --force to overwrite)")
        return

    a, b = cfg_a["model"], cfg_b["model"]
    spine_a = np.asarray(load_prediction(a, stack_path)[1])
    lab_a, n_a = segment_stack(spine_a, cfg_a)
    spine_b = np.asarray(load_prediction(b, stack_path)[1])
    lab_b, n_b = segment_stack(spine_b, cfg_b)
    matched_a = split_matched(lab_a, n_a, dilate(lab_b > 0, args.dilate_xy_um, args.dilate_z_um))
    matched_b = split_matched(lab_b, n_b, dilate(lab_a > 0, args.dilate_xy_um, args.dilate_z_um))
    ids = np.arange(len(matched_a))
    only_a, both_a = ids[1:][~matched_a[1:]], ids[1:][matched_a[1:]]
    only_b, both_b = np.arange(1, n_b + 1)[~matched_b[1:]], np.arange(1, n_b + 1)[matched_b[1:]]

    def pick(pool, n):
        return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))

    groups = [(f"only_{short(a)}", a, lab_a, spine_a, pick(only_a, args.n)),
              (f"only_{short(b)}", b, lab_b, spine_b, pick(only_b, args.n)),
              (f"both_{short(a)}", a, lab_a, spine_a, pick(both_a, args.n_both)),
              (f"both_{short(b)}", b, lab_b, spine_b, pick(both_b, args.n_both))]
    key = pd.concat([roi_info(lab, sel, sp).assign(group=g, model=m) for g, m, lab, sp, sel in groups],
                    ignore_index=True)
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key.insert(0, "sample_id", np.arange(1, len(key) + 1))
    if key.empty:
        print(f"{stem}: no spines to sample ({short(a)} {n_a}, {short(b)} {n_b}) — skipped")
        return

    raw = pt.load_raw(stack_path)
    labs = {a: lab_a, b: lab_b}
    nz, nxy = args.crop_planes, args.crop_px
    frames = np.zeros((len(key), nz, 2, nxy, nxy), dtype=np.uint16)
    for i, r in key.iterrows():
        frames[i, :, 0] = crop(raw, r.z, r.y, r.x, nz, nxy)
        frames[i, :, 1] = outline(crop(labs[r.model], r.z, r.y, r.x, nz, nxy) == r.spine_id)
    lo, hi = np.percentile(raw[::4], [1, 99.8])
    del raw, lab_a, lab_b, labs

    SPOT_DIR.mkdir(parents=True, exist_ok=True)
    tif_fn = SPOT_DIR / f"spotcheck_{stem}.tif"
    tifffile.imwrite(
        tif_fn, frames, imagej=True,
        resolution=(1 / pt.DIMENSIONS['xy'], 1 / pt.DIMENSIONS['xy']),
        metadata={"axes": "TZCYX", "mode": "composite", "unit": "um", "spacing": pt.DIMENSIONS['z'],
                  "Ranges": (float(lo), float(hi), 0.0, 1.0),
                  "LUTs": [_lut((1, 1, 1)), _lut((1, 0, 1))]},
    )
    sheet = pd.DataFrame({"sample_id": key["sample_id"], "x_px": key["x"], "y_px": key["y"],
                          "fiji_slice": key["z"] + 1, "first_slice": key["z_min"] + 1,
                          "last_slice": key["z_max"] + 1, "verdict": "", "note": ""})
    sheet.to_csv(sheet_fn, index=False)
    key.to_csv(SPOT_DIR / f"spotcheck_{stem}_key.csv", index=False)
    counts = {"model_a": a, "model_b": b, "setting_a": cfg_a, "setting_b": cfg_b,
              "n_a": n_a, "n_b": n_b, "n_only_a": len(only_a), "n_only_b": len(only_b),
              "n_both_a": len(both_a), "n_both_b": len(both_b),
              "dilate_xy_um": args.dilate_xy_um, "dilate_z_um": args.dilate_z_um, "seed": args.seed}
    (SPOT_DIR / f"spotcheck_{stem}_counts.json").write_text(json.dumps(counts, indent=2))
    print(f"{stem}: {short(a)} {n_a} (only {len(only_a)}) | {short(b)} {n_b} (only {len(only_b)}) | "
          f"sampled {key['group'].value_counts().to_dict()}")
    print(f"  Saved → {sheet_fn}\n  Saved → {tif_fn} ({len(key)} frames)")


# ── scoring ──
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)


def group_sizes(counts):
    a, b = short(counts["model_a"]), short(counts["model_b"])
    return {f"only_{a}": counts["n_only_a"], f"only_{b}": counts["n_only_b"],
            f"both_{a}": counts["n_both_a"], f"both_{b}": counts["n_both_b"]}


def _real(n, frac):
    """Estimated real spines in a group of n ROIs (0 for an empty group, NaN if unscored)."""
    return 0.0 if n == 0 else n * frac


def model_rows(stem, counts, est):
    """Precision and relative recall per model from per-group real-spine estimates `est`.
    Real spines of a model = its agreed ROIs x frac(both real) + its own-only ROIs x frac(only real);
    relative recall = its real spines / (its real spines + real spines only the other model finds)."""
    a, b = short(counts["model_a"]), short(counts["model_b"])
    rows = []
    for m, o, n in [(a, b, counts["n_a"]), (b, a, counts["n_b"])]:
        real, miss = est[f"both_{m}"] + est[f"only_{m}"], est[f"only_{o}"]
        rows.append({"stack": stem, "row": f"model_{m}", "n_in_stack": n, "est_real_spines": real,
                     "est_precision": real / n if n else np.nan,
                     "est_relative_recall": real / (real + miss) if real + miss > 0 else np.nan})
    return rows


def score_one(stem, sheet, key, counts):
    """Per-group fraction real, and the per-group real-spine estimates used for the model rows."""
    d = key.merge(sheet[["sample_id", "verdict", "note"]], on="sample_id")
    d["real"] = d["verdict"].fillna("").astype(str).str.strip().str.lower().map(VERDICTS)
    rows, est = [], {}
    for g, n_g in group_sizes(counts).items():
        s = d.loc[d["group"] == g, "real"]
        k, n = int(s.sum()), int(s.notna().sum())
        p, lo, hi = wilson(k, n)
        est[g] = _real(n_g, p)
        rows.append({"stack": stem, "row": g, "n_sampled": len(s), "n_scored": n, "n_real": k,
                     "frac_real": p, "ci_low": lo, "ci_high": hi, "n_in_stack": n_g})
    return pd.DataFrame(rows + model_rows(stem, counts, est)), est


def pool_key(c):
    return json.dumps([c["model_a"], c["model_b"], c["setting_a"], c["setting_b"],
                       c["dilate_xy_um"], c["dilate_z_um"]], sort_keys=True)


def pooled(scored):
    """Sum per-stack estimates: each stack counts by its number of spines, not of samples."""
    counts = dict(scored[0][0])
    for k in ["n_a", "n_b", "n_only_a", "n_only_b", "n_both_a", "n_both_b"]:
        counts[k] = sum(c[k] for c, _, _ in scored)
    est = {g: sum(e[g] for _, e, _ in scored) for g in scored[0][1]}
    rows = []
    for g, n_g in group_sizes(counts).items():
        sub = pd.concat([t[t["row"] == g] for _, _, t in scored])
        rows.append({"stack": "ALL (pooled)", "row": g, "n_sampled": sub["n_sampled"].sum(),
                     "n_scored": sub["n_scored"].sum(), "n_real": sub["n_real"].sum(),
                     "frac_real": est[g] / n_g if n_g else np.nan, "n_in_stack": n_g})
    return pd.DataFrame(rows + model_rows("ALL (pooled)", counts, est))


def score(args):
    stems = sorted(f.name[len("spotcheck_"):-len("_counts.json")] for f in SPOT_DIR.glob("spotcheck_*_counts.json"))
    if args.stack is not None:
        target = Path(resolve_stack(args.stack, None)).stem
        stems = [s for s in stems if s == target]
    tables, scored = [], []
    for stem in stems:
        sheet_fn = SPOT_DIR / f"spotcheck_{stem}.csv"
        if not sheet_has_verdicts(sheet_fn):
            print(f"{stem}: no verdicts yet — skipped")
            continue
        sheet = pd.read_csv(sheet_fn, dtype={"verdict": str, "note": str})
        key = pd.read_csv(SPOT_DIR / f"spotcheck_{stem}_key.csv")
        counts = json.loads((SPOT_DIR / f"spotcheck_{stem}_counts.json").read_text())
        bad = set(sheet["verdict"].dropna().str.strip().str.lower()) - set(VERDICTS) - {"?", ""}
        if bad:
            print(f"{stem}: unrecognised verdicts {sorted(bad)} treated as '?' (use y / n / ?)")
        t, est = score_one(stem, sheet, key, counts)
        tables.append(t)
        scored.append((counts, est, t))
    if not tables:
        raise SystemExit(f"No scored sheets in {SPOT_DIR}")

    if len(scored) > 1:
        if len({pool_key(c) for c, _, _ in scored}) == 1:
            tables.append(pooled(scored))
        else:
            print("Stacks were sampled with different models, settings or tolerances — no pooled estimate")

    out = pd.concat(tables, ignore_index=True)
    out.to_csv(SPOT_DIR / "spotcheck_scores.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(out.assign(stack=out["stack"].str[:24]).round(3).to_string(index=False))
    print(f"\nSaved → {SPOT_DIR / 'spotcheck_scores.csv'}")
    groups = out[~out["row"].str.startswith("model_") & (out["n_in_stack"] > 0)]
    if (groups["n_scored"] == 0).any():
        print("Note: some groups have spines but no scored samples — their estimates are NaN.")
    elif (groups["n_scored"] < 20).any():
        print("Note: groups with < 20 scored samples have wide intervals; score more before deciding.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", help="index or unique part of the stack name (default: all stacks)")
    ap.add_argument("--score", action="store_true", help="summarise filled-in sheets instead of sampling")
    ap.add_argument("--model-a", default="DeepD3_16F")
    ap.add_argument("--model-b", default="DeepD3_32F")
    ap.add_argument("--selection", choices=["stable", "chosen"], default="stable")
    ap.add_argument("--n", type=int, default=50, help="samples per only-one-model group")
    ap.add_argument("--n-both", type=int, default=20, help="samples per model of spines found by both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop-px", type=int, default=96)
    ap.add_argument("--crop-planes", type=int, default=11, help="covers ROIs up to max_planes (9)")
    ap.add_argument("--dilate-xy-um", type=float, default=MATCH_DILATION_XY_UM)
    ap.add_argument("--dilate-z-um", type=float, default=MATCH_DILATION_Z_UM)
    ap.add_argument("--force", action="store_true", help="overwrite sheets that already have verdicts")
    args = ap.parse_args()

    if args.score:
        return score(args)
    per_model, _ = load_settings(args.selection)
    for m in (args.model_a, args.model_b):
        if m not in per_model:
            raise SystemExit(f"No settings for {m}; available: {sorted(per_model)}")
    stacks = pt.stack_paths if args.stack is None else [resolve_stack(args.stack, None)]
    for stack_path in stacks:   # seeded per stack: the same stack gets the same sample however it is selected
        rng = np.random.default_rng([args.seed, pt.stack_paths.index(stack_path)])
        sample_stack(stack_path, per_model[args.model_a], per_model[args.model_b], args, rng)


if __name__ == "__main__":
    main()
