# Usage: python test_deepd3_models.py
"""
Testing framework for DeepD3 pre-trained models (no ground truth).
Run: python test_deepd3_models.py
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR = Path("./deepd3/inference/pretrained_models")  # put your .h5 files here
MODELS = {
    "8F":     MODEL_DIR / "DeepD3_8F.h5",
    "16F":    MODEL_DIR / "DeepD3_16F.h5",
    "32F":    MODEL_DIR / "DeepD3_32F.h5"
}

# Test data
processed_dir = Path("./data/processed")
test_files = sorted(processed_dir.glob("*_green.tif"))
if not test_files:
    raise FileNotFoundError(f"No *_green.tif files found in {processed_dir}")
TEST_DATA_PATH = str(test_files[0])
SWEEP_FILES = test_files[:3]

# Inference tile size (must be multiple of 32)
TILE_SIZE = 512
OVERLAP = 128  # overlap between tiles

# Binarization thresholds (from DeepD3 paper)
SPINE_THRESH = 0.3
DENDRITE_THRESH = 0.7
MIN_SPINE_AREA = 3
DILATION_ITERS = 12


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model(path: Path) -> tf.keras.Model:
    """Load a DeepD3 .h5 Keras model."""
    return tf.keras.models.load_model(str(path), compile=False)


def prepare_input(volume: np.ndarray) -> np.ndarray:
    """
    Prepare a single-channel 3D volume for DeepD3 inference.
    DeepD3 expects input in [-1, 1], shape (1, Z, Y, X, 1) for 3D
    or processed tile-by-tile in XY with full Z.
    """
    # Rescale to [-1, 1]
    vol = volume.astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = 2.0 * (vol - vmin) / (vmax - vmin) - 1.0
    else:
        vol = np.zeros_like(vol)
    
    # Add batch and channel dims: (1, Z, Y, X, 1)
    return vol[np.newaxis, ..., np.newaxis]


def tile_inference(model: tf.keras.Model, volume: np.ndarray,
                   tile_size: int = TILE_SIZE, overlap: int = OVERLAP) -> tuple:
    """
    Sliding-window inference over XY, full Z.
    Returns (spine_probs, dendrite_probs) as (Z, Y, X) arrays.
    """
    Z, Y, X = volume.shape
    vol_4d = prepare_input(volume)  # (1, Z, Y, X, 1)
    
    # Output accumulators
    spine_out = np.zeros((Z, Y, X), dtype=np.float32)
    dend_out = np.zeros((Z, Y, X), dtype=np.float32)
    weight = np.zeros((Z, Y, X), dtype=np.float32)  # for averaging overlaps
    
    step = tile_size - overlap
    
    for y0 in range(0, max(1, Y - overlap), step):
        for x0 in range(0, max(1, X - overlap), step):
            y1 = min(y0 + tile_size, Y)
            x1 = min(x0 + tile_size, X)
            
            # Pad tile if it's smaller than tile_size (edge case)
            tile = vol_4d[:, :, y0:y1, x0:x1, :]
            pad_y = tile_size - (y1 - y0)
            pad_x = tile_size - (x1 - x0)
            if pad_y > 0 or pad_x > 0:
                tile = np.pad(tile, 
                              ((0,0),(0,0),(0,pad_y),(0,pad_x),(0,0)),
                              mode='edge')
            
            # Run inference
            pred = model.predict(tile, verbose=0)
            
            # DeepD3 dual-decoder: outputs [spine, dendrite] or two separate outputs
            # Adjust based on your model's output structure
            if isinstance(pred, (list, tuple)):
                spine_tile, dend_tile = pred[0], pred[1]
            else:
                # Single output with 2 channels: (1, Z, Y, X, 2)
                spine_tile = pred[:, :, :, :, 0]
                dend_tile = pred[:, :, :, :, 1]
            
            # Remove padding
            spine_tile = spine_tile[:, :, :y1-y0, :x1-x0]
            dend_tile = dend_tile[:, :, :y1-y0, :x1-x0]
            
            # Accumulate with weighting (center of tile weighted more)
            spine_out[:, y0:y1, x0:x1] += spine_tile
            dend_out[:, y0:y1, x0:x1] += dend_tile
            weight[:, y0:y1, x0:x1] += 1
    
    # Average overlapping regions
    weight[weight == 0] = 1  # avoid div by zero
    spine_out /= weight
    dend_out /= weight
    
    return spine_out, dend_out


def postprocess(spine_probs: np.ndarray, dend_probs: np.ndarray) -> tuple:
    """
    Apply DeepD3 post-processing: binarize, remove small spines,
    remove spines outside dilated dendrite mask.
    Returns (spine_mask, dend_mask) as binary uint8 arrays.
    """
    # Binarize
    spine_mask = (spine_probs >= SPINE_THRESH).astype(np.uint8)
    dend_mask = (dend_probs >= DENDRITE_THRESH).astype(np.uint8)
    
    # Remove small spine components
    labeled, n_comp = ndimage.label(spine_mask)
    for i in range(1, n_comp + 1):
        if np.sum(labeled == i) < MIN_SPINE_AREA:
            spine_mask[labeled == i] = 0
    
    # Dilation of dendrite mask, remove disconnected spines
    if np.any(dend_mask):
        dilated = ndimage.binary_dilation(dend_mask, iterations=DILATION_ITERS)
        spine_mask[~dilated] = 0
    
    return spine_mask, dend_mask


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def ok(self, name, detail=""):
        self.passed.append(name)
        print(f"  ✅ PASS: {name} {detail}")
    
    def fail(self, name, detail=""):
        self.failed.append(name)
        print(f"  ❌ FAIL: {name} {detail}")
    
    def warn(self, name, detail=""):
        self.warnings.append(name)
        print(f"  ⚠️  WARN: {name} {detail}")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"RESULTS: {len(self.passed)} passed, "
              f"{len(self.failed)} failed, {len(self.warnings)} warnings")
        if self.failed:
            print(f"  Failed: {self.failed}")
        print(f"{'='*60}")


def test_model_loads(model_name: str, model_path: Path, results: TestResults):
    """Test 1: Model loads without error and has expected structure."""
    try:
        model = load_model(model_path)
        results.ok(f"{model_name} loads", f"({len(model.layers)} layers)")
        return model
    except Exception as e:
        results.fail(f"{model_name} loads", str(e))
        return None


def test_output_shape(model, volume: np.ndarray, results: TestResults, name: str):
    """Test 2: Output shape matches input spatial dimensions."""
    Z, Y, X = volume.shape
    try:
        spine_probs, dend_probs = tile_inference(model, volume)
        
        assert spine_probs.shape == (Z, Y, X), \
            f"Spine shape {spine_probs.shape} != input {(Z,Y,X)}"
        assert dend_probs.shape == (Z, Y, X), \
            f"Dendrite shape {dend_probs.shape} != input {(Z,Y,X)}"
        
        results.ok(f"{name} output shape")
        return spine_probs, dend_probs
    except Exception as e:
        results.fail(f"{name} output shape", str(e))
        return None, None


def test_output_range(spine_probs, dend_probs, results: TestResults, name: str):
    """Test 3: Probabilities are in [0, 1]."""
    s_min, s_max = spine_probs.min(), spine_probs.max()
    d_min, d_max = dend_probs.min(), dend_probs.max()
    
    if 0 <= s_min and s_max <= 1 and 0 <= d_min and d_max <= 1:
        results.ok(f"{name} output range", 
                   f"(spine: [{s_min:.3f}, {s_max:.3f}], dend: [{d_min:.3f}, {d_max:.3f}])")
    else:
        results.fail(f"{name} output range",
                     f"(spine: [{s_min:.3f}, {s_max:.3f}], dend: [{d_min:.3f}, {d_max:.3f}])")


def test_not_trivial(spine_probs, dend_probs, results: TestResults, name: str):
    """Test 4: Output is not all-zeros or all-ones (model is actually doing something)."""
    s_frac = spine_probs.mean()
    d_frac = dend_probs.mean()
    
    if 0.001 < s_frac < 0.999 and 0.001 < d_frac < 0.999:
        results.ok(f"{name} non-trivial output",
                   f"(spine mean={s_frac:.4f}, dend mean={d_frac:.4f})")
    else:
        results.warn(f"{name} non-trivial output",
                     f"(spine mean={s_frac:.4f}, dend mean={d_frac:.4f})")


def test_reproducibility(model, volume: np.ndarray, results: TestResults, name: str):
    """Test 5: Running inference twice gives identical results."""
    s1, d1 = tile_inference(model, volume)
    s2, d2 = tile_inference(model, volume)
    
    s_diff = np.abs(s1 - s2).max()
    d_diff = np.abs(d1 - d2).max()
    
    if s_diff < 1e-6 and d_diff < 1e-6:
        results.ok(f"{name} reproducibility", f"(max diff: {s_diff:.2e})")
    else:
        results.fail(f"{name} reproducibility", f"(max diff: spine={s_diff:.2e}, dend={d_diff:.2e})")


def test_postprocess_valid(spine_probs, dend_probs, results: TestResults, name: str):
    """Test 6: After post-processing, spine mask ⊆ dilated dendrite mask."""
    spine_mask, dend_mask = postprocess(spine_probs, dend_probs)
    
    n_spines = ndimage.label(spine_mask)[1]
    n_dend = ndimage.label(dend_mask)[1]
    
    # All spines should be within dilated dendrite region
    if np.any(dend_mask):
        dilated = ndimage.binary_dilation(dend_mask, iterations=DILATION_ITERS)
        orphan_spines = np.sum(spine_mask & ~dilated)
        if orphan_spines == 0:
            results.ok(f"{name} post-process valid",
                       f"({n_spines} spine CCs, {n_dend} dend CCs)")
        else:
            results.fail(f"{name} post-process valid",
                         f"({orphan_spines} orphan spine voxels)")
    else:
        results.warn(f"{name} post-process valid", "(no dendrites detected)")
    
    return spine_mask, dend_mask


def test_cross_model_consistency(results: TestResults, 
                                  all_probs: dict):
    """
    Test 7: Different model sizes should produce correlated outputs.
    If all models give wildly different results, something is wrong.
    """
    names = list(all_probs.keys())
    if len(names) < 2:
        results.warn("cross-model consistency", "(need ≥2 models)")
        return
    
    # Compare pairwise using mean absolute difference
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            s_i = all_probs[names[i]][0]
            s_j = all_probs[names[j]][0]
            # Normalize to same scale for comparison
            mad = np.abs(s_i - s_j).mean()
            # Heuristic: if MAD > 0.5, models disagree a lot
            if mad < 0.5:
                results.ok(f"cross-model {names[i]}↔{names[j]}", f"(MAD={mad:.3f})")
            else:
                results.warn(f"cross-model {names[i]}↔{names[j]}", f"(MAD={mad:.3f} — large disagreement)")


def test_noise_robustness(model, volume: np.ndarray, results: TestResults, name: str):
    """
    Test 8: Adding small Gaussian noise should not drastically change the output.
    """
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, volume.shape).astype(np.float32)
    noisy_volume = np.clip(volume + noise, 0, 1)
    
    s_clean, d_clean = tile_inference(model, volume)
    s_noisy, d_noisy = tile_inference(model, noisy_volume)
    
    # IoU on binarized outputs
    s_clean_bin = (s_clean >= SPINE_THRESH)
    s_noisy_bin = (s_noisy >= SPINE_THRESH)
    
    if s_clean_bin.sum() > 0:
        iou = np.sum(s_clean_bin & s_noisy_bin) / np.sum(s_clean_bin | s_noisy_bin)
        if iou > 0.8:
            results.ok(f"{name} noise robustness", f"(spine IoU={iou:.3f})")
        else:
            results.warn(f"{name} noise robustness", f"(spine IoU={iou:.3f})")
    else:
        results.warn(f"{name} noise robustness", "(no spines in clean output)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DeepD3 Pre-trained Model Testing Framework")
    print("=" * 60)
    
    # Load test volume
    print(f"\nLoading test data: {TEST_DATA_PATH}")
    volume = tifffile.imread(TEST_DATA_PATH).astype(np.float32)
    if volume.max() > 1:
        volume = volume / volume.max()
    print(f"  Shape: {volume.shape} (Z, Y, X)")
    
    results = TestResults()
    all_probs = {}
    
    for name, path in MODELS.items():
        if not path.exists():
            results.warn(f"{name} exists", f"({path} not found)")
            continue
        
        print(f"\n--- {name} ({path.name}) ---")
        
        # Test 1: Load
        model = test_model_loads(name, path, results)
        if model is None:
            continue
        
        # Test 2: Output shape
        spine_probs, dend_probs = test_output_shape(model, volume, results, name)
        if spine_probs is None:
            continue
        
        # Store for cross-model comparison
        all_probs[name] = (spine_probs, dend_probs)
        
        # Test 3: Range
        test_output_range(spine_probs, dend_probs, results, name)
        
        # Test 4: Non-trivial
        test_not_trivial(spine_probs, dend_probs, results, name)
        
        # Test 5: Reproducibility
        test_reproducibility(model, volume, results, name)
        
        # Test 6: Post-process validity
        test_postprocess_valid(spine_probs, dend_probs, results, name)
        
        # Test 8: Noise robustness (skip for 32F if too slow)
        if "32" not in name:
            test_noise_robustness(model, volume, results, name)
        
        # Free GPU memory
        del model
        tf.keras.backend.clear_session()
    
    # Test 7: Cross-model consistency
    print(f"\n--- Cross-model consistency ---")
    test_cross_model_consistency(results, all_probs)
    
    # Final summary
    results.summary()
    
    return len(results.failed) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)   