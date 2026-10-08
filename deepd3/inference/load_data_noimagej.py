# Usage: python deepd3/inference/load_data_noimagej.py ./deepd3/inference/data -o ./deepd3/inference/data/processed --channel 1 --no-crop

import tifffile
import numpy as np
from pathlib import Path
import argparse
import time


def read_image_stack(filename: str, n_channels: int = 3) -> np.ndarray:
    """
    Returns: (Z, C, Y, X) array
    """
    vol = tifffile.imread(filename)
    
    # Case 1: tifffile already gives (Z, Y, X, C) — move C to axis 1
    if vol.ndim == 4 and vol.shape[-1] == n_channels:
        # (Z, Y, X, C) → (Z, C, Y, X)
        return np.transpose(vol, (0, 3, 1, 2))
    
    # Case 2: tifffile gives (C, Z, Y, X) — already correct
    if vol.ndim == 4 and vol.shape[0] == n_channels:
        # (C, Z, Y, X) → (Z, C, Y, X)
        return np.transpose(vol, (1, 0, 2, 3))
    
    # Case 3: flat pages (numPages, Y, X)
    if vol.ndim == 3:
        num_pages, ny, nx = vol.shape
        if num_pages % n_channels != 0:
            raise ValueError(f"{num_pages} pages not divisible by {n_channels}")
        num_z = num_pages // n_channels
        return vol.reshape(num_z, n_channels, ny, nx)
    
    raise ValueError(f"Unexpected shape {vol.shape}")   


def auto_crop(volume: np.ndarray, threshold: float = 0.01, margin: int = 32) -> np.ndarray:
    """Crop to bounding box of non-zero signal, with margin. volume is (Z, Y, X)."""
    nonzero = np.any(volume > threshold, axis=0)
    if not nonzero.any():
        print("  ⚠ No signal found, skipping crop")
        return volume
    rows = np.any(nonzero, axis=1)
    cols = np.any(nonzero, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = max(0, y0 - margin), min(volume.shape[1], y1 + margin)
    x0, x1 = max(0, x0 - margin), min(volume.shape[2], x1 + margin)
    return volume[:, y0:y1, x0:x1]


def save_for_deepd3(arr: np.ndarray, output_path: Path):
    """Save as single-channel 16-bit TIFF (multi-page, one page per Z-slice)."""
    arr16 = (arr * 65535).astype(np.uint16)
    tifffile.imwrite(str(output_path), arr16, photometric="minisblack")


def process_folder(input_dir: str, output_dir: str,
                   channel_idx: int = 1,
                   n_channels: int = 3,
                   crop: bool = True,
                   pattern: str = "*.tif*"):
    """
    Batch-process: read interleaved-page TIFFs → extract one channel → save.
    
    Parameters
    ----------
    input_dir : folder with raw multi-channel TIFFs (interleaved pages)
    output_dir : output folder
    channel_idx : 0-based index of channel to keep (1 = green/2nd)
    n_channels : total number of channels per Z-slice (default 3)
    crop : auto-crop to signal bounding box
    pattern : glob pattern
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    

    # Add timestamp so re-runs don't collide
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = sorted(in_path.glob(pattern))

    # Exclude already-processed files
    files = [f for f in files if "_green" not in f.stem]
    
    if not files:
        print(f"No files matching '{pattern}' in {in_path}")
        return []
    
    print(f"Found {len(files)} file(s) in {in_path}\n")
    
    processed = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        t0 = time.time()
        
        try:
            # Read and reshape: (Z, C, Y, X)
            stack = read_image_stack(str(f), n_channels=n_channels)
            
            # Extract the desired channel: (Z, Y, X)
            green = stack[:, channel_idx, :, :]
            
            if crop:
                green = auto_crop(green)
            
            # Normalize to [0, 1]
            gmin, gmax = green.min(), green.max()
            if gmax > gmin:
                green = (green - gmin) / (gmax - gmin)
            green = green.astype(np.float32)
            
            out_file = out_path / (f.stem + "_green.tif")
            save_for_deepd3(green, out_file)
            
            elapsed = time.time() - t0
            print(f"  → {out_file.name}  shape={green.shape}  ({elapsed:.1f}s)")
            processed.append({
                "source": f.name,
                "output": out_file.name,
                "shape": green.shape,
                "time_s": round(elapsed, 2)
            })
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        
        del stack, green
    
    print(f"\n{'='*50}")
    print(f"Done. {len(processed)}/{len(files)} files processed.")
    print(f"Output: {out_path}")
    
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-process raw TIFFs for DeepD3")
    parser.add_argument("input_dir", help="Folder with raw multi-channel TIFFs")
    parser.add_argument("-o", "--output", default="data/processed",
                        help="Output folder")
    parser.add_argument("--channel", type=int, default=1,
                        help="Channel index to keep, 0-based (default: 1 = 2nd/green)")
    parser.add_argument("--n-channels", type=int, default=3,
                        help="Total channels per Z-slice in the source (default: 3)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Disable auto-crop")
    parser.add_argument("--pattern", default="*.tif*",
                        help="Glob pattern for input files")
    
    args = parser.parse_args()
    process_folder(args.input_dir, args.output,
                   channel_idx=args.channel,
                   n_channels=args.n_channels,
                   crop=not args.no_crop,
                   pattern=args.pattern)   