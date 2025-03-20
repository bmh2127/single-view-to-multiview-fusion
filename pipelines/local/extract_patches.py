import os
import argparse
import numpy as np
import tifffile
from pathlib import Path
import re
import json
import time
from tqdm import tqdm

def find_signal_regions(volume, threshold=160, min_size=5000):
    """
    Find regions with signal above threshold to focus patch extraction.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        threshold: Intensity threshold for considering a region as signal
        min_size: Minimum size (in voxels) for a region to be considered
        
    Returns:
        List of (z_start, z_end, y_start, y_end, x_start, x_end) for signal regions
    """
    # Get binary mask of voxels above threshold
    mask = volume > threshold
    
    # Find connected components (this is a simple approach - could use scikit-image)
    from scipy.ndimage import label, find_objects
    labeled_volume, num_features = label(mask)
    
    # Get bounding boxes for each region
    regions = []
    for i, region in enumerate(find_objects(labeled_volume)):
        # region is a tuple of slices (z_slice, y_slice, x_slice)
        # Convert to explicit coordinates
        z_start, z_end = region[0].start, region[0].stop
        y_start, y_end = region[1].start, region[1].stop
        x_start, x_end = region[2].start, region[2].stop
        
        # Calculate region size
        size = (z_end - z_start) * (y_end - y_start) * (x_end - x_start)
        
        # Only keep regions above minimum size
        if size >= min_size:
            regions.append((z_start, z_end, y_start, y_end, x_start, x_end))
    
    return regions

def extract_focused_patches(angle_volume, fused_volume, patch_size, z_range=None, 
                           focus_on_signal=True, signal_threshold=160, random_patches=0):
    """
    Extract patches focusing on signal-rich regions and the specified z-range.
    
    Args:
        angle_volume: 3D numpy array for angle view
        fused_volume: 3D numpy array for fused view
        patch_size: Tuple (z, y, x) for patch dimensions
        z_range: Tuple (z_start, z_end) to restrict extraction to specific z-range
        focus_on_signal: Whether to focus on signal-rich regions
        signal_threshold: Threshold for considering a region as signal
        random_patches: Number of additional random patches
        
    Returns:
        List of tuples (angle_patch, fused_patch, coords)
    """
    pz, py, px = patch_size
    z, y, x = angle_volume.shape
    
    # Restrict to specified z-range if provided
    if z_range:
        z_start, z_end = z_range
        angle_volume_subset = angle_volume[z_start:z_end]
        fused_volume_subset = fused_volume[z_start:z_end]
        # Adjust for the offset in coordinates
        z_offset = z_start
    else:
        angle_volume_subset = angle_volume
        fused_volume_subset = fused_volume
        z_offset = 0
    
    patches = []
    
    # Find signal regions if requested
    if focus_on_signal:
        print("Finding signal-rich regions...")
        regions = find_signal_regions(angle_volume_subset, signal_threshold)
        print(f"Found {len(regions)} signal-rich regions")
        
        # Extract patches centered on each region
        for region in regions:
            z_s, z_e, y_s, y_e, x_s, x_e = region
            
            # Calculate center of the region
            z_center = (z_s + z_e) // 2
            y_center = (y_s + y_e) // 2
            x_center = (x_s + x_e) // 2
            
            # Calculate patch boundaries centered on the region
            z_start = max(0, z_center - pz // 2)
            y_start = max(0, y_center - py // 2)
            x_start = max(0, x_center - px // 2)
            
            # Ensure patch fits within the volume
            if z_start + pz > angle_volume_subset.shape[0]:
                z_start = angle_volume_subset.shape[0] - pz
            if y_start + py > angle_volume_subset.shape[1]:
                y_start = angle_volume_subset.shape[1] - py
            if x_start + px > angle_volume_subset.shape[2]:
                x_start = angle_volume_subset.shape[2] - px
            
            # Extract patches
            angle_patch = angle_volume_subset[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px]
            fused_patch = fused_volume_subset[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px]
            
            # Add z_offset to coordinates to get original coordinates
            patches.append((angle_patch, fused_patch, (z_start + z_offset, y_start, x_start)))
            
            # Extract additional patches around the region
            for _ in range(2):  # Extract 2 additional patches per region
                # Random offset within the region
                dz = np.random.randint(-pz//4, pz//4+1)
                dy = np.random.randint(-py//4, py//4+1)
                dx = np.random.randint(-px//4, px//4+1)
                
                # New starting position
                nz_start = max(0, min(z_start + dz, angle_volume_subset.shape[0] - pz))
                ny_start = max(0, min(y_start + dy, angle_volume_subset.shape[1] - py))
                nx_start = max(0, min(x_start + dx, angle_volume_subset.shape[2] - px))
                
                # Extract patches
                angle_patch = angle_volume_subset[nz_start:nz_start+pz, ny_start:ny_start+py, nx_start:nx_start+px]
                fused_patch = fused_volume_subset[nz_start:nz_start+pz, ny_start:ny_start+py, nx_start:nx_start+px]
                
                # Add z_offset to coordinates
                patches.append((angle_patch, fused_patch, (nz_start + z_offset, ny_start, nx_start)))
    
    # Extract random patches if requested
    if random_patches > 0:
        print(f"Extracting {random_patches} random patches...")
        for _ in range(random_patches):
            # Random starting position
            z_size = angle_volume_subset.shape[0]
            y_size = angle_volume_subset.shape[1]
            x_size = angle_volume_subset.shape[2]
            
            if z_size > pz:    
                z_start = np.random.randint(0, z_size - pz)
            else:
                z_start = 0
                
            if y_size > py:    
                y_start = np.random.randint(0, y_size - py)
            else:
                y_start = 0
                
            if x_size > px:
                x_start = np.random.randint(0, x_size - px)
            else:
                x_start = 0
            
            # Extract patches
            angle_patch = angle_volume_subset[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px]
            fused_patch = fused_volume_subset[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px]
            
            # Skip if patches are the wrong size
            if angle_patch.shape != (pz, py, px) or fused_patch.shape != (pz, py, px):
                continue
                
            # Add z_offset to coordinates
            patches.append((angle_patch, fused_patch, (z_start + z_offset, y_start, x_start)))
    
    return patches

def find_file_pairs(data_dir, angle_pattern=r"image_(\d+)_nucleus_angle\.tif", fused_pattern=r"image_(\d+)_nucleus_fused\.tif"):
    """Find all pairs of angle and fused files."""
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.tif"))
    
    # Separate angle and fused files
    angle_files = {}
    fused_files = {}
    
    for file_path in all_files:
        # Check for angle files
        angle_match = re.match(angle_pattern, file_path.name)
        if angle_match:
            image_num = int(angle_match.group(1))
            angle_files[image_num] = file_path
            continue
            
        # Check for fused files
        fused_match = re.match(fused_pattern, file_path.name)
        if fused_match:
            image_num = int(fused_match.group(1))
            fused_files[image_num] = file_path
    
    # Create pairs
    file_pairs = []
    for image_num in sorted(set(angle_files.keys()) & set(fused_files.keys())):
        file_pairs.append((angle_files[image_num], fused_files[image_num]))
    
    return file_pairs

def normalize_volume(volume, pmin=2, pmax=99.8):
    """Apply percentile normalization to the volume."""
    low = np.percentile(volume, pmin)
    high = np.percentile(volume, pmax)
    
    if high > low:
        volume = (volume - low) / (high - low)
        # Clip values to [0, 1] range
        volume = np.clip(volume, 0, 1)
    
    return volume

def main():
    parser = argparse.ArgumentParser(description="Extract focused patches from TIFF files for FuseMyCell training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing TIFF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save patches")
    parser.add_argument("--patch-size", type=str, default="32,64,64", help="Patch size (z,y,x)")
    parser.add_argument("--z-range", type=str, default="50,175", help="Z-range for patch extraction (z_start,z_end)")
    parser.add_argument("--focus-on-signal", action="store_true", help="Focus extraction on signal-rich regions")
    parser.add_argument("--signal-threshold", type=int, default=160, help="Threshold for signal regions")
    parser.add_argument("--random-patches", type=int, default=10, help="Number of random patches per pair")
    parser.add_argument("--normalize", action="store_true", help="Apply percentile normalization to volumes")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of file pairs to process")
    parser.add_argument("--format", type=str, default="uint16", choices=["uint16", "float32"], 
                       help="Output format for patches")
    args = parser.parse_args()
    
    # Parse patch size
    patch_size = tuple(map(int, args.patch_size.split(',')))
    
    # Parse z-range
    z_range = tuple(map(int, args.z_range.split(','))) if args.z_range else None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    angle_dir = output_dir / "angle_patches"
    fused_dir = output_dir / "fused_patches"
    angle_dir.mkdir(exist_ok=True, parents=True)
    fused_dir.mkdir(exist_ok=True, parents=True)
    
    # Find file pairs
    file_pairs = find_file_pairs(args.data_dir)
    print(f"Found {len(file_pairs)} file pairs")
    
    # Limit number of samples if specified
    if args.max_samples is not None and len(file_pairs) > args.max_samples:
        file_pairs = file_pairs[:args.max_samples]
        print(f"Limited to {args.max_samples} file pairs")
    
    # Process each pair
    all_metadata = []
    
    for pair_idx, (angle_file, fused_file) in enumerate(tqdm(file_pairs, desc="Processing pairs")):
        print(f"\nProcessing pair {pair_idx+1}/{len(file_pairs)}: {angle_file.name} and {fused_file.name}")
        
        try:
            # Load angle file
            angle_volume = tifffile.imread(angle_file)
            
            # Handle different dimensions
            if len(angle_volume.shape) == 2:  # Single 2D image
                angle_volume = angle_volume[np.newaxis, ...]
            elif len(angle_volume.shape) == 4:  # Multiple channels
                angle_volume = angle_volume[..., 0]  # Take first channel
            
            # Load fused file
            fused_volume = tifffile.imread(fused_file)
            
            # Handle different dimensions
            if len(fused_volume.shape) == 2:  # Single 2D image
                fused_volume = fused_volume[np.newaxis, ...]
            elif len(fused_volume.shape) == 4:  # Multiple channels
                fused_volume = fused_volume[..., 0]  # Take first channel
            
            print(f"Loaded volumes: Angle shape {angle_volume.shape}, Fused shape {fused_volume.shape}")
            
            # Normalize if requested
            if args.normalize:
                print("Normalizing volumes...")
                angle_volume = normalize_volume(angle_volume)
                fused_volume = normalize_volume(fused_volume)
            
            # Extract patches using the focused approach
            patches = extract_focused_patches(
                angle_volume, 
                fused_volume, 
                patch_size, 
                z_range=z_range, 
                focus_on_signal=args.focus_on_signal,
                signal_threshold=args.signal_threshold,
                random_patches=args.random_patches
            )
            
            print(f"Extracted {len(patches)} patch pairs")
            
            # Extract image number for filenames
            match = re.match(r"image_(\d+)_", angle_file.name)
            if match:
                image_num = match.group(1)
            else:
                image_num = f"{pair_idx:03d}"
            
            # Save patches
            for i, (angle_patch, fused_patch, coords) in enumerate(patches):
                # Create unique ID for this patch
                patch_id = f"{image_num}_{i:04d}"
                
                # Convert to the specified format
                if args.format == "float32" and angle_patch.dtype != np.float32:
                    angle_patch = angle_patch.astype(np.float32)
                    fused_patch = fused_patch.astype(np.float32)
                elif args.format == "uint16" and angle_patch.dtype != np.uint16:
                    # Scale back to uint16 range
                    angle_patch = (angle_patch * 65535).astype(np.uint16)
                    fused_patch = (fused_patch * 65535).astype(np.uint16)
                
                # Save patches
                angle_path = angle_dir / f"angle_{patch_id}.tif"
                fused_path = fused_dir / f"fused_{patch_id}.tif"
                
                tifffile.imwrite(angle_path, angle_patch)
                tifffile.imwrite(fused_path, fused_patch)
                
                # Add to metadata
                z, y, x = coords
                all_metadata.append({
                    "patch_id": patch_id,
                    "image_num": image_num,
                    "angle_file": str(angle_file.name),
                    "fused_file": str(fused_file.name),
                    "angle_path": str(angle_path.relative_to(output_dir)),
                    "fused_path": str(fused_path.relative_to(output_dir)),
                    "coords": {"z": int(z), "y": int(y), "x": int(x)},
                    "shape": {"z": int(angle_patch.shape[0]), "y": int(angle_patch.shape[1]), "x": int(angle_patch.shape[2])},
                })
            
            # Clean up to free memory
            del angle_volume, fused_volume, patches
            
        except Exception as e:
            print(f"Error processing pair: {e}")
    
    # Save metadata
    metadata_path = output_dir / "patch_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nExtracted {len(all_metadata)} patch pairs")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)