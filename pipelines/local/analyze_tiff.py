import os
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import time
import psutil

def memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    return memory_gb

def format_bytes(bytes):
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024 or unit == 'TB':
            return f"{bytes:.2f} {unit}"
        bytes /= 1024

def analyze_tiff(filepath):
    """Analyze a TIFF file and display its properties"""
    start_time = time.time()
    print(f"Starting memory usage: {memory_usage():.2f} GB")
    print(f"Loading file: {filepath}")
    
    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {format_bytes(file_size)}")
    
    # Load TIFF
    try:
        with tifffile.TiffFile(filepath) as tif:
            # Get metadata without loading entire image
            print("\nTIFF Structure:")
            print(f"Number of pages: {len(tif.pages)}")
            print(f"ImageJ metadata: {tif.is_imagej}")
            print(f"OME metadata: {tif.is_ome}")
            print(f"Micromanager metadata: {tif.is_micromanager}")
            
            # Get first page info
            page = tif.pages[0]
            print(f"\nFirst page tags:")
            print(f"  Shape: {page.shape}")
            print(f"  Dtype: {page.dtype}")
            print(f"  Axes: {page.axes}")
            print(f"  Compression: {page.compression}")
            
            # Check if file is a stack and what are its dimensions
            if len(tif.series) > 0:
                series = tif.series[0]
                print(f"\nSeries info:")
                print(f"  Shape: {series.shape}")
                print(f"  Dtype: {series.dtype}")
                print(f"  Axes: {series.axes}")
            
            # Load the actual image data - be careful with memory
            print(f"\nLoading full image data...")
            before_mem = memory_usage()
            data = tif.asarray()
            after_mem = memory_usage()
            
            print(f"Memory before loading: {before_mem:.2f} GB")
            print(f"Memory after loading: {after_mem:.2f} GB")
            print(f"Memory increase: {after_mem - before_mem:.2f} GB")
            print(f"Image array shape: {data.shape}")
            print(f"Image array dtype: {data.dtype}")
            print(f"Theoretical memory usage: {np.prod(data.shape) * data.itemsize / (1024**3):.2f} GB")
            
            # Calculate statistics
            print("\nImage statistics:")
            print(f"  Min value: {data.min()}")
            print(f"  Max value: {data.max()}")
            print(f"  Mean value: {data.mean()}")
            print(f"  Std deviation: {data.std()}")
            
            # Non-zero values analysis
            non_zero = np.count_nonzero(data)
            print(f"  Non-zero values: {non_zero} ({non_zero/data.size*100:.2f}%)")
            
            # Check for slices with no data (all zeros)
            if len(data.shape) == 3:  # 3D volume
                z_slices = data.shape[0]
                empty_slices = sum(np.all(data[z] == 0) for z in range(z_slices))
                print(f"  Empty Z slices: {empty_slices}/{z_slices}")
                
                # Plot middle slice
                middle_z = z_slices // 2
                plt.figure(figsize=(10, 8))
                plt.imshow(data[middle_z], cmap='gray')
                plt.colorbar(label='Intensity')
                plt.title(f'Middle Z-slice (z={middle_z})')
                plt.tight_layout()
                
                # Save to avoid memory issues with display
                output_dir = Path("analysis_output")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{Path(filepath).stem}_middle_slice.png"
                plt.savefig(output_path)
                print(f"\nSaved middle slice visualization to: {output_path}")
                plt.close()
                
                # Histograms for intensity distribution
                plt.figure(figsize=(12, 10))
                
                # Overall histogram
                plt.subplot(2, 1, 1)
                plt.hist(data.flatten(), bins=100, alpha=0.7)
                plt.title('Intensity Histogram (all slices)')
                plt.xlabel('Intensity')
                plt.ylabel('Frequency')
                
                # Middle slice histogram
                plt.subplot(2, 1, 2)
                plt.hist(data[middle_z].flatten(), bins=100, alpha=0.7)
                plt.title(f'Intensity Histogram (middle slice z={middle_z})')
                plt.xlabel('Intensity')
                plt.ylabel('Frequency')
                
                plt.tight_layout()
                hist_path = output_dir / f"{Path(filepath).stem}_histograms.png"
                plt.savefig(hist_path)
                print(f"Saved intensity histograms to: {hist_path}")
                plt.close()
                
                # Z-profile (mean intensity across Z)
                z_means = np.mean(data, axis=(1, 2))
                plt.figure(figsize=(10, 6))
                plt.plot(range(z_slices), z_means)
                plt.title('Mean Intensity Across Z-slices')
                plt.xlabel('Z-slice')
                plt.ylabel('Mean Intensity')
                plt.grid(True, alpha=0.3)
                z_profile_path = output_dir / f"{Path(filepath).stem}_z_profile.png"
                plt.savefig(z_profile_path)
                print(f"Saved Z-profile to: {z_profile_path}")
                plt.close()
            
            # Create downsampled preview of full volume
            if len(data.shape) == 3 and data.shape[0] > 100:
                # Downsample factors
                downsample_z = max(1, data.shape[0] // 64)
                downsample_y = max(1, data.shape[1] // 512)
                downsample_x = max(1, data.shape[2] // 512)
                
                downsampled = data[::downsample_z, ::downsample_y, ::downsample_x]
                print(f"\nCreated downsampled preview with shape: {downsampled.shape}")
                print(f"Downsampling factors: z={downsample_z}, y={downsample_y}, x={downsample_x}")
                
                # Save downsampled version
                preview_path = output_dir / f"{Path(filepath).stem}_preview.tif"
                tifffile.imwrite(preview_path, downsampled)
                print(f"Saved downsampled preview to: {preview_path}")
                
    except Exception as e:
        print(f"Error analyzing TIFF: {e}")
        return
    
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
    print(f"Final memory usage: {memory_usage():.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Analyze a TIFF file and display its properties")
    parser.add_argument("filepath", help="Path to the TIFF file to analyze")
    args = parser.parse_args()
    
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist.")
        return
    
    analyze_tiff(filepath)

if __name__ == "__main__":
    main()
