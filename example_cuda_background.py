#!/usr/bin/env python3
"""
Example usage of CUDA-accelerated background subtraction.

This example shows how to use the enhanced BackgroundSubtractor with automatic
CUDA acceleration and CPU fallback.
"""

import numpy as np
import logging
from perinuclear_analysis.imaging_preprocessing.background_subtraction.background_subtractor import BackgroundSubtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate CUDA-accelerated background subtraction."""
    
    # Create a sample 3D image (Z, Y, X)
    logger.info("Creating sample 3D microscopy image...")
    z_slices, height, width = 20, 256, 256
    
    # Create image with background and signal
    image = np.zeros((z_slices, height, width), dtype=np.float32)
    
    # Add background gradient
    for z in range(z_slices):
        for y in range(height):
            for x in range(width):
                bg = 100 + 50 * (x / width) + 30 * (y / height) + 20 * (z / z_slices)
                image[z, y, x] = bg
    
    # Add some bright objects (simulating cells/structures)
    for _ in range(10):
        z = np.random.randint(2, z_slices - 2)
        y = np.random.randint(20, height - 20)
        x = np.random.randint(20, width - 20)
        radius = np.random.randint(5, 15)
        intensity = np.random.randint(200, 400)
        
        # Create sphere
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dz*dz + dy*dy + dx*dx <= radius*radius:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < z_slices and 0 <= ny < height and 0 <= nx < width:
                            image[nz, ny, nx] += intensity
    
    # Add noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, None)
    
    logger.info(f"Sample image created: shape={image.shape}, range=[{image.min():.1f}, {image.max():.1f}]")
    
    # Initialize background subtractor with CUDA acceleration
    logger.info("\nInitializing CUDA-accelerated background subtractor...")
    subtractor = BackgroundSubtractor(use_cuda=True)  # Auto-detect CUDA
    
    # Check GPU info
    gpu_info = subtractor.get_gpu_info()
    if gpu_info['gpu_accelerated']:
        logger.info(f"✓ GPU acceleration enabled: {gpu_info['gpu_name']}")
        logger.info(f"  Memory: {gpu_info['free_memory_gb']:.1f}GB free")
    else:
        logger.info(f"✗ GPU acceleration disabled: {gpu_info.get('reason', 'Unknown')}")
    
    # Test different background subtraction methods
    methods = ['rolling_ball', 'gaussian', 'morphological']
    
    for method in methods:
        logger.info(f"\n--- Testing {method} method ---")
        
        # Estimate speedup
        speedup_estimate = subtractor.estimate_speedup(image.shape, method)
        logger.info(f"Estimated speedup: {speedup_estimate['estimated_speedup']:.1f}x")
        
        # Apply background subtraction
        try:
            corrected_image, metadata = subtractor.subtract_background(
                image=image,
                method=method,
                channel_name="DAPI",  # Use DAPI parameters
                pixel_size=0.1  # 0.1 μm per pixel
            )
            
            logger.info(f"✓ {method} completed successfully")
            logger.info(f"  Method used: {metadata['method']}")
            logger.info(f"  GPU accelerated: {metadata.get('gpu_accelerated', False)}")
            logger.info(f"  Result range: [{corrected_image.min():.1f}, {corrected_image.max():.1f}]")
            
            # Show some statistics
            if 'background_stats' in metadata:
                stats = metadata['background_stats']
                logger.info(f"  Background mean: {stats.get('mean_across_slices', 'N/A'):.1f}")
                logger.info(f"  Background std: {stats.get('std_across_slices', 'N/A'):.1f}")
            
        except Exception as e:
            logger.error(f"✗ {method} failed: {e}")
    
    # Benchmark performance
    if gpu_info['gpu_accelerated']:
        logger.info("\n--- Performance Benchmark ---")
        try:
            benchmark_results = subtractor.benchmark_methods(image, "DAPI")
            
            for method in methods:
                if f'{method}_speedup' in benchmark_results:
                    speedup = benchmark_results[f'{method}_speedup']
                    gpu_time = benchmark_results[f'{method}_gpu']['time_seconds']
                    cpu_time = benchmark_results[f'{method}_cpu']['time_seconds']
                    
                    logger.info(f"{method}:")
                    logger.info(f"  CPU time: {cpu_time:.2f}s")
                    logger.info(f"  GPU time: {gpu_time:.2f}s")
                    logger.info(f"  Speedup: {speedup:.1f}x")
                    
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("Example completed successfully!")
    logger.info("="*50)

if __name__ == "__main__":
    main()
