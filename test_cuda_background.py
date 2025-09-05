#!/usr/bin/env python3
"""
Test script for CUDA-accelerated background subtraction.

This script demonstrates the performance improvements of CUDA acceleration
for background subtraction operations on 3D microscopy images.
"""

import numpy as np
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image(shape=(20, 512, 512), noise_level=0.1):
    """Create a synthetic 3D test image with background and signal."""
    logger.info(f"Creating test image with shape {shape}")
    
    # Create base image
    image = np.zeros(shape, dtype=np.float32)
    
    # Add background gradient
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                # Background gradient
                bg = 100 + 50 * (x / shape[2]) + 30 * (y / shape[1]) + 20 * (z / shape[0])
                image[z, y, x] = bg
    
    # Add some signal objects (spheres)
    n_objects = 20
    for _ in range(n_objects):
        # Random position
        z = np.random.randint(5, shape[0] - 5)
        y = np.random.randint(50, shape[1] - 50)
        x = np.random.randint(50, shape[2] - 50)
        radius = np.random.randint(10, 30)
        intensity = np.random.randint(200, 500)
        
        # Create sphere
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dz*dz + dy*dy + dx*dx <= radius*radius:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                            image[nz, ny, nx] += intensity
    
    # Add noise
    noise = np.random.normal(0, noise_level * image.max(), shape)
    image = image + noise
    
    # Ensure positive values
    image = np.clip(image, 0, None)
    
    logger.info(f"Test image created: min={image.min():.1f}, max={image.max():.1f}, mean={image.mean():.1f}")
    return image

def test_cuda_background_subtraction():
    """Test CUDA-accelerated background subtraction."""
    try:
        # Import the background subtractor
        from perinuclear_analysis.imaging_preprocessing.background_subtraction.background_subtractor import BackgroundSubtractor
        
        # Create test image
        test_image = create_test_image(shape=(15, 256, 256))  # Smaller for testing
        
        logger.info("=" * 60)
        logger.info("CUDA Background Subtraction Test")
        logger.info("=" * 60)
        
        # Test with CUDA (if available)
        logger.info("\n1. Testing with CUDA acceleration:")
        cuda_subtractor = BackgroundSubtractor(use_cuda=True)
        
        # Get GPU info
        gpu_info = cuda_subtractor.get_gpu_info()
        logger.info(f"GPU Info: {gpu_info}")
        
        if gpu_info['gpu_accelerated']:
            # Test different methods
            methods = ['rolling_ball', 'gaussian', 'morphological']
            
            for method in methods:
                logger.info(f"\nTesting {method} method:")
                
                # Estimate speedup
                speedup_estimate = cuda_subtractor.estimate_speedup(test_image.shape, method)
                logger.info(f"Estimated speedup: {speedup_estimate['estimated_speedup']:.1f}x")
                
                # Time the operation
                start_time = time.time()
                try:
                    result, metadata = cuda_subtractor.subtract_background(
                        test_image, 
                        method=method, 
                        channel_name="test"
                    )
                    cuda_time = time.time() - start_time
                    
                    logger.info(f"CUDA {method}: {cuda_time:.2f}s")
                    logger.info(f"Result shape: {result.shape}, dtype: {result.dtype}")
                    logger.info(f"GPU accelerated: {metadata.get('gpu_accelerated', False)}")
                    
                except Exception as e:
                    logger.error(f"CUDA {method} failed: {e}")
        
        # Test with CPU fallback
        logger.info("\n2. Testing with CPU implementation:")
        cpu_subtractor = BackgroundSubtractor(use_cuda=False)
        
        for method in methods:
            logger.info(f"\nTesting CPU {method} method:")
            
            start_time = time.time()
            try:
                result, metadata = cpu_subtractor.subtract_background(
                    test_image, 
                    method=method, 
                    channel_name="test"
                )
                cpu_time = time.time() - start_time
                
                logger.info(f"CPU {method}: {cpu_time:.2f}s")
                logger.info(f"Result shape: {result.shape}, dtype: {result.dtype}")
                
            except Exception as e:
                logger.error(f"CPU {method} failed: {e}")
        
        # Benchmark comparison
        if gpu_info['gpu_accelerated']:
            logger.info("\n3. Running benchmark comparison:")
            benchmark_results = cuda_subtractor.benchmark_methods(test_image, "test")
            
            for method in methods:
                if f'{method}_speedup' in benchmark_results:
                    speedup = benchmark_results[f'{method}_speedup']
                    logger.info(f"{method}: {speedup:.1f}x speedup")
        
        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"Failed to import background subtractor: {e}")
        logger.info("Make sure you're running from the correct directory")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_gpu_detection():
    """Test GPU detection and capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("GPU Detection Test")
    logger.info("=" * 60)
    
    try:
        import cupy as cp
        logger.info("✓ CuPy is available")
        
        # Get GPU info
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"GPU count: {gpu_count}")
        
        if gpu_count > 0:
            device = cp.cuda.Device(0)
            device.use()
            
            # Get memory info
            mem_info = device.mem_info
            total_mem = mem_info[1] / (1024**3)
            free_mem = mem_info[0] / (1024**3)
            
            logger.info(f"GPU 0 memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")
            
            # Get GPU properties
            props = cp.cuda.runtime.getDeviceProperties(0)
            logger.info(f"GPU name: {props['name'].decode()}")
            logger.info(f"Compute capability: {props['major']}.{props['minor']}")
            logger.info(f"Multiprocessors: {props['multiProcessorCount']}")
            
            # Test basic operations
            logger.info("\nTesting basic GPU operations:")
            test_array = cp.random.random((100, 100), dtype=cp.float32)
            result = cp.sum(test_array)
            logger.info(f"✓ Basic GPU operation successful: sum = {float(result):.2f}")
            
        else:
            logger.warning("No CUDA-capable GPUs found")
            
    except ImportError:
        logger.warning("✗ CuPy not available - CUDA acceleration disabled")
    except Exception as e:
        logger.error(f"GPU detection failed: {e}")

if __name__ == "__main__":
    logger.info("Starting CUDA Background Subtraction Tests")
    
    # Test GPU detection first
    test_gpu_detection()
    
    # Test background subtraction
    test_cuda_background_subtraction()
    
    logger.info("All tests completed!")
