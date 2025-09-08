"""
Background subtraction preprocessing for confocal microscopy images.

This module implements various background subtraction methods optimized for 3D ome.tiff
confocal z-stack images with different channel types. Follows the critical first step 
in the preprocessing pipeline: background subtraction → denoising → deconvolution.

Features:
- CUDA-only implementation (requires CuPy/CUDA)
- Memory-efficient processing with chunk-based operations (GPU-aware)
- Channel-specific parameter auto-selection
- Integration with existing ImageLoader workflow

Input: 3D numpy arrays (Z, Y, X) from ome.tiff files
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage import filters, morphology

from ...core.config import BackgroundSubtractionConfig, PreprocessingConfig
from ...core.utils import convert_microns_to_pixels
from ...data_processing.image_loader import ImageLoader

# Try to import CUDA libraries
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUDA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("CUDA libraries available - GPU acceleration enabled")
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    cp_ndimage = None
    logger = logging.getLogger(__name__)
    logger.error("CUDA libraries not available - this module requires CUDA (CuPy)")

logger = logging.getLogger(__name__)


class BackgroundSubtractor:
    """
    Background subtraction processor for 3D confocal z-stack ome.tiff images.
    
    Implements multiple background subtraction methods optimized for 3D data:
    - Rolling ball: Best for uneven illumination (slice-by-slice processing)
    - Gaussian: Fast approximation for gradual backgrounds
    - Morphological: For structured backgrounds
    
    Features:
    - CUDA acceleration for 10-50x speedup on compatible GPUs
    - Automatic CPU fallback when CUDA is not available
    - Memory-efficient processing with chunk-based operations
    - Channel-specific parameter auto-selection
    - Integration with existing ImageLoader workflow
    
    Input: 3D numpy arrays (Z, Y, X) from ome.tiff files
    """
    
    def __init__(self, config: Optional[BackgroundSubtractionConfig] = None, 
                 use_cuda: Optional[bool] = None):
        """
        Initialize the background subtractor.
        
        Args:
            config: Background subtraction configuration. If None, uses defaults.
            use_cuda: Force CUDA usage (True) or CPU (False). If None, auto-detect.
        """
        self.config = config or BackgroundSubtractionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Enforce CUDA-only usage
        self.use_cuda = True
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA libraries not available; BackgroundSubtractor requires CUDA (CuPy)")
        self.logger.info("Initializing CUDA-accelerated background subtractor (CUDA required)")
        self._initialize_cuda()
    
    def _initialize_cuda(self) -> None:
        """Initialize CUDA context and check GPU memory."""
        try:
            # Get GPU info
            mempool = cp.get_default_memory_pool()
            
            # Get GPU memory info
            gpu_mem = cp.cuda.Device().mem_info
            total_mem = gpu_mem[1] / (1024**3)  # Convert to GB
            free_mem = gpu_mem[0] / (1024**3)
            
            self.logger.info(f"GPU Memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")
            
            # Set memory pool limits (use 80% of available memory)
            max_mem_gb = free_mem * 0.8
            mempool.set_limit(size=int(max_mem_gb * 1024**3))
            
            self.gpu_memory_gb = free_mem
            self.max_gpu_memory_gb = max_mem_gb
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CUDA: {e}")
            raise RuntimeError(f"CUDA initialization failed: {e}")
        
    def subtract_background(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        channel_name: Optional[str] = None,
        pixel_size: Optional[float] = None,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Subtract background from a 3D ome.tiff z-stack image.
        
        Args:
            image: Input 3D image array (Z, Y, X) from ome.tiff
            method: Background subtraction method ('rolling_ball', 'gaussian', 'morphological')
                   If None, uses config default
            channel_name: Name of the channel for automatic parameter selection
            pixel_size: Pixel size in micrometers for micron-based parameters
            chunk_size: Number of z-slices to process at once for memory efficiency
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (background_corrected_image, metadata_dict)
        """
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D (Z, Y, X), got {image.ndim}D with shape {image.shape}")
        
        method = method or self.config.method
        
        self.logger.info(f"Applying CUDA {method} background subtraction to 3D image {image.shape}")
        return self._subtract_background_cuda(image, method, channel_name, pixel_size, **kwargs)
    
    def _subtract_background_cuda(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CUDA-accelerated background subtraction."""
        try:
            # Check if image fits in GPU memory
            image_memory_gb = image.nbytes / (1024**3)
            if image_memory_gb > self.max_gpu_memory_gb:
                self.logger.error(f"Image too large for GPU memory ({image_memory_gb:.1f}GB > {self.max_gpu_memory_gb:.1f}GB)")
                raise MemoryError("Input image exceeds available GPU memory. Consider reducing image size or parameters.")
            
            # Transfer image to GPU
            gpu_image = cp.asarray(image, dtype=cp.float32)
            
            # Get method-specific parameters
            params = self._get_method_parameters(method, channel_name, pixel_size, **kwargs)
            
            # Apply background subtraction
            if method == 'rolling_ball':
                corrected_image, metadata = self._rolling_ball_subtraction_3d_cuda(gpu_image, params)
            elif method == 'gaussian':
                corrected_image, metadata = self._gaussian_subtraction_3d_cuda(gpu_image, params)
            elif method == 'morphological':
                corrected_image, metadata = self._morphological_subtraction_3d_cuda(gpu_image, params)
            else:
                raise ValueError(f"Unknown background subtraction method: {method}")
            
            # Transfer result back to CPU
            result = cp.asnumpy(corrected_image)
            
            # Clean up GPU memory
            del gpu_image, corrected_image
            cp.get_default_memory_pool().free_all_blocks()
            
            # Post-processing
            if self.config.clip_negative_values:
                result = np.clip(result, 0, None)
                
            if self.config.normalize_output:
                result = self._normalize_image(result)
            
            # Update metadata
            metadata.update({
                'method': f'{method}_cuda',
                'original_shape': image.shape,
                'original_dtype': str(image.dtype),
                'clipped_negative': self.config.clip_negative_values,
                'normalized': self.config.normalize_output,
                'parameters_used': params,
                'gpu_accelerated': True,
                'gpu_memory_used_gb': image_memory_gb
            })
            
            return result, metadata
            
        except Exception as e:
            self.logger.error(f"CUDA processing failed: {e}")
            raise
    
    def _subtract_background_cpu(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        chunk_size: Optional[int],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CPU implementation of background subtraction."""
        # Determine chunk size for memory efficiency
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(image.shape, image.dtype)
        
        # Get method-specific parameters
        params = self._get_method_parameters(method, channel_name, pixel_size, **kwargs)
        
        # Apply background subtraction with chunked processing
        if method == 'rolling_ball':
            corrected_image, metadata = self._rolling_ball_subtraction_3d(image, params, chunk_size)
        elif method == 'gaussian':
            corrected_image, metadata = self._gaussian_subtraction_3d(image, params)
        elif method == 'morphological':
            corrected_image, metadata = self._morphological_subtraction_3d(image, params, chunk_size)
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")
        
        # Post-processing
        if self.config.clip_negative_values:
            corrected_image = np.clip(corrected_image, 0, None)
            
        if self.config.normalize_output:
            corrected_image = self._normalize_image(corrected_image)
        
        # Update metadata
        metadata.update({
            'method': method,
            'original_shape': image.shape,
            'original_dtype': str(image.dtype),
            'clipped_negative': self.config.clip_negative_values,
            'normalized': self.config.normalize_output,
            'parameters_used': params,
            'chunk_size_used': chunk_size,
            'memory_efficient': chunk_size < image.shape[0],
            'gpu_accelerated': False
        })
        
        return corrected_image, metadata
    
    def _get_method_parameters(
        self, 
        method: str, 
        channel_name: Optional[str], 
        pixel_size: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Get parameters for the specified method and channel."""
        params = {}
        
        if method == 'rolling_ball':
            # Get radius based on channel type
            if channel_name:
                radius = self._get_rolling_ball_radius_for_channel(channel_name)
            else:
                radius = kwargs.get('radius', 30)  # Default radius
            params['radius'] = radius
            
        elif method == 'gaussian':
            # Get sigma based on channel type
            if channel_name:
                sigma = self._get_gaussian_sigma_for_channel(channel_name)
            else:
                sigma = kwargs.get('sigma', 15.0)  # Default sigma
            params['sigma'] = sigma
            
        elif method == 'morphological':
            # Get structuring element size
            if pixel_size and 'size_um' in kwargs:
                size_pixels = convert_microns_to_pixels(kwargs['size_um'], pixel_size)
            else:
                size_pixels = kwargs.get('size_pixels', 20)
            params['size'] = int(size_pixels)
            params['shape'] = kwargs.get('shape', 'disk')
        
        # Override with any explicit kwargs
        params.update({k: v for k, v in kwargs.items() if k not in ['channel_name', 'pixel_size']})
        
        return params
    
    def _get_rolling_ball_radius_for_channel(self, channel_name: str) -> int:
        """Get appropriate rolling ball radius for channel type."""
        channel_lower = channel_name.lower()
        
        if any(name in channel_lower for name in ['dapi', 'hoechst']):
            return self.config.rolling_ball_radius_dapi
        elif any(name in channel_lower for name in ['phalloidin', 'actin', 'af488']):
            return self.config.rolling_ball_radius_phalloidin
        else:
            return self.config.rolling_ball_radius_protein
    
    def _get_gaussian_sigma_for_channel(self, channel_name: str) -> float:
        """Get appropriate Gaussian sigma for channel type."""
        channel_lower = channel_name.lower()
        
        if any(name in channel_lower for name in ['dapi', 'hoechst']):
            return self.config.gaussian_sigma_dapi
        elif any(name in channel_lower for name in ['phalloidin', 'actin', 'af488']):
            return self.config.gaussian_sigma_phalloidin
        else:
            return self.config.gaussian_sigma_protein
    
    def _calculate_optimal_chunk_size(self, image_shape: Tuple[int, int, int], dtype: np.dtype) -> int:
        """
        Calculate optimal chunk size for memory-efficient processing on M3 Pro.
        
        Args:
            image_shape: Shape of the 3D image (Z, Y, X)
            dtype: Data type of the image
            
        Returns:
            Optimal number of z-slices to process at once
        """
        z_slices, height, width = image_shape
        bytes_per_element = np.dtype(dtype).itemsize
        
        # Estimate memory per slice (including intermediate arrays)
        slice_memory_mb = (height * width * bytes_per_element * 3) / (1024 * 1024)  # 3x for intermediate processing
        
        # Use conservative 2GB limit for chunk processing (from 14GB total available)
        max_chunk_memory_mb = 2048
        
        optimal_chunk_size = max(1, min(z_slices, int(max_chunk_memory_mb / slice_memory_mb)))
        
        self.logger.info(f"Calculated chunk size: {optimal_chunk_size} slices "
                        f"({slice_memory_mb:.1f} MB per slice)")
        
        return optimal_chunk_size

    def _rolling_ball_subtraction_3d(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any],
        chunk_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform rolling ball background subtraction on 3D image with chunked processing.
        
        This method processes z-slices in chunks to manage memory usage on M3 Pro.
        """
        radius = params['radius']
        z_slices, height, width = image.shape
        
        # Pre-allocate output array
        corrected_image = np.empty_like(image, dtype=np.float32)
        
        # Track background statistics
        background_stats_list = []
        
        # Process in chunks
        for start_z in range(0, z_slices, chunk_size):
            end_z = min(start_z + chunk_size, z_slices)
            chunk_slice = slice(start_z, end_z)
            
            self.logger.debug(f"Processing z-slices {start_z} to {end_z-1}")
            
            # Process each slice in the chunk
            for z in range(start_z, end_z):
                light_background = params.get('light_background', False)
                background = self._create_rolling_ball_background(image[z], radius, light_background)
                corrected_image[z] = image[z].astype(np.float32) - background
                
                # Collect background statistics
                background_stats_list.append({
                    'z_slice': z,
                    'mean': float(np.mean(background)),
                    'std': float(np.std(background))
                })
            
            # Force garbage collection after processing chunk
            gc.collect()
        
        # Aggregate background statistics
        all_means = [stats['mean'] for stats in background_stats_list]
        all_stds = [stats['std'] for stats in background_stats_list]
        
        metadata = {
            'background_method': 'rolling_ball_3d',
            'radius_pixels': radius,
            'z_slices_processed': z_slices,
            'chunk_size': chunk_size,
            'background_stats': {
                'mean_across_slices': float(np.mean(all_means)),
                'std_across_slices': float(np.mean(all_stds)),
                'mean_variation': float(np.std(all_means)),
                'per_slice_stats': background_stats_list[:5]  # Keep first 5 for debugging
            }
        }
        
        return corrected_image, metadata
    
    def _create_rolling_ball_background(self, image: np.ndarray, radius: int, light_background: bool = False) -> np.ndarray:
        """
        Create rolling ball background using proper rolling ball algorithm.
        
        Uses the opencv-rolling-ball package which is ported from ImageJ's Background Subtractor.
        This is the most accurate implementation of the rolling ball algorithm.
        """
        import time
        start_time = time.time()
        
        # Check if image is too large and suggest downsampling
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            self.logger.warning(f"Large image detected ({image.shape}). Rolling ball may be slow. Consider using smaller radius or downsampling.")
        
        try:
            from cv2_rolling_ball import subtract_background_rolling_ball
        except ImportError:
            # Fallback to scikit-image if opencv-rolling-ball is not available
            from skimage import restoration
            self.logger.warning("opencv-rolling-ball not available, falling back to scikit-image")
            
            if image.dtype != np.float64:
                image = image.astype(np.float64)
            background = restoration.rolling_ball(image, radius=radius, light_background=light_background)
            return background.astype(np.float32)
        
        # Convert to 8-bit if needed (opencv-rolling-ball requirement)
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image_8bit = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_8bit = np.zeros_like(image, dtype=np.uint8)
        else:
            image_8bit = image
        
        self.logger.info(f"Starting rolling ball background subtraction (radius={radius}, size={image.shape})")
        
        # Use opencv-rolling-ball implementation
        corrected_img, background = subtract_background_rolling_ball(
            image_8bit, 
            radius, 
            light_background=light_background,
            use_paraboloid=False,  # Use rolling ball, not paraboloid
            do_presmooth=True
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Rolling ball completed in {elapsed_time:.2f}s")
        
        # Convert background back to original data type and range
        if image.dtype != np.uint8:
            background = background.astype(np.float32) / 255.0 * (img_max - img_min) + img_min
        
        return background.astype(np.float32)
    
    def _gaussian_subtraction_3d(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform 3D Gaussian background subtraction.
        
        This method is faster than rolling ball and works well for gradual
        background variations. Applies Gaussian filtering across all three dimensions.
        """
        sigma = params['sigma']
        
        self.logger.info(f"Applying 3D Gaussian background subtraction with sigma={sigma}")
        
        # Create 3D Gaussian background
        background = filters.gaussian(image, sigma=sigma, preserve_range=True)
        
        # Subtract background
        corrected_image = image.astype(np.float32) - background.astype(np.float32)
        
        metadata = {
            'background_method': 'gaussian_3d',
            'sigma': sigma,
            'background_stats': {
                'mean': float(np.mean(background)),
                'std': float(np.std(background)),
                'min': float(np.min(background)),
                'max': float(np.max(background))
            }
        }
        
        # Clean up memory
        del background
        gc.collect()
        
        return corrected_image, metadata
    
    def _morphological_subtraction_3d(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any],
        chunk_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform morphological background subtraction on 3D image with chunked processing.
        
        Uses morphological closing to estimate background, useful for
        images with structured backgrounds. Processes slice-by-slice for memory efficiency.
        """
        size = params['size']
        shape = params.get('shape', 'disk')
        z_slices = image.shape[0]
        
        # Create structuring element for 2D operations on each slice
        if shape == 'disk':
            selem = morphology.disk(size)
        elif shape == 'square':
            selem = morphology.square(size)
        else:
            selem = morphology.disk(size)  # Default to disk
        
        # Pre-allocate output array
        corrected_image = np.empty_like(image, dtype=np.float32)
        
        # Track background statistics
        background_stats_list = []
        
        # Process in chunks
        for start_z in range(0, z_slices, chunk_size):
            end_z = min(start_z + chunk_size, z_slices)
            
            self.logger.debug(f"Processing morphological background for z-slices {start_z} to {end_z-1}")
            
            # Process each slice in the chunk
            for z in range(start_z, end_z):
                # Morphological closing to estimate background
                background = morphology.closing(image[z], selem)
                corrected_image[z] = image[z].astype(np.float32) - background
                
                # Collect background statistics
                background_stats_list.append({
                    'z_slice': z,
                    'mean': float(np.mean(background)),
                    'std': float(np.std(background))
                })
            
            # Force garbage collection after processing chunk
            gc.collect()
        
        # Aggregate background statistics
        all_means = [stats['mean'] for stats in background_stats_list]
        all_stds = [stats['std'] for stats in background_stats_list]
        
        metadata = {
            'background_method': 'morphological_3d',
            'size': size,
            'shape': shape,
            'z_slices_processed': z_slices,
            'chunk_size': chunk_size,
            'background_stats': {
                'mean_across_slices': float(np.mean(all_means)),
                'std_across_slices': float(np.mean(all_stds)),
                'mean_variation': float(np.std(all_means)),
                'per_slice_stats': background_stats_list[:5]  # Keep first 5 for debugging
            }
        }
        
        return corrected_image, metadata
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range while preserving relative intensities."""
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            normalized = (image - image_min) / (image_max - image_min)
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    # ==================== CUDA-ACCELERATED METHODS ====================
    
    def _rolling_ball_subtraction_3d_cuda(
        self, 
        gpu_image: cp.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """CUDA-accelerated rolling ball background subtraction (morphological approximation).

        Approximates ImageJ-style rolling-ball using greyscale morphology with a disk footprint.
        Uses opening for dark background (typical) and closing if light_background=True.
        """
        radius = params['radius']
        z_slices, height, width = gpu_image.shape
        
        self.logger.info(f"Processing {z_slices} z-slices with CUDA rolling ball approximation (radius={radius})")
        
        # Create structuring element (disk) once on GPU
        selem = self._create_disk_selem_cuda(radius)
        
        # Pre-allocate output array on GPU
        corrected_image = cp.empty_like(gpu_image, dtype=cp.float32)
        
        background_means: List[float] = []
        background_stds: List[float] = []
        light_background = params.get('light_background', False)
        
        # Process slices
        for z in range(z_slices):
            if light_background:
                background = cp_ndimage.grey_closing(gpu_image[z], footprint=selem)
            else:
                background = cp_ndimage.grey_opening(gpu_image[z], footprint=selem)
            corrected_image[z] = gpu_image[z].astype(cp.float32) - background.astype(cp.float32)
            background_means.append(float(cp.mean(background)))
            background_stds.append(float(cp.std(background)))
        
        metadata = {
            'background_method': 'rolling_ball_3d_cuda',
            'radius_pixels': radius,
            'z_slices_processed': z_slices,
            'background_stats': {
                'mean_across_slices': float(np.mean(background_means)),
                'std_across_slices': float(np.mean(background_stds)),
                'gpu_processing': True
            }
        }
        
        return corrected_image, metadata
    
    def _create_rolling_ball_background_cuda(self, gpu_slice: cp.ndarray, radius: int, light_background: bool = False) -> cp.ndarray:
        """Create rolling ball background on GPU using proper rolling ball algorithm."""
        # For CUDA, we need to fall back to CPU implementation since scikit-image's rolling_ball
        # doesn't have a GPU version. Transfer to CPU, process, and transfer back.
        
        # Transfer to CPU
        cpu_slice = cp.asnumpy(gpu_slice)
        
        # Use the corrected CPU implementation
        cpu_background = self._create_rolling_ball_background(cpu_slice, radius, light_background)
        
        # Transfer back to GPU
        gpu_background = cp.asarray(cpu_background, dtype=cp.float32)
        
        return gpu_background
    
    def _create_disk_selem_cuda(self, radius: int) -> cp.ndarray:
        """Create disk-shaped structuring element on GPU."""
        # Create disk on CPU first (small operation)
        cpu_disk = morphology.disk(radius)
        # Transfer to GPU
        return cp.asarray(cpu_disk, dtype=cp.uint8)
    
    def _gaussian_subtraction_3d_cuda(
        self, 
        gpu_image: cp.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """CUDA-accelerated 3D Gaussian background subtraction."""
        sigma = params['sigma']
        
        self.logger.info(f"Applying CUDA 3D Gaussian background subtraction with sigma={sigma}")
        
        # Create 3D Gaussian background on GPU
        background = cp_ndimage.gaussian_filter(gpu_image, sigma=sigma)
        
        # Subtract background
        corrected_image = gpu_image.astype(cp.float32) - background.astype(cp.float32)
        
        # Calculate statistics on GPU
        bg_mean = float(cp.mean(background))
        bg_std = float(cp.std(background))
        bg_min = float(cp.min(background))
        bg_max = float(cp.max(background))
        
        metadata = {
            'background_method': 'gaussian_3d_cuda',
            'sigma': sigma,
            'background_stats': {
                'mean': bg_mean,
                'std': bg_std,
                'min': bg_min,
                'max': bg_max,
                'gpu_processing': True
            }
        }
        
        # Clean up GPU memory
        del background
        cp.get_default_memory_pool().free_all_blocks()
        
        return corrected_image, metadata
    
    def _morphological_subtraction_3d_cuda(
        self, 
        gpu_image: cp.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """CUDA-accelerated morphological background subtraction."""
        size = params['size']
        shape = params.get('shape', 'disk')
        z_slices = gpu_image.shape[0]
        
        # Create structuring element on GPU
        if shape == 'disk':
            selem = self._create_disk_selem_cuda(size)
        elif shape == 'square':
            cpu_square = morphology.square(size)
            selem = cp.asarray(cpu_square, dtype=cp.uint8)
        else:
            selem = self._create_disk_selem_cuda(size)
        
        # Pre-allocate output array on GPU
        corrected_image = cp.empty_like(gpu_image, dtype=cp.float32)
        
        # Process all slices on GPU using greyscale morphology
        background_means: List[float] = []
        background_stds: List[float] = []
        for z in range(z_slices):
            background = cp_ndimage.grey_closing(gpu_image[z], footprint=selem)
            corrected_image[z] = gpu_image[z].astype(cp.float32) - background.astype(cp.float32)
            background_means.append(float(cp.mean(background)))
            background_stds.append(float(cp.std(background)))
        
        metadata = {
            'background_method': 'morphological_3d_cuda',
            'size': size,
            'shape': shape,
            'z_slices_processed': z_slices,
            'background_stats': {
                'mean_across_slices': float(np.mean(background_means)),
                'std_across_slices': float(np.mean(background_stds)),
                'gpu_processing': True
            }
        }
        
        return corrected_image, metadata
    
    def process_from_loader(
        self,
        image_loader: ImageLoader,
        channel_index: Optional[int] = None,
        channel_name: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process image directly from ImageLoader instance.
        
        Args:
            image_loader: Initialized ImageLoader instance
            channel_index: Index of channel to process
            channel_name: Name of channel to process (alternative to index)
            method: Background subtraction method override
            **kwargs: Additional method parameters
            
        Returns:
            Tuple of (background_corrected_image, metadata_dict)
        """
        if not hasattr(image_loader, 'image_data') or image_loader.image_data is None:
            raise ValueError("ImageLoader must be loaded with image data")
        
        # Get the image data
        if channel_index is not None:
            if len(image_loader.image_data.shape) < 4:
                raise ValueError("Channel index specified but image is not multi-channel")
            image = image_loader.image_data[:, :, :, channel_index]
        elif channel_name is not None:
            if not hasattr(image_loader, 'metadata') or 'channels' not in image_loader.metadata:
                raise ValueError("Channel name specified but no channel metadata available")
            
            channels = image_loader.metadata['channels']
            try:
                channel_idx = channels.index(channel_name)
                image = image_loader.image_data[:, :, :, channel_idx]
            except ValueError:
                raise ValueError(f"Channel '{channel_name}' not found in {channels}")
        else:
            # Use entire image
            image = image_loader.image_data
        
        # Get pixel size from metadata if available
        pixel_size = None
        if hasattr(image_loader, 'metadata') and 'pixel_size' in image_loader.metadata:
            pixel_size = image_loader.metadata['pixel_size']
        
        # Process the image
        return self.subtract_background(
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            **kwargs
        )
    
    def batch_process(
        self,
        images: Dict[str, np.ndarray],
        channel_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_size: Optional[float] = None
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process multiple images/channels in batch.
        
        Args:
            images: Dictionary mapping channel names to image arrays
            channel_configs: Optional per-channel configuration overrides
            pixel_size: Pixel size in micrometers
            
        Returns:
            Dictionary mapping channel names to (corrected_image, metadata) tuples
        """
        results = {}
        channel_configs = channel_configs or {}
        
        for channel_name, image in images.items():
            self.logger.info(f"Processing channel: {channel_name}")
            
            # Get channel-specific config
            channel_config = channel_configs.get(channel_name, {})
            
            # Process the image
            corrected_image, metadata = self.subtract_background(
                image=image,
                channel_name=channel_name,
                pixel_size=pixel_size,
                **channel_config
            )
            
            results[channel_name] = (corrected_image, metadata)
        
        return results
    
    @classmethod
    def from_preprocessing_config(cls, preprocessing_config: PreprocessingConfig) -> 'BackgroundSubtractor':
        """
        Create BackgroundSubtractor from a PreprocessingConfig.
        
        Args:
            preprocessing_config: Full preprocessing configuration
            
        Returns:
            BackgroundSubtractor instance
        """
        return cls(config=preprocessing_config.background_subtraction)
    
    def get_recommended_parameters(self, channel_name: str) -> Dict[str, Any]:
        """
        Get recommended parameters for a specific channel type.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Dictionary of recommended parameters
        """
        channel_lower = channel_name.lower()
        
        if any(name in channel_lower for name in ['dapi', 'hoechst']):
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_dapi,
                'description': 'DAPI/nuclear staining - rolling ball with large radius for nuclear regions'
            }
        elif any(name in channel_lower for name in ['phalloidin', 'actin']):
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_phalloidin,
                'description': 'Phalloidin/actin - smaller radius to preserve filament structures'
            }
        elif any(name in channel_lower for name in ['lamp1', 'lysosome']):
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_protein,
                'description': 'LAMP1 punctate structures - medium radius for vesicle preservation'
            }
        else:
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_protein,
                'description': 'Protein marker - standard radius for punctate/diffuse patterns'
            }
    
    def plot_background_subtraction_comparison(self, 
                                             original_data: np.ndarray,
                                             corrected_results: Dict[str, Tuple[np.ndarray, Dict]],
                                             channel_names: List[str],
                                             z_slice: Optional[int] = None,
                                             figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Create a detailed comparison plot showing original vs background-subtracted images.
        
        This method generates a comprehensive visualization with:
        - Original and corrected images side by side
        - Intensity histograms for both original and corrected data
        - Statistics and method information for each channel
        
        Args:
            original_data: Original 3D image data (Z, Y, X, C) from ImageLoader
            corrected_results: Dictionary mapping channel names to (corrected_image, metadata) tuples.
                              Can come from batch_process(), individual subtract_background() calls,
                              or any custom processing results.
            channel_names: List of channel names corresponding to the data
            z_slice: Z-slice index to display. If None, uses middle slice
            figsize: Figure size tuple (width, height). If None, auto-calculates
            
        Returns:
            Figure: Matplotlib figure object for further customization
            
        Examples:
            # From batch processing:
            >>> results = bg_subtractor.batch_process(images_dict, pixel_size)
            >>> fig = bg_subtractor.plot_background_subtraction_comparison(
            ...     loaded_data, results, channel_names
            ... )
            
            # From individual processing:
            >>> corrected_img, metadata = bg_subtractor.subtract_background(
            ...     channel_data, channel_name="DAPI", pixel_size=pixel_size
            ... )
            >>> results = {"DAPI": (corrected_img, metadata)}
            >>> fig = bg_subtractor.plot_background_subtraction_comparison(
            ...     loaded_data, results, ["DAPI"]
            ... )
            
            # Compare different parameter settings:
            >>> results1 = {"DAPI": bg_subtractor.subtract_background(data, radius=50, ...)}
            >>> results2 = {"DAPI": bg_subtractor.subtract_background(data, radius=100, ...)}
            >>> # Plot each separately to compare
        """
        if z_slice is None:
            z_slice = original_data.shape[0] // 2
        
        n_channels = len(channel_names)
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (5 * n_channels, 12)
        
        # Create subplots: 3 rows (original, corrected, histogram) x n_channels
        fig, axes = plt.subplots(3, n_channels, figsize=figsize)
        
        # Handle single channel case
        if n_channels == 1:
            axes = axes.reshape(-1, 1)
        
        for i, channel_name in enumerate(channel_names):
            # Get data for this channel
            original_channel = original_data[:, :, :, i]
            corrected_img, metadata = corrected_results[channel_name]
            
            # Calculate display ranges using percentiles for better visualization
            orig_p1, orig_p99 = np.percentile(original_channel, [1, 99])
            corr_p1, corr_p99 = np.percentile(corrected_img, [1, 99])
            
            # Row 1: Original images
            im_orig = axes[0, i].imshow(original_channel[z_slice], 
                                       cmap='viridis', 
                                       vmin=orig_p1, vmax=orig_p99)
            axes[0, i].set_title(f'Original {channel_name}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            plt.colorbar(im_orig, ax=axes[0, i], shrink=0.8, label='Intensity')
            
            # Row 2: Corrected images
            im_corr = axes[1, i].imshow(corrected_img[z_slice], 
                                       cmap='viridis', 
                                       vmin=corr_p1, vmax=corr_p99)
            
            # Create informative title with method and parameters
            method = metadata['method']
            params = metadata.get('parameters_used', {})
            if method == 'rolling_ball':
                radius = params.get('radius', 'N/A')
                title = f'Corrected {channel_name}\n{method} (radius={radius})'
            elif method == 'gaussian':
                sigma = params.get('sigma', 'N/A')
                title = f'Corrected {channel_name}\n{method} (σ={sigma})'
            else:
                title = f'Corrected {channel_name}\n{method}'
            
            axes[1, i].set_title(title, fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
            plt.colorbar(im_corr, ax=axes[1, i], shrink=0.8, label='Intensity')
            
            # Row 3: Histograms
            # Sample data for histogram (every 10th pixel to speed up plotting)
            orig_sample = original_channel.flatten()[::10]
            corr_sample = corrected_img.flatten()[::10]
            
            # Calculate statistics for display
            orig_mean = np.mean(orig_sample)
            orig_std = np.std(orig_sample)
            corr_mean = np.mean(corr_sample)
            corr_std = np.std(corr_sample)
            
            axes[2, i].hist(orig_sample, bins=50, alpha=0.7, label='Original', 
                           density=True, color='blue', edgecolor='black', linewidth=0.5)
            axes[2, i].hist(corr_sample, bins=50, alpha=0.7, label='Corrected', 
                           density=True, color='red', edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            stats_text = (f'Original: {orig_mean:.1f} ± {orig_std:.1f}\n'
                         f'Corrected: {corr_mean:.1f} ± {corr_std:.1f}')
            axes[2, i].text(0.02, 0.98, stats_text, transform=axes[2, i].transAxes,
                           verticalalignment='top', fontsize=9, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[2, i].set_title(f'{channel_name} Intensity Distribution', fontsize=11)
            axes[2, i].set_yscale('log')
            axes[2, i].legend(fontsize=10)
            axes[2, i].set_xlabel('Intensity', fontsize=10)
            axes[2, i].set_ylabel('Density (log)', fontsize=10)
            axes[2, i].grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Background Subtraction Analysis - Z-slice {z_slice}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        
        return fig
    
    def compare_parameter_settings(self,
                                 original_data: np.ndarray,
                                 channel_name: str,
                                 channel_index: int,
                                 parameter_sets: List[Dict[str, Any]],
                                 pixel_size: float,
                                 z_slice: Optional[int] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Compare different parameter settings for background subtraction on a single channel.
        
        This is useful for parameter optimization - you can test different radius values,
        methods, or other parameters and see their effects side by side.
        
        Args:
            original_data: Original 3D image data (Z, Y, X, C) from ImageLoader
            channel_name: Name of the channel to process
            channel_index: Index of the channel in the original data
            parameter_sets: List of parameter dictionaries to test
            pixel_size: Pixel size in micrometers
            z_slice: Z-slice index to display. If None, uses middle slice
            figsize: Figure size tuple (width, height). If None, auto-calculates
            
        Returns:
            Figure: Matplotlib figure object showing comparison of all parameter sets
            
        Example:
            >>> parameter_sets = [
            ...     {'method': 'rolling_ball', 'radius': 50},
            ...     {'method': 'rolling_ball', 'radius': 100},
            ...     {'method': 'rolling_ball', 'radius': 150},
            ...     {'method': 'gaussian', 'sigma': 2.0}
            ... ]
            >>> fig = bg_subtractor.compare_parameter_settings(
            ...     loaded_data, "DAPI", 0, parameter_sets, pixel_size
            ... )
        """
        if z_slice is None:
            z_slice = original_data.shape[0] // 2
        
        n_params = len(parameter_sets)
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (5 * n_params, 10)
        
        # Create subplots: 2 rows (original, corrected) x n_params
        fig, axes = plt.subplots(2, n_params, figsize=figsize)
        
        # Handle single parameter case
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        # Get original channel data
        original_channel = original_data[:, :, :, channel_index]
        orig_p1, orig_p99 = np.percentile(original_channel, [1, 99])
        
        # Process each parameter set
        for i, params in enumerate(parameter_sets):
            # Process with these parameters
            corrected_img, metadata = self.subtract_background(
                image=original_channel,
                channel_name=channel_name,
                pixel_size=pixel_size,
                **params
            )
            
            # Calculate display range for corrected image
            corr_p1, corr_p99 = np.percentile(corrected_img, [1, 99])
            
            # Row 1: Original image (show only once, in first column)
            if i == 0:
                im_orig = axes[0, i].imshow(original_channel[z_slice], 
                                           cmap='viridis', 
                                           vmin=orig_p1, vmax=orig_p99)
                axes[0, i].set_title(f'Original {channel_name}', fontsize=12, fontweight='bold')
                axes[0, i].axis('off')
                plt.colorbar(im_orig, ax=axes[0, i], shrink=0.8, label='Intensity')
            else:
                axes[0, i].axis('off')  # Hide other original images
            
            # Row 2: Corrected images
            im_corr = axes[1, i].imshow(corrected_img[z_slice], 
                                       cmap='viridis', 
                                       vmin=corr_p1, vmax=corr_p99)
            
            # Create informative title with parameters
            method = metadata['method']
            params_used = metadata.get('parameters_used', {})
            if method == 'rolling_ball':
                radius = params_used.get('radius', 'N/A')
                title = f'{method}\nradius={radius}'
            elif method == 'gaussian':
                sigma = params_used.get('sigma', 'N/A')
                title = f'{method}\nσ={sigma}'
            else:
                title = f'{method}'
            
            axes[1, i].set_title(title, fontsize=11, fontweight='bold')
            axes[1, i].axis('off')
            plt.colorbar(im_corr, ax=axes[1, i], shrink=0.8, label='Intensity')
        
        # Add overall title
        fig.suptitle(f'Parameter Comparison - {channel_name} (Z-slice {z_slice})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return fig
    
    # ==================== GPU UTILITY METHODS ====================
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about GPU availability and memory."""
        if not self.use_cuda:
            return {
                'cuda_available': False,
                'gpu_accelerated': False,
                'reason': 'CUDA libraries not available or disabled'
            }
        
        try:
            gpu_mem = cp.cuda.Device().mem_info
            total_mem = gpu_mem[1] / (1024**3)
            free_mem = gpu_mem[0] / (1024**3)
            
            return {
                'cuda_available': True,
                'gpu_accelerated': True,
                'gpu_name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                'total_memory_gb': total_mem,
                'free_memory_gb': free_mem,
                'max_usable_memory_gb': self.max_gpu_memory_gb,
                'cupy_version': cp.__version__
            }
        except Exception as e:
            return {
                'cuda_available': True,
                'gpu_accelerated': False,
                'error': str(e)
            }
    
    def benchmark_methods(self, image: np.ndarray, channel_name: str = "test") -> Dict[str, Any]:
        """Benchmark GPU performance for different methods (CUDA-only)."""
        import time
        
        results = {}
        methods = ['rolling_ball', 'gaussian', 'morphological']
        
        for method in methods:
            self.logger.info(f"Benchmarking {method} method...")
            
            # Test GPU performance only
            start_time = time.time()
            try:
                gpu_result, gpu_metadata = self._subtract_background_cuda(
                    image, method, channel_name, None
                )
                # Ensure timing includes GPU work
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start_time
                results[f'{method}_gpu'] = {
                    'time_seconds': gpu_time,
                    'success': True,
                    'metadata': gpu_metadata
                }
            except Exception as e:
                results[f'{method}_gpu'] = {
                    'time_seconds': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    