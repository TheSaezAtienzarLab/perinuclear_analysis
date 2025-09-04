"""
Background subtraction preprocessing for confocal microscopy images.

This module implements various background subtraction methods optimized for 3D ome.tiff
confocal z-stack images with different channel types. Follows the critical first step 
in the preprocessing pipeline: background subtraction → denoising → deconvolution.

Optimized for M3 Pro MacBook (18GB memory) with memory-efficient chunk processing.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import gc

import numpy as np
from skimage import filters, morphology

from ...core.config import BackgroundSubtractionConfig, PreprocessingConfig
from ...core.utils import convert_microns_to_pixels
from ...data_processing.image_loader import ImageLoader

logger = logging.getLogger(__name__)


class BackgroundSubtractor:
    """
    Background subtraction processor for 3D confocal z-stack ome.tiff images.
    
    Implements multiple background subtraction methods optimized for 3D data:
    - Rolling ball: Best for uneven illumination (slice-by-slice processing)
    - Gaussian: Fast approximation for gradual backgrounds
    - Morphological: For structured backgrounds
    
    Designed for memory-efficient processing on M3 Pro (18GB) with:
    - Chunk-based z-stack processing to manage memory usage
    - Channel-specific parameter auto-selection
    - Integration with existing ImageLoader workflow
    
    Input: 3D numpy arrays (Z, Y, X) from ome.tiff files
    """
    
    def __init__(self, config: Optional[BackgroundSubtractionConfig] = None):
        """
        Initialize the background subtractor.
        
        Args:
            config: Background subtraction configuration. If None, uses defaults.
        """
        self.config = config or BackgroundSubtractionConfig()
        self.logger = logging.getLogger(__name__)
        
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
        
        self.logger.info(f"Applying {method} background subtraction to 3D image {image.shape}")
        
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
            'memory_efficient': chunk_size < image.shape[0]
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
                background = self._create_rolling_ball_background(image[z], radius)
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
    
    def _create_rolling_ball_background(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Create rolling ball background using morphological operations.
        
        This implementation uses morphological opening with a disk-shaped
        structuring element, which approximates the rolling ball algorithm.
        """
        # Create disk-shaped structuring element
        selem = morphology.disk(radius)
        
        # Morphological opening (erosion followed by dilation)
        background = morphology.opening(image, selem)
        
        # Smooth the background to reduce artifacts
        background = filters.gaussian(background, sigma=radius/10)
        
        return background
    
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