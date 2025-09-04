"""
Phase 2: Maximum Intensity Projection (MIP) creation for z-stack images.
Supports different projection methods and multi-channel data.
"""

from typing import Optional, Tuple, Union, Dict, Any, List
from pathlib import Path
import logging
import warnings
import json

import numpy as np
import tifffile

from ..core.config import Phase2Config
from .image_loader import ImageLoader
from ..core.utils import ensure_directory


logger = logging.getLogger(__name__)


class MIPCreator:
    """Create Maximum Intensity Projections from z-stack images."""
    
    def __init__(self, config: Optional[Phase2Config] = None):
        """Initialize the MIP creator.
        
        Args:
            config: Phase 2 configuration. If None, uses defaults.
        """
        self.config = config or Phase2Config()
        self.quality_metrics: Dict[str, Any] = {}
    
    def create_mip(self, 
                   image: np.ndarray,
                   method: Optional[str] = None,
                   z_range: Optional[Tuple[int, int]] = None,
                   channel: Optional[Union[int, str]] = None) -> np.ndarray:
        """Create a Maximum Intensity Projection from a z-stack.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            method: Projection method ('max', 'mean', 'sum', 'median', 'std').
                   If None, uses config default.
            z_range: Optional range of z-slices to use (start, end).
                    If None, uses all slices or config default.
            channel: Optional specific channel to project. If None, projects all.
            
        Returns:
            np.ndarray: Projected image (Y, X, C) or (Y, X).
            
        Raises:
            ValueError: If image dimensions are invalid or method unknown.
        """
        # Validate input
        if image.ndim not in [3, 4]:
            raise ValueError(f"Image must be 3D or 4D z-stack, got {image.ndim}D")
        
        if image.ndim == 3:
            # Add channel dimension if missing
            image = image[..., np.newaxis]
            single_channel = True
        else:
            single_channel = False
        
        # Use config defaults if not specified
        method = method or self.config.projection_method
        z_range = z_range or self.config.z_range
        
        # Apply z-range if specified
        if z_range is not None:
            start, end = z_range
            if start < 0 or end > image.shape[0]:
                raise ValueError(
                    f"Invalid z_range {z_range}. Valid range: (0, {image.shape[0]})"
                )
            image = image[start:end]
            logger.info(f"Using z-slices {start} to {end}")
        
        # Extract specific channel if requested
        if channel is not None:
            if isinstance(channel, str):
                raise ValueError("Channel names require ImageLoader context")
            if channel < 0 or channel >= image.shape[-1]:
                raise ValueError(f"Invalid channel {channel}. Valid range: 0-{image.shape[-1]-1}")
            image = image[..., channel:channel+1]
        
        # Perform projection
        logger.info(f"Creating {method} projection from shape {image.shape}")
        
        if method == 'max':
            projection = np.max(image, axis=0)
        elif method == 'mean':
            projection = np.mean(image, axis=0)
        elif method == 'sum':
            projection = np.sum(image, axis=0)
        elif method == 'median':
            projection = np.median(image, axis=0)
        elif method == 'std':
            projection = np.std(image, axis=0)
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
        # Remove channel dimension if single channel
        if single_channel and projection.shape[-1] == 1:
            projection = projection[..., 0]
        
        # Calculate quality metrics if requested
        if self.config.calculate_quality_metrics:
            self._calculate_quality_metrics(image, projection, method)
        
        logger.info(f"Created {method} projection with shape {projection.shape}")
        
        return projection
    
    def create_color_mip(self,
                         image: np.ndarray,
                         channel_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                         method: str = 'max') -> np.ndarray:
        """Create a color MIP with specified colors for each channel.
        
        Args:
            image: Input z-stack image (Z, Y, X, C).
            channel_colors: Dict mapping channel index to RGB color (0-1).
                          If None, uses default colors.
            method: Projection method.
            
        Returns:
            np.ndarray: Color MIP image (Y, X, 3) in RGB format.
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        # Create MIP for each channel
        mip = self.create_mip(image, method=method)
        
        if mip.ndim == 2:
            mip = mip[..., np.newaxis]
        
        # Setup colors
        if channel_colors is None:
            # Default colors: blue, green, red, cyan, magenta, yellow
            default_colors = [
                (0, 0, 1),  # Blue
                (0, 1, 0),  # Green
                (1, 0, 0),  # Red
                (0, 1, 1),  # Cyan
                (1, 0, 1),  # Magenta
                (1, 1, 0),  # Yellow
            ]
            channel_colors = {i: default_colors[i % len(default_colors)] 
                            for i in range(mip.shape[-1])}
        
        # Create color image
        height, width, n_channels = mip.shape
        color_image = np.zeros((height, width, 3), dtype=np.float32)
        
        for c in range(n_channels):
            if c in channel_colors:
                color = channel_colors[c]
            else:
                # Default to grayscale for unmapped channels
                color = (1, 1, 1)
            
            # Normalize channel
            channel_data = mip[..., c].astype(np.float32)
            if channel_data.max() > 0:
                channel_data = channel_data / channel_data.max()
            
            # Apply color
            for i, color_val in enumerate(color):
                color_image[..., i] += channel_data * color_val
        
        # Clip to valid range
        color_image = np.clip(color_image, 0, 1)
        
        return color_image
    
    def create_multi_method_mip(self,
                                image: np.ndarray,
                                methods: List[str] = ['max', 'mean', 'std']) -> Dict[str, np.ndarray]:
        """Create multiple projections using different methods.
        
        Args:
            image: Input z-stack image.
            methods: List of projection methods to apply.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping method names to projections.
        """
        projections = {}
        
        for method in methods:
            try:
                projections[method] = self.create_mip(image, method=method)
            except Exception as e:
                logger.warning(f"Failed to create {method} projection: {e}")
                continue
        
        return projections
    
    def _calculate_quality_metrics(self,
                                   z_stack: np.ndarray,
                                   projection: np.ndarray,
                                   method: str) -> None:
        """Calculate quality metrics for the projection.
        
        Args:
            z_stack: Original z-stack.
            projection: Created projection.
            method: Projection method used.
        """
        metrics = {}
        
        # Basic statistics
        metrics['method'] = method
        metrics['z_slices'] = z_stack.shape[0]
        metrics['shape'] = projection.shape
        
        # Intensity statistics
        metrics['min_intensity'] = float(projection.min())
        metrics['max_intensity'] = float(projection.max())
        metrics['mean_intensity'] = float(projection.mean())
        metrics['std_intensity'] = float(projection.std())
        
        # Dynamic range
        if projection.max() > projection.min():
            metrics['dynamic_range'] = float(
                (projection.max() - projection.min()) / projection.max()
            )
        else:
            metrics['dynamic_range'] = 0.0
        
        # Signal-to-noise ratio estimate
        if projection.std() > 0:
            metrics['snr_estimate'] = float(projection.mean() / projection.std())
        else:
            metrics['snr_estimate'] = 0.0
        
        # Focus quality score (higher std along z indicates better focus variation)
        z_std = np.std(z_stack, axis=0)
        metrics['focus_variation'] = float(z_std.mean())
        
        # Coverage (fraction of non-zero pixels)
        metrics['coverage'] = float(np.sum(projection > 0) / projection.size)
        
        # Information content (entropy-based)
        if projection.max() > 0:
            normalized = projection / projection.max()
            hist, _ = np.histogram(normalized.flatten(), bins=256)
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / hist.sum()
                metrics['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            else:
                metrics['entropy'] = 0.0
        else:
            metrics['entropy'] = 0.0
        
        # Overall quality score (0-1)
        quality_score = 0.0
        if metrics['dynamic_range'] > 0.3:
            quality_score += 0.25
        if metrics['snr_estimate'] > 5:
            quality_score += 0.25
        if metrics['coverage'] > 0.1:
            quality_score += 0.25
        if metrics['entropy'] > 2:
            quality_score += 0.25
        
        metrics['quality_score'] = quality_score
        
        # Store metrics
        self.quality_metrics = metrics
        
        if quality_score < self.config.quality_threshold:
            warnings.warn(
                f"Low quality projection (score: {quality_score:.2f}). "
                f"Consider adjusting parameters or checking input data."
            )
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get the quality metrics from the last projection.
        
        Returns:
            Dict[str, Any]: Quality metrics dictionary.
        """
        return self.quality_metrics.copy()
    
    def optimize_z_range(self,
                         image: np.ndarray,
                         metric: str = 'focus',
                         percentile: float = 90) -> Tuple[int, int]:
        """Automatically determine optimal z-range for projection.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            metric: Metric to use ('focus', 'intensity', 'variance').
            percentile: Percentile threshold for selecting slices.
            
        Returns:
            Tuple[int, int]: Optimal z-range (start, end).
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        n_slices = image.shape[0]
        scores = np.zeros(n_slices)
        
        for z in range(n_slices):
            slice_img = image[z]
            
            if metric == 'focus':
                # Use variance of Laplacian as focus measure
                from scipy.ndimage import laplace
                for c in range(slice_img.shape[-1]):
                    lap = laplace(slice_img[..., c])
                    scores[z] += lap.var()
            
            elif metric == 'intensity':
                scores[z] = slice_img.mean()
            
            elif metric == 'variance':
                scores[z] = slice_img.var()
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        # Find range containing high-scoring slices
        threshold = np.percentile(scores, 100 - percentile)
        good_slices = np.where(scores >= threshold)[0]
        
        if len(good_slices) > 0:
            z_start = good_slices.min()
            z_end = good_slices.max() + 1
        else:
            z_start = 0
            z_end = n_slices
        
        logger.info(f"Optimized z-range: {z_start}-{z_end} (using {metric} metric)")
        
        return (z_start, z_end)
    
    def create_depth_coded_mip(self,
                               image: np.ndarray,
                               colormap: str = 'viridis') -> Tuple[np.ndarray, np.ndarray]:
        """Create a depth-coded MIP where color represents z-position.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            colormap: Matplotlib colormap name.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Depth-coded color image (Y, X, 3)
                - Depth map (Y, X) with z-indices
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        # Find maximum intensity position for each pixel
        if image.shape[-1] == 1:
            # Single channel
            max_z_indices = np.argmax(image[..., 0], axis=0)
            max_intensities = np.max(image[..., 0], axis=0)
        else:
            # Multi-channel: use mean across channels
            mean_image = np.mean(image, axis=-1)
            max_z_indices = np.argmax(mean_image, axis=0)
            max_intensities = np.max(mean_image, axis=0)
        
        # Normalize depth to 0-1
        n_slices = image.shape[0]
        depth_normalized = max_z_indices.astype(np.float32) / (n_slices - 1)
        
        # Apply colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        depth_colored = cmap(depth_normalized)[..., :3]  # RGB only
        
        # Weight by intensity
        max_intensities_norm = max_intensities / (max_intensities.max() + 1e-10)
        depth_colored = depth_colored * max_intensities_norm[..., np.newaxis]
        
        return depth_colored.astype(np.float32), max_z_indices
    
    def create_extended_depth_of_field(self,
                                       image: np.ndarray,
                                       window_size: int = 5) -> np.ndarray:
        """Create an extended depth of field projection using local focus measures.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            window_size: Window size for local focus calculation.
            
        Returns:
            np.ndarray: Extended depth of field image.
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        from scipy.ndimage import generic_filter
        
        n_slices, height, width, n_channels = image.shape
        result = np.zeros((height, width, n_channels))
        
        for c in range(n_channels):
            # Calculate focus measure for each slice
            focus_stack = np.zeros((n_slices, height, width))
            
            for z in range(n_slices):
                # Use variance as focus measure
                focus_stack[z] = generic_filter(
                    image[z, ..., c],
                    np.var,
                    size=window_size
                )
            
            # Select pixel from slice with best focus
            best_focus_indices = np.argmax(focus_stack, axis=0)
            
            for y in range(height):
                for x in range(width):
                    best_z = best_focus_indices[y, x]
                    result[y, x, c] = image[best_z, y, x, c]
        
        if result.shape[-1] == 1:
            result = result[..., 0]
        
        return result

    def save_mip_tiff(self,
                      mip: np.ndarray,
                      filepath: Union[str, Path],
                      dtype: str = 'auto',
                      normalize: bool = True,
                      compress: Union[int, None] = 6,
                      metadata: Optional[Dict[str, Any]] = None,
                      overwrite: bool = False) -> Path:
        """Save a MIP image to a TIFF file.
        
        Args:
            mip: MIP image array (Y, X) or (Y, X, C).
            filepath: Output TIFF path.
            dtype: Output dtype: 'auto', 'uint8', 'uint16', or 'float32'.
            normalize: If True, scale intensities to full range for integer dtypes.
            compress: Deflate compression level (0-9) or None for no compression.
            metadata: Optional metadata dict stored in TIFF tags where supported.
            overwrite: If False, raise if file exists.
        
        Returns:
            Path: The written file path.
        
        Raises:
            ValueError: If image shape is invalid or dtype unsupported.
            FileExistsError: If file exists and overwrite is False.
        """
        # Validate input dimensions
        if mip.ndim not in (2, 3):
            raise ValueError(f"MIP must be 2D or 3D (Y, X[, C]), got shape {mip.shape}")

        out_path = Path(filepath)
        ensure_directory(out_path.parent)

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {out_path}")

        # Determine target dtype
        target_dtype: str
        if dtype == 'auto':
            if np.issubdtype(mip.dtype, np.floating):
                target_dtype = 'uint16'
            elif np.issubdtype(mip.dtype, np.integer):
                target_dtype = 'uint8' if np.max(mip) <= 255 else 'uint16'
            else:
                target_dtype = 'uint16'
        else:
            target_dtype = dtype

        # Prepare array for saving
        array_to_save: np.ndarray
        if target_dtype in ('uint8', 'uint16'):
            # Convert to float for scaling if needed
            arr = mip.astype(np.float32)
            if normalize:
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
                else:
                    arr = np.zeros_like(arr)
            else:
                # If values are outside [0,1] and we're writing integer, clip sensibly
                arr = np.clip(arr, 0, 1) if np.issubdtype(mip.dtype, np.floating) else arr

            if target_dtype == 'uint8':
                array_to_save = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                array_to_save = (arr * 65535.0 + 0.5).astype(np.uint16)
        elif target_dtype == 'float32':
            arr = mip.astype(np.float32)
            if normalize:
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
            array_to_save = arr
        else:
            raise ValueError(f"Unsupported dtype: {target_dtype}")

        # Save using tifffile - keep original (Y,X,C) format for multichannel
        if array_to_save.ndim == 3:
            # Save as OME-TIFF for proper multichannel handling
            # Keep original (Y, X, C) format - don't transpose
            tifffile.imwrite(
                str(out_path),
                array_to_save,
                ome=True,
                metadata={
                    'axes': 'YXC',
                    'Channel': {'Name': [f'Channel_{i}' for i in range(array_to_save.shape[-1])]},
                    **({k: v for k, v in (metadata or {}).items() if k != 'axes'})
                },
                compression=('deflate' if compress is not None else None),
                compressionargs=({'level': int(compress)} if compress is not None else None)
            )
        else:
            # Single channel image
            tifffile.imwrite(
                str(out_path),
                array_to_save,
                photometric='minisblack',
                compression=('deflate' if compress is not None else None),
                compressionargs=({'level': int(compress)} if compress is not None else None),
                description=json.dumps(metadata or {})
            )

        logger.info(f"Saved MIP to {out_path} ({array_to_save.dtype}, shape={array_to_save.shape})")
        return out_path