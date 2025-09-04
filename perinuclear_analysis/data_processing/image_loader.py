"""
Phase 1: Image loading functionality for .nd2 and .tif files.
Comprehensive support for multi-channel, multi-z-stack microscopy data.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
import logging

import numpy as np
from PIL import Image
import nd2reader
import tifffile
from skimage import io

from ..core.config import Phase1Config
from ..core.utils import validate_file_path, get_pixel_size_from_metadata
from ..core.format_converter import FormatConverter


logger = logging.getLogger(__name__)


class ImageLoader:
    """Comprehensive image loader for microscopy data files."""
    
    def __init__(self, config: Optional[Phase1Config] = None, auto_convert: bool = True):
        """Initialize the image loader.
        
        Args:
            config: Phase 1 configuration. If None, uses defaults.
            auto_convert: If True, automatically converts non-TIFF formats to TIFF.
        """
        self.config = config or Phase1Config()
        self.metadata: Dict[str, Any] = {}
        self.pixel_size_um: Optional[float] = None
        self.channels: List[str] = []
        self.auto_convert = auto_convert
        self.converter = FormatConverter(preserve_original=True)
        self.converted_files: Dict[str, Path] = {}  # Track converted files
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if self.config.verbose:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def load_image(self, filepath: Union[str, Path], force_convert: bool = None) -> np.ndarray:
        """Load an image file, automatically handling format conversion.
        
        Args:
            filepath: Path to the image file.
            force_convert: Override auto_convert setting. If None, uses class setting.
            
        Returns:
            np.ndarray: Image data array with dimensions (Z, Y, X, C) or (Y, X, C).
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        should_convert = force_convert if force_convert is not None else self.auto_convert
        
        # Determine file format
        suffix = filepath.suffix.lower()
        
        if suffix == '.nd2' and should_convert:
            # Convert ND2 to OME-TIFF
            logger.info(f"Converting {filepath.name} to OME-TIFF format for pipeline compatibility")
            
            # Check if already converted
            ome_tiff_path = filepath.with_suffix('.ome.tiff')
            if ome_tiff_path.exists():
                # Check if we have metadata
                metadata_path = ome_tiff_path.with_suffix('.json')
                if metadata_path.exists():
                    logger.info(f"Using existing converted file: {ome_tiff_path}")
                    self.converted_files[str(filepath)] = ome_tiff_path
                    return self.load_tiff_with_metadata(ome_tiff_path)
            
            # Perform conversion
            output_path, metadata = self.converter.nd2_to_ome_tiff(filepath, ome_tiff_path, save_metadata=True)
            self.converted_files[str(filepath)] = output_path
            
            # Load metadata
            self.metadata = metadata
            self.pixel_size_um = metadata.get('pixel_size_um')
            self.channels = metadata.get('channel_names', [])
            
            # Load the converted TIFF
            return self.load_tiff_with_metadata(output_path)
        
        elif suffix == '.nd2' and not should_convert:
            # Load ND2 directly (original behavior)
            return self.load_nd2(filepath)
        
        elif suffix in ['.tif', '.tiff']:
            # Check if this is a converted file with metadata
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                return self.load_tiff_with_metadata(filepath)
            else:
                # Load as regular TIFF
                return self.load_tif_image(filepath)
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def load_tiff_with_metadata(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load a TIFF file with associated metadata.
        
        Args:
            filepath: Path to the TIFF file.
            
        Returns:
            np.ndarray: Image data array.
        """
        filepath = Path(filepath)
        
        # Load metadata
        metadata = self.converter.load_metadata(filepath)
        if metadata:
            self.metadata = metadata
            self.pixel_size_um = metadata.get('pixel_size_um')
            self.channels = metadata.get('channel_names', [])
            
            logger.info(f"Loaded metadata from converted TIFF: pixel_size={self.pixel_size_um}μm")
        
        # Load image data
        try:
            image = tifffile.imread(str(filepath))
            
            # Ensure proper dimensions
            if image.ndim == 2:
                image = image[np.newaxis, ..., np.newaxis]
            elif image.ndim == 3:
                # Check if it's (Z, Y, X) or (Y, X, C)
                if metadata and metadata.get('dimensions', {}).get('channels', 1) > 1:
                    # Multi-channel, single Z
                    if image.shape[-1] == metadata['dimensions']['channels']:
                        image = image[np.newaxis, ...]
                    else:
                        # Z-stack, need to add channel dimension
                        image = image[..., np.newaxis]
                else:
                    # Single channel Z-stack
                    image = image[..., np.newaxis]
            
            self._validate_image_dimensions(image)
            
            logger.info(f"Successfully loaded TIFF image with shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load TIFF file: {e}")
            raise ValueError(f"Error loading TIFF file: {e}")
    
    def load_nd2(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load an .nd2 file and extract metadata.
        
        Args:
            filepath: Path to the .nd2 file.
            
        Returns:
            np.ndarray: Image data array with dimensions (Z, Y, X, C) or (Y, X, C).
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid or corrupted.
        """
        filepath = Path(filepath)
        validate_file_path(filepath, ['.nd2'])
        
        logger.info(f"Loading ND2 file: {filepath}")
        
        try:
            with nd2reader.ND2Reader(str(filepath)) as reader:
                # Extract metadata
                self._extract_nd2_metadata(reader)
                
                # Determine dimensions
                z_levels = len(reader.metadata.get('z_levels', [1]))
                channels = len(reader.metadata.get('channels', []))
                
                if channels == 0:
                    channels = 1
                
                # Handle different dimension configurations
                if 'z' in reader.axes and 'c' in reader.axes:
                    # Multi-channel, multi-z
                    reader.iter_axes = 'zc'
                    reader.bundle_axes = 'yx'
                    
                    images = []
                    for z in range(z_levels):
                        channel_images = []
                        for c in range(channels):
                            reader.default_coords['z'] = z
                            reader.default_coords['c'] = c
                            channel_images.append(reader[0])
                        images.append(np.stack(channel_images, axis=-1))
                    
                    image_array = np.stack(images, axis=0)
                    
                elif 'z' in reader.axes:
                    # Single channel, multi-z
                    reader.iter_axes = 'z'
                    reader.bundle_axes = 'yx'
                    
                    images = []
                    for z in range(z_levels):
                        reader.default_coords['z'] = z
                        img = reader[0]
                        if img.ndim == 2:
                            img = img[..., np.newaxis]
                        images.append(img)
                    
                    image_array = np.stack(images, axis=0)
                    
                elif 'c' in reader.axes:
                    # Multi-channel, single z
                    reader.iter_axes = 'c'
                    reader.bundle_axes = 'yx'
                    
                    images = []
                    for c in range(channels):
                        reader.default_coords['c'] = c
                        images.append(reader[0])
                    
                    image_array = np.stack(images, axis=-1)
                    if image_array.ndim == 3:
                        image_array = image_array[np.newaxis, ...]
                        
                else:
                    # Single channel, single z
                    image_array = np.array(reader[0])
                    if image_array.ndim == 2:
                        image_array = image_array[np.newaxis, ..., np.newaxis]
                    elif image_array.ndim == 3:
                        image_array = image_array[np.newaxis, ...]
                
                # Validate dimensions
                self._validate_image_dimensions(image_array)
                
                logger.info(f"Successfully loaded ND2 image with shape: {image_array.shape}")
                logger.info(f"Pixel size: {self.pixel_size_um} μm")
                logger.info(f"Channels: {self.channels}")
                
                return image_array
                
        except Exception as e:
            logger.error(f"Failed to load ND2 file: {e}")
            raise ValueError(f"Error loading ND2 file: {e}")
    
    def _extract_nd2_metadata(self, reader: nd2reader.ND2Reader) -> None:
        """Extract metadata from ND2 reader.
        
        Args:
            reader: ND2Reader instance.
        """
        metadata = reader.metadata
        
        if self.config.extract_all_metadata:
            self.metadata = dict(metadata)
        
        # Extract pixel size
        if 'pixel_microns' in metadata:
            self.pixel_size_um = metadata['pixel_microns']
        elif 'calibration' in metadata:
            self.pixel_size_um = metadata['calibration']
        else:
            raise ValueError(
                "No pixel size found in image metadata. This is required for accurate "
                "quantitative analysis. Please ensure your image files contain proper "
                "calibration information, or manually specify pixel size when loading."
            )
        
        # Extract channel information
        if 'channels' in metadata:
            self.channels = metadata['channels']
        else:
            raise ValueError(
                "No channel information found in image metadata. Channel names are "
                "required for proper analysis. Please ensure your image files contain "
                "proper channel information."
            )
        
        # Store additional useful metadata
        self.metadata['axes'] = reader.axes
        self.metadata['shape'] = reader.frame_shape
        self.metadata['z_levels'] = len(metadata.get('z_levels', [1]))
        self.metadata['timepoints'] = reader.metadata.get('total_images_per_channel', 1)
    
    def load_tif_mask(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load a .tif mask file.
        
        Args:
            filepath: Path to the .tif file.
            
        Returns:
            np.ndarray: Mask array (2D or 3D).
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        filepath = Path(filepath)
        validate_file_path(filepath, ['.tif', '.tiff'])
        
        logger.info(f"Loading TIF mask: {filepath}")
        
        try:
            # Load using scikit-image
            mask = io.imread(str(filepath))
            
            # Handle multi-page TIFFs
            if mask.ndim == 3 and mask.shape[2] in [3, 4]:
                # RGB or RGBA image, convert to grayscale
                mask = mask.mean(axis=2).astype(mask.dtype)
            
            # Validate mask
            self._validate_mask(mask)
            
            logger.info(f"Successfully loaded mask with shape: {mask.shape}")
            
            return mask
            
        except Exception as e:
            logger.error(f"Failed to load TIF mask: {e}")
            raise ValueError(f"Error loading TIF mask: {e}")
    
    def load_tif_image(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load a regular .tif image file (not specifically a mask).
        
        Args:
            filepath: Path to the .tif file.
            
        Returns:
            np.ndarray: Image array.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        filepath = Path(filepath)
        validate_file_path(filepath, ['.tif', '.tiff'])
        
        logger.info(f"Loading TIF image: {filepath}")
        
        try:
            # Load using scikit-image
            image = io.imread(str(filepath))
            
            # Extract metadata if available
            with Image.open(str(filepath)) as img:
                if hasattr(img, 'tag'):
                    tags = dict(img.tag)
                    self.metadata['tif_tags'] = tags
                    
                    # Try to extract pixel size from TIF tags
                    if 282 in tags and 283 in tags:  # X and Y resolution tags
                        x_res = tags[282][0] if isinstance(tags[282], tuple) else tags[282]
                        y_res = tags[283][0] if isinstance(tags[283], tuple) else tags[283]
                        
                        # Convert to micrometers if in pixels per inch
                        if 296 in tags:  # Resolution unit tag
                            unit = tags[296]
                            if unit == 2:  # Inches
                                self.pixel_size_um = 25400.0 / x_res  # Convert from DPI to μm
            
            # Ensure proper dimensions
            if image.ndim == 2:
                image = image[np.newaxis, ..., np.newaxis]
            elif image.ndim == 3:
                # Could be Z-stack or multi-channel
                if image.shape[2] <= 4:  # Likely channels
                    image = image[np.newaxis, ...]
                else:  # Likely z-stack
                    image = image[..., np.newaxis]
            
            self._validate_image_dimensions(image)
            
            logger.info(f"Successfully loaded TIF image with shape: {image.shape}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load TIF image: {e}")
            raise ValueError(f"Error loading TIF image: {e}")
    
    def _validate_image_dimensions(self, image: np.ndarray) -> None:
        """Validate image dimensions.
        
        Args:
            image: Image array to validate.
            
        Raises:
            ValueError: If dimensions are invalid.
        """
        if image.ndim not in [3, 4]:
            raise ValueError(f"Image must be 3D or 4D, got {image.ndim}D")
        
        # Check size limits
        height, width = image.shape[-3:-1] if image.ndim == 4 else image.shape[-2:]
        
        if height < self.config.min_image_dimension or width < self.config.min_image_dimension:
            raise ValueError(
                f"Image dimensions too small: {height}x{width}. "
                f"Minimum: {self.config.min_image_dimension}x{self.config.min_image_dimension}"
            )
        
        if height > self.config.max_image_dimension or width > self.config.max_image_dimension:
            raise ValueError(
                f"Image dimensions too large: {height}x{width}. "
                f"Maximum: {self.config.max_image_dimension}x{self.config.max_image_dimension}"
            )
    
    def _validate_mask(self, mask: np.ndarray) -> None:
        """Validate mask array.
        
        Args:
            mask: Mask array to validate.
            
        Raises:
            ValueError: If mask is invalid.
        """
        if mask.ndim not in [2, 3]:
            raise ValueError(f"Mask must be 2D or 3D, got {mask.ndim}D")
        
        # Check if mask is binary or label mask
        unique_values = np.unique(mask)
        
        if len(unique_values) == 2:
            # Binary mask
            if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0, 255]):
                warnings.warn(f"Unusual binary mask values: {unique_values}")
        
        logger.info(f"Mask contains {len(unique_values)} unique values")
    
    def get_pixel_size(self) -> float:
        """Get the pixel size in micrometers.
        
        Returns:
            float: Pixel size in micrometers.
            
        Raises:
            ValueError: If no pixel size is available.
        """
        if self.pixel_size_um is None:
            raise ValueError(
                "No pixel size available. Image metadata must contain calibration information."
            )
        return self.pixel_size_um
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all extracted metadata.
        
        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return self.metadata.copy()
    
    def get_channel_names(self) -> List[str]:
        """Get channel names.
        
        Returns:
            List[str]: List of channel names.
        """
        return self.channels.copy()
    
    def rename_channels(self, new_names: List[str]) -> None:
        """Rename channels.
        
        Args:
            new_names: List of new channel names. Must match the number of channels.
            
        Raises:
            ValueError: If the number of new names doesn't match the number of channels.
        """
        if len(new_names) != len(self.channels):
            raise ValueError(
                f"Number of new names ({len(new_names)}) must match number of channels ({len(self.channels)})"
            )
        self.channels = new_names.copy()
        logger.info(f"Renamed channels to: {self.channels}")
    
    def set_channel_name(self, channel_index: int, new_name: str) -> None:
        """Set the name of a specific channel.
        
        Args:
            channel_index: Index of the channel to rename.
            new_name: New name for the channel.
            
        Raises:
            IndexError: If channel_index is out of range.
        """
        if channel_index < 0 or channel_index >= len(self.channels):
            raise IndexError(f"Channel index {channel_index} out of range (0-{len(self.channels)-1})")
        
        old_name = self.channels[channel_index]
        self.channels[channel_index] = new_name
        logger.info(f"Renamed channel {channel_index} from '{old_name}' to '{new_name}'")
    
    def extract_channel(self, image: np.ndarray, channel: Union[int, str]) -> np.ndarray:
        """Extract a specific channel from the image.
        
        Args:
            image: Input image array (Z, Y, X, C) or (Y, X, C).
            channel: Channel index or name.
            
        Returns:
            np.ndarray: Single channel image.
            
        Raises:
            ValueError: If channel is invalid.
        """
        if isinstance(channel, str):
            if channel not in self.channels:
                raise ValueError(f"Unknown channel: {channel}. Available: {self.channels}")
            channel = self.channels.index(channel)
        
        if image.ndim == 4:
            return image[..., channel]
        elif image.ndim == 3:
            return image[..., channel]
        else:
            raise ValueError(f"Image must be 3D or 4D for channel extraction, got {image.ndim}D")
    
    def split_channels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Split multi-channel image into separate channels.
        
        Args:
            image: Input image array (Z, Y, X, C) or (Y, X, C).
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping channel names to arrays.
        """
        channels = {}
        
        num_channels = image.shape[-1]
        for i in range(num_channels):
            channel_name = self.channels[i] if i < len(self.channels) else f'Channel_{i}'
            channels[channel_name] = self.extract_channel(image, i)
        
        return channels
    
    def get_z_stack(self, image: np.ndarray, z_index: int) -> np.ndarray:
        """Extract a specific z-slice from the image.
        
        Args:
            image: Input image array (Z, Y, X, C).
            z_index: Z-slice index.
            
        Returns:
            np.ndarray: Single z-slice (Y, X, C).
            
        Raises:
            ValueError: If image is not 4D or z_index is invalid.
        """
        if image.ndim != 4:
            raise ValueError(f"Image must be 4D for z-stack extraction, got {image.ndim}D")
        
        if z_index < 0 or z_index >= image.shape[0]:
            raise ValueError(f"Invalid z-index: {z_index}. Valid range: 0-{image.shape[0]-1}")
        
        return image[z_index]
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about an image array.
        
        Args:
            image: Input image array.
            
        Returns:
            Dict[str, Any]: Image information.
        """
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'ndim': image.ndim,
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
        }
        
        if image.ndim == 4:
            info['z_slices'] = image.shape[0]
            info['height'] = image.shape[1]
            info['width'] = image.shape[2]
            info['channels'] = image.shape[3]
        elif image.ndim == 3:
            info['height'] = image.shape[0]
            info['width'] = image.shape[1]
            info['channels'] = image.shape[2]
        
        info['pixel_size_um'] = self.get_pixel_size()
        info['channel_names'] = self.get_channel_names()
        
        return info
    
    def batch_process_directory(self, directory: Union[str, Path], pattern: str = "*.nd2") -> Dict[Path, Path]:
        """Process all matching files in a directory, converting to TIFF if needed.
        
        Args:
            directory: Directory containing image files.
            pattern: File pattern to match.
            
        Returns:
            Dictionary mapping original paths to processed paths.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all matching files
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files matching pattern {pattern}")
        
        processed = {}
        for file in files:
            try:
                # Load and potentially convert
                self.load_image(file)
                
                # Track the result
                if str(file) in self.converted_files:
                    processed[file] = self.converted_files[str(file)]
                else:
                    processed[file] = file
                    
                logger.info(f"Processed: {file.name}")
            except Exception as e:
                logger.error(f"Failed to process {file}: {e}")
                processed[file] = None
        
        return processed
    
    def get_converted_files(self) -> Dict[str, Path]:
        """Get dictionary of files that have been converted.
        
        Returns:
            Dictionary mapping original paths to converted paths.
        """
        return self.converted_files.copy()