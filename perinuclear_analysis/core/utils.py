"""
Phase 1: Utility functions for the perinuclear analysis module.
Basic helpers for file I/O, coordinate transformations, and validation.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import hashlib
import json
import logging

import numpy as np


logger = logging.getLogger(__name__)


def validate_file_path(filepath: Path, valid_extensions: List[str]) -> None:
    """Validate that a file exists and has the correct extension.
    
    Args:
        filepath: Path to validate.
        valid_extensions: List of valid file extensions (e.g., ['.nd2', '.tif']).
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file extension is not valid.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    extension = filepath.suffix.lower()
    valid_extensions = [ext.lower() for ext in valid_extensions]
    
    if extension not in valid_extensions:
        raise ValueError(
            f"Invalid file extension: {extension}. "
            f"Valid extensions: {valid_extensions}"
        )


def get_pixel_size_from_metadata(metadata: Dict[str, Any]) -> Optional[float]:
    """Extract pixel size from metadata dictionary.
    
    Args:
        metadata: Metadata dictionary from image file.
        
    Returns:
        Optional[float]: Pixel size in micrometers, or None if not found.
    """
    # Common metadata keys for pixel size
    pixel_size_keys = [
        'pixel_microns',
        'pixel_size',
        'pixelSize',
        'calibration',
        'PhysicalSizeX',
        'PhysicalSizeY',
        'pixel_size_um',
        'pixel_size_µm',
    ]
    
    for key in pixel_size_keys:
        if key in metadata:
            value = metadata[key]
            
            # Handle different value formats
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                return float(value[0])
            elif isinstance(value, dict):
                # Some formats store x and y separately
                if 'x' in value:
                    return float(value['x'])
                elif 'value' in value:
                    return float(value['value'])
    
    return None


def convert_pixels_to_microns(pixels: Union[float, np.ndarray], 
                              pixel_size_um: float) -> Union[float, np.ndarray]:
    """Convert pixel measurements to micrometers.
    
    Args:
        pixels: Measurement in pixels.
        pixel_size_um: Size of one pixel in micrometers.
        
    Returns:
        Union[float, np.ndarray]: Measurement in micrometers.
    """
    return pixels * pixel_size_um


def convert_microns_to_pixels(microns: Union[float, np.ndarray], 
                               pixel_size_um: float) -> Union[float, np.ndarray]:
    """Convert micrometer measurements to pixels.
    
    Args:
        microns: Measurement in micrometers.
        pixel_size_um: Size of one pixel in micrometers.
        
    Returns:
        Union[float, np.ndarray]: Measurement in pixels.
    """
    return microns / pixel_size_um


def get_fluorophore_color(channel_name: str) -> str:
    """Map fluorophore names to appropriate display colors based on their spectral properties.
    
    Args:
        channel_name: Name of the fluorescent channel (e.g., 'AF488', 'DAPI', 'GFP').
        
    Returns:
        str: Appropriate color name for visualization.
    """
    channel_lower = channel_name.lower().replace(' ', '').replace('-', '').replace('_', '')
    
    # Blue fluorophores (350-450nm excitation)
    if any(x in channel_lower for x in ['dapi', 'hoechst', 'af350', 'af405', 'pacific']):
        return 'blue'
    
    # Cyan fluorophores (450-490nm excitation)  
    elif any(x in channel_lower for x in ['cfp', 'af430', 'af440', 'cyan']):
        return 'cyan'
    
    # Green fluorophores (480-520nm excitation)
    elif any(x in channel_lower for x in ['gfp', 'fitc', 'af488', 'alexa488', 'af514', 'green']):
        return 'green'
    
    # Yellow fluorophores (520-570nm excitation)
    elif any(x in channel_lower for x in ['yfp', 'af532', 'af546', 'alexa532', 'alexa546', 'yellow']):
        return 'yellow'
    
    # Orange fluorophores (550-580nm excitation)
    elif any(x in channel_lower for x in ['af555', 'af568', 'alexa555', 'alexa568', 'tritc', 'orange', 'dsred']):
        return 'orange'
    
    # Red fluorophores (580-650nm excitation)
    elif any(x in channel_lower for x in ['af594', 'af633', 'alexa594', 'alexa633', 'texas', 'red']):
        return 'red'
    
    # Far-red/Near-infrared fluorophores (650+ nm excitation)
    elif any(x in channel_lower for x in ['af647', 'af680', 'af750', 'alexa647', 'alexa680', 'cy5', 'cy7', 'farred']):
        return 'magenta'  # Often displayed as magenta for visibility
    
    # Bright field or phase contrast
    elif any(x in channel_lower for x in ['bf', 'brightfield', 'phase', 'dic', 'transmission']):
        return 'gray'
    
    # Fallback for unknown fluorophores
    else:
        return 'white'


def create_channel_color_mapping(channel_names: List[str]) -> Dict[str, str]:
    """Create automatic color mapping for a list of channel names.
    
    Args:
        channel_names: List of channel names from microscopy data.
        
    Returns:
        Dict mapping channel names to appropriate colors.
    """
    return {channel: get_fluorophore_color(channel) for channel in channel_names}


def get_colormap_from_fluorophore(channel_name: str):
    """Get appropriate matplotlib colormap for a fluorophore channel.
    
    Args:
        channel_name: Name of the fluorescent channel (e.g., 'AF488', 'DAPI', 'GFP').
        
    Returns:
        matplotlib.colors.Colormap: Appropriate colormap for visualization.
    """
    import matplotlib.pyplot as plt
    
    color_name = get_fluorophore_color(channel_name)
    
    color_to_cmap = {
        'red': 'Reds',
        'orange': 'Oranges', 
        'green': 'Greens',
        'blue': 'Blues',
        'cyan': 'Blues',  # Use Blues for cyan
        'yellow': 'YlOrBr',  # Yellow-Orange-Brown for yellow
        'magenta': 'plasma',  # Plasma colormap for magenta/far-red
        'gray': 'gray',
        'white': 'viridis'  # Fallback to viridis for unknown
    }
    
    cmap_name = color_to_cmap.get(color_name, 'viridis')
    return plt.cm.get_cmap(cmap_name)


def calculate_image_checksum(image: np.ndarray) -> str:
    """Calculate MD5 checksum of an image array.
    
    Args:
        image: Image array.
        
    Returns:
        str: MD5 checksum as hexadecimal string.
    """
    return hashlib.md5(image.tobytes()).hexdigest()


def normalize_image(image: np.ndarray, 
                    method: str = 'minmax',
                    percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """Normalize image intensities.
    
    Args:
        image: Input image array.
        method: Normalization method ('minmax', 'percentile', 'zscore').
        percentile_range: Percentile range for 'percentile' method.
        
    Returns:
        np.ndarray: Normalized image.
    """
    image = image.astype(np.float32)
    
    if method == 'minmax':
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
    
    elif method == 'percentile':
        min_val = np.percentile(image, percentile_range[0])
        max_val = np.percentile(image, percentile_range[1])
        if max_val > min_val:
            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = image.mean()
        std_val = image.std()
        if std_val > 0:
            image = (image - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return image


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path.
        
    Returns:
        Path: Directory path as Path object.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_metadata(metadata: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save metadata to a JSON file.
    
    Args:
        metadata: Metadata dictionary.
        filepath: Output file path.
    """
    filepath = Path(filepath)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    metadata = convert_types(metadata)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata from a JSON file.
    
    Args:
        filepath: Input file path.
        
    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def calculate_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate the overlap between two binary masks.
    
    Args:
        mask1: First binary mask.
        mask2: Second binary mask.
        
    Returns:
        float: Overlap coefficient (Jaccard index).
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get the bounding box of a binary mask.
    
    Args:
        mask: Binary mask.
        
    Returns:
        Tuple[int, int, int, int]: (min_row, min_col, max_row, max_col).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return (rmin, cmin, rmax + 1, cmax + 1)


def crop_to_content(image: np.ndarray, mask: Optional[np.ndarray] = None, 
                     padding: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop an image to its content with optional padding.
    
    Args:
        image: Input image.
        mask: Optional mask to define content area.
        padding: Padding to add around content.
        
    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: 
            Cropped image and bounding box coordinates.
    """
    if mask is None:
        # Use image content (non-zero pixels) to determine crop
        if image.ndim == 2:
            mask = image > 0
        elif image.ndim == 3:
            mask = image.mean(axis=2) > 0
        else:
            mask = image.mean(axis=-1).mean(axis=0) > 0
    
    rmin, cmin, rmax, cmax = get_bounding_box(mask)
    
    # Add padding
    rmin = max(0, rmin - padding)
    cmin = max(0, cmin - padding)
    rmax = min(image.shape[0], rmax + padding)
    cmax = min(image.shape[1], cmax + padding)
    
    # Crop image
    if image.ndim == 2:
        cropped = image[rmin:rmax, cmin:cmax]
    elif image.ndim == 3:
        cropped = image[rmin:rmax, cmin:cmax, :]
    else:
        cropped = image[:, rmin:rmax, cmin:cmax, :]
    
    return cropped, (rmin, cmin, rmax, cmax)


def generate_synthetic_image(shape: Tuple[int, ...], 
                              pattern: str = 'gradient',
                              noise_level: float = 0.1) -> np.ndarray:
    """Generate a synthetic test image.
    
    Args:
        shape: Image shape.
        pattern: Pattern type ('gradient', 'spots', 'stripes').
        noise_level: Amount of noise to add.
        
    Returns:
        np.ndarray: Synthetic image.
    """
    if pattern == 'gradient':
        if len(shape) == 2:
            image = np.meshgrid(
                np.linspace(0, 1, shape[0]),
                np.linspace(0, 1, shape[1]),
                indexing='ij'
            )[0]
        elif len(shape) == 3:
            image = np.meshgrid(
                np.linspace(0, 1, shape[0]),
                np.linspace(0, 1, shape[1]),
                indexing='ij'
            )[0]
            image = np.stack([image] * shape[2], axis=2)
        else:
            image = np.ones(shape) * 0.5
    
    elif pattern == 'spots':
        image = np.zeros(shape)
        # Add random spots
        n_spots = 20
        for _ in range(n_spots):
            if len(shape) == 2:
                y, x = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                yy, xx = np.ogrid[:shape[0], :shape[1]]
                mask = (yy - y) ** 2 + (xx - x) ** 2 <= 100
                image[mask] = 1
            else:
                # For higher dimensions, just use first 2D slice
                y, x = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
                yy, xx = np.ogrid[:shape[0], :shape[1]]
                mask = (yy - y) ** 2 + (xx - x) ** 2 <= 100
                if len(shape) == 3:
                    image[mask, :] = 1
                else:
                    image[:, mask, :] = 1
    
    elif pattern == 'stripes':
        if len(shape) == 2:
            x = np.arange(shape[1])
            image = np.sin(x * 0.1)[np.newaxis, :] * np.ones(shape)
        else:
            x = np.arange(shape[1] if len(shape) == 3 else shape[2])
            stripes = np.sin(x * 0.1)
            image = np.ones(shape)
            if len(shape) == 3:
                image *= stripes[np.newaxis, :, np.newaxis]
            else:
                image *= stripes[np.newaxis, np.newaxis, :, np.newaxis]
    
    else:
        image = np.ones(shape) * 0.5
    
    # Add noise
    if noise_level > 0:
        noise = np.random.randn(*shape) * noise_level
        image = image + noise
    
    # Normalize to 0-1
    image = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    return image


def get_memory_usage(obj: Any) -> int:
    """Get memory usage of an object in bytes.
    
    Args:
        obj: Object to measure.
        
    Returns:
        int: Memory usage in bytes.
    """
    import sys
    
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    else:
        return sys.getsizeof(obj)


def format_bytes(bytes_value: int) -> str:
    """Format bytes value as human-readable string.
    
    Args:
        bytes_value: Number of bytes.
        
    Returns:
        str: Formatted string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"