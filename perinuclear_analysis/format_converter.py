"""
Format converter for microscopy image files.
Converts .nd2 files to .tiff while preserving metadata.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging

import numpy as np
import tifffile
import nd2reader

logger = logging.getLogger(__name__)


class FormatConverter:
    """Convert microscopy image formats while preserving metadata."""
    
    def __init__(self, preserve_original: bool = True):
        """Initialize the format converter.
        
        Args:
            preserve_original: If True, keeps the original file after conversion.
        """
        self.preserve_original = preserve_original
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def nd2_to_tiff(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        save_metadata: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """Convert .nd2 file to .tiff format with metadata preservation.
        
        Args:
            input_path: Path to the input .nd2 file.
            output_path: Path for the output .tiff file. If None, uses same name with .tiff extension.
            save_metadata: If True, saves metadata to a separate JSON file.
            
        Returns:
            Tuple of (output_path, metadata_dict)
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If conversion fails.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not input_path.suffix.lower() == '.nd2':
            raise ValueError(f"Input file must be .nd2 format, got: {input_path.suffix}")
        
        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix('.tiff')
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting {input_path} to {output_path}")
        
        try:
            # Read ND2 file and extract metadata
            with nd2reader.ND2Reader(str(input_path)) as reader:
                metadata = self._extract_comprehensive_metadata(reader)
                image_data = self._read_nd2_as_array(reader)
            
            # Save as TIFF with metadata
            self._save_as_tiff(image_data, output_path, metadata)
            
            # Save metadata to JSON if requested
            if save_metadata:
                metadata_path = output_path.with_suffix('.json')
                self._save_metadata_json(metadata, metadata_path)
                logger.info(f"Metadata saved to {metadata_path}")
            
            # Cache metadata for quick access
            self.metadata_cache[str(output_path)] = metadata
            
            logger.info(f"Successfully converted to {output_path}")
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert {input_path}: {e}")
            raise ValueError(f"Conversion failed: {e}")
    
    def _extract_comprehensive_metadata(self, reader: nd2reader.ND2Reader) -> Dict[str, Any]:
        """Extract comprehensive metadata from ND2 reader.
        
        Args:
            reader: ND2Reader instance.
            
        Returns:
            Dictionary containing all relevant metadata.
        """
        metadata = {}
        
        # Basic metadata
        metadata['original_format'] = 'nd2'
        metadata['axes'] = reader.axes
        metadata['frame_shape'] = reader.frame_shape
        
        # Dimensional information
        metadata['dimensions'] = {
            'width': reader.metadata.get('width', reader.frame_shape[1] if len(reader.frame_shape) > 1 else None),
            'height': reader.metadata.get('height', reader.frame_shape[0] if len(reader.frame_shape) > 0 else None),
            'z_levels': len(reader.metadata.get('z_levels', [1])),
            'channels': len(reader.metadata.get('channels', [])) or 1,
            'timepoints': reader.metadata.get('total_images_per_channel', 1)
        }
        
        # Pixel/voxel information - CRITICAL for physical measurements
        metadata['pixel_info'] = {
            'pixel_microns': reader.metadata.get('pixel_microns'),
            'pixel_microns_x': reader.metadata.get('pixel_microns_x'),
            'pixel_microns_y': reader.metadata.get('pixel_microns_y'),
            'voxel_size_z': reader.metadata.get('z_step', None),
            'calibration': reader.metadata.get('calibration'),
        }
        
        # Calculate actual pixel size
        if metadata['pixel_info']['pixel_microns']:
            metadata['pixel_size_um'] = metadata['pixel_info']['pixel_microns']
        elif metadata['pixel_info']['pixel_microns_x'] and metadata['pixel_info']['pixel_microns_y']:
            # Use average if x and y are different
            metadata['pixel_size_um'] = (
                metadata['pixel_info']['pixel_microns_x'] + 
                metadata['pixel_info']['pixel_microns_y']
            ) / 2
        elif metadata['pixel_info']['calibration']:
            metadata['pixel_size_um'] = metadata['pixel_info']['calibration']
        else:
            metadata['pixel_size_um'] = None
            logger.warning("No pixel size information found in ND2 metadata")
        
        # Channel information
        if 'channels' in reader.metadata:
            metadata['channel_names'] = reader.metadata['channels']
        else:
            raise ValueError(
                "No channel information found in image file. Channel names are "
                "required for proper analysis. Please ensure your image acquisition "
                "software saves proper channel metadata."
            )
        
        # Z-stack information
        if 'z_levels' in reader.metadata:
            metadata['z_levels'] = reader.metadata['z_levels']
            metadata['z_coordinates'] = reader.metadata.get('z_coordinates', [])
        
        # Acquisition information
        metadata['acquisition'] = {
            'date': reader.metadata.get('date'),
            'time': reader.metadata.get('time'),
            'experiment': reader.metadata.get('experiment', {}),
        }
        
        # Microscope settings
        metadata['microscope'] = {
            'objective': reader.metadata.get('objective'),
            'magnification': reader.metadata.get('magnification'),
            'numerical_aperture': reader.metadata.get('numerical_aperture'),
            'immersion': reader.metadata.get('immersion'),
        }
        
        # Additional raw metadata (for completeness)
        metadata['raw_metadata'] = {
            k: v for k, v in reader.metadata.items() 
            if k not in ['channels', 'z_levels', 'pixel_microns', 'width', 'height']
            and not isinstance(v, (list, dict)) or len(str(v)) < 1000  # Avoid huge data structures
        }
        
        return metadata
    
    def _read_nd2_as_array(self, reader: nd2reader.ND2Reader) -> np.ndarray:
        """Read ND2 file as numpy array.
        
        Args:
            reader: ND2Reader instance.
            
        Returns:
            Image array with dimensions (Z, Y, X, C) or appropriate subset.
        """
        z_levels = len(reader.metadata.get('z_levels', [1]))
        channels = len(reader.metadata.get('channels', [])) or 1
        
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
            
            return np.stack(images, axis=0)
            
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
            
            return np.stack(images, axis=0)
            
        elif 'c' in reader.axes:
            # Multi-channel, single z
            reader.iter_axes = 'c'
            reader.bundle_axes = 'yx'
            
            images = []
            for c in range(channels):
                reader.default_coords['c'] = c
                images.append(reader[0])
            
            image_array = np.stack(images, axis=-1)
            return image_array[np.newaxis, ...]  # Add Z dimension
            
        else:
            # Single channel, single z
            image_array = np.array(reader[0])
            if image_array.ndim == 2:
                return image_array[np.newaxis, ..., np.newaxis]
            return image_array[np.newaxis, ...]
    
    def _save_as_tiff(
        self, 
        image_data: np.ndarray, 
        output_path: Path, 
        metadata: Dict[str, Any]
    ) -> None:
        """Save image data as TIFF with metadata.
        
        Args:
            image_data: Image array to save.
            output_path: Output file path.
            metadata: Metadata dictionary to embed.
        """
        # Prepare ImageJ-compatible metadata
        imagej_metadata = {
            'axes': 'ZCYX' if image_data.ndim == 4 else 'ZYX',
            'unit': 'um',
        }
        
        # Add pixel size information if available
        if metadata.get('pixel_size_um'):
            resolution = 1.0 / metadata['pixel_size_um']  # Convert to pixels per unit
            imagej_metadata['spacing'] = metadata['pixel_size_um']
            
            # Set resolution for TIFF tags
            resolution_tag = (resolution, resolution)
        else:
            resolution_tag = None
        
        # Add z-spacing if available
        if metadata['pixel_info'].get('voxel_size_z'):
            imagej_metadata['spacing_z'] = metadata['pixel_info']['voxel_size_z']
        
        # Convert metadata to JSON string for TIFF description tag
        description = json.dumps(metadata, indent=2, default=str)
        
        # Save with tifffile
        tifffile.imwrite(
            output_path,
            image_data,
            imagej=True,
            metadata=imagej_metadata,
            description=description,
            resolution=resolution_tag if resolution_tag else None,
            resolutionunit='CENTIMETER' if resolution_tag else None,  # tifffile uses cm internally
            compression='zlib',
            compressionargs={'level': 6}
        )
    
    def _save_metadata_json(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary.
            output_path: Output JSON file path.
        """
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_metadata(self, tiff_path: Union[str, Path]) -> Dict[str, Any]:
        """Load metadata from a converted TIFF file.
        
        Args:
            tiff_path: Path to the TIFF file.
            
        Returns:
            Metadata dictionary.
        """
        tiff_path = Path(tiff_path)
        
        # Check cache first
        if str(tiff_path) in self.metadata_cache:
            return self.metadata_cache[str(tiff_path)]
        
        # Try to load from JSON file
        json_path = tiff_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                self.metadata_cache[str(tiff_path)] = metadata
                return metadata
        
        # Extract from TIFF tags
        try:
            with tifffile.TiffFile(tiff_path) as tif:
                if tif.pages[0].description:
                    metadata = json.loads(tif.pages[0].description)
                    self.metadata_cache[str(tiff_path)] = metadata
                    return metadata
        except Exception as e:
            logger.warning(f"Could not extract metadata from TIFF: {e}")
        
        return {}
    
    def batch_convert(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.nd2"
    ) -> Dict[Path, Path]:
        """Convert multiple ND2 files to TIFF format.
        
        Args:
            input_dir: Directory containing ND2 files.
            output_dir: Output directory for TIFF files. If None, uses input_dir.
            pattern: File pattern to match (default: "*.nd2").
            
        Returns:
            Dictionary mapping input paths to output paths.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all ND2 files
        nd2_files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(nd2_files)} files to convert")
        
        conversions = {}
        for nd2_file in nd2_files:
            try:
                output_file = output_dir / nd2_file.with_suffix('.tiff').name
                output_path, _ = self.nd2_to_tiff(nd2_file, output_file)
                conversions[nd2_file] = output_path
                logger.info(f"Converted: {nd2_file.name} -> {output_path.name}")
            except Exception as e:
                logger.error(f"Failed to convert {nd2_file}: {e}")
                conversions[nd2_file] = None
        
        return conversions