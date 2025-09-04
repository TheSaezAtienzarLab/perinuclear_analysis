"""
Format converter for microscopy image files.
Converts .nd2 files to .ome.tiff while preserving metadata.
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
    
    def nd2_to_ome_tiff(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        save_metadata: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """Convert .nd2 file to .ome.tiff format with metadata preservation.
        
        Args:
            input_path: Path to the input .nd2 file.
            output_path: Path for the output .ome.tiff file. If None, uses same name with .ome.tiff extension.
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
            output_path = input_path.with_suffix('.ome.tiff')
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
        # Get metadata for proper channel extraction
        z_levels = len(reader.metadata.get('z_levels', [1]))
        channels = len(reader.metadata.get('channels', [])) or 1
        
        logger.info(f"Expected Z levels: {z_levels}, Channels: {channels}")
        logger.info(f"Reader axes: {reader.axes}")
        logger.info(f"Reader sizes: {reader.sizes}")
        logger.info(f"Available frames: {len(reader)}")
        logger.info(f"Frame shape: {reader[0].shape}")
        
        # Check if reader has channel iteration capability
        original_iter_axes = getattr(reader, 'iter_axes', [])
        try:
            reader.iter_axes = 'c'
            logger.info(f"Can iterate over channels: True")
        except Exception as e:
            logger.info(f"Can iterate over channels: False ({e})")
            
        try:
            reader.iter_axes = 'z'
            logger.info(f"Can iterate over z: True")
        except Exception as e:
            logger.info(f"Can iterate over z: False ({e})")
            
        # Reset to original
        try:
            reader.iter_axes = original_iter_axes
        except:
            pass
        
        # Manually extract all channels and z-slices to ensure we get all data
        try:
            if 'z' in reader.axes and 'c' in reader.axes:
                # Multi-channel, multi-z - use proper iteration
                logger.info("Using nd2reader iteration approach for multi-channel z-stack")
                
                # Set up iteration over both z and c
                reader.iter_axes = ['z', 'c']
                reader.bundle_axes = ['y', 'x']
                
                logger.info(f"Set iter_axes to {reader.iter_axes}")
                logger.info(f"Reader now has {len(reader)} frames after iteration setup")
                
                try:
                    all_data = []
                    frame_count = 0
                    
                    # Now iterate - should give us all Z×C combinations
                    for frame in reader:
                        all_data.append(frame)
                        frame_count += 1
                        
                        # Debug first few frames
                        if frame_count <= 10:
                            logger.info(f"Iteration frame {frame_count}: shape={frame.shape}, mean={frame.mean():.2f}")
                        
                        # Safety check to avoid infinite loop
                        if frame_count >= z_levels * channels * 2:  # 2x safety margin
                            logger.warning(f"Safety break at {frame_count} frames")
                            break
                    
                    logger.info(f"Iteration collected {len(all_data)} frames")
                    
                    if len(all_data) == z_levels * channels:
                        # Perfect: we have all Z×C frames
                        frame_array = np.array(all_data)  # (Z*C, Y, X)
                        
                        # Reshape to (Z, C, Y, X) then transpose to (Z, Y, X, C)
                        reshaped = frame_array.reshape(z_levels, channels, frame_array.shape[1], frame_array.shape[2])
                        result = np.transpose(reshaped, (0, 2, 3, 1))
                        
                        logger.info(f"Perfect iteration result shape: {result.shape}")
                        
                        # Validate different z-slices and channels
                        if result.shape[0] > 1:
                            z_means = [result[z, :, :, 0].mean() for z in range(min(5, result.shape[0]))]
                            logger.info(f"Z-slice means (ch0): {z_means}")
                            
                        if result.shape[3] > 1:
                            c_means = [result[0, :, :, c].mean() for c in range(result.shape[3])]
                            logger.info(f"Channel means (z0): {c_means}")
                            
                        return result
                        
                    elif len(all_data) == z_levels:
                        # Got z-slices but no channel separation - this is what's happening
                        logger.warning("Got z-slices but no channel separation - trying channel extraction per z")
                        
                        # Try to extract channels for each z-slice manually
                        multichannel_data = []
                        for z in range(z_levels):
                            z_channels = []
                            for c in range(channels):
                                try:
                                    # Set specific coordinates
                                    reader.default_coords = {'z': z, 'c': c, 't': 0}
                                    frame = reader[0]  # Get first frame with these coords
                                    z_channels.append(frame)
                                    
                                    if z == 0:  # Debug first z-slice
                                        logger.info(f"Manual z={z}, c={c}: shape={frame.shape}, mean={frame.mean():.2f}")
                                        
                                except Exception as e:
                                    logger.error(f"Manual extraction failed z={z}, c={c}: {e}")
                                    # Use the first available frame as fallback
                                    z_channels.append(all_data[0] if all_data else np.zeros((1800, 1800)))
                            
                            multichannel_data.append(np.stack(z_channels, axis=-1))
                        
                        result = np.stack(multichannel_data, axis=0)
                        logger.info(f"Manual channel extraction result shape: {result.shape}")
                        
                        # Validate
                        if result.shape[3] > 1:
                            c_means = [result[0, :, :, c].mean() for c in range(result.shape[3])]
                            logger.info(f"Manual channel means (z0): {c_means}")
                            
                        return result
                        
                    else:
                        logger.error(f"Unexpected number of frames from iteration: {len(all_data)}")
                        return np.array(all_data)
                        
                except Exception as e:
                    logger.error(f"Iteration approach failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Last resort fallback
                    logger.info("Using basic fallback")
                    frame = reader[0]
                    return frame[np.newaxis, ..., np.newaxis]
                
            elif 'c' in reader.axes:
                # Multi-channel, single z
                all_data = []
                for c in range(channels):
                    reader.default_coords['c'] = c
                    frame = reader[0]
                    all_data.append(frame)
                
                result = np.stack(all_data, axis=-1)[np.newaxis, ...]  # Add Z dimension
                logger.info(f"Extracted shape: {result.shape}")
                return result
                
            elif 'z' in reader.axes:
                # Single channel, multi-z
                all_data = []
                for z in range(z_levels):
                    reader.default_coords['z'] = z
                    frame = reader[0]
                    all_data.append(frame)
                
                result = np.stack(all_data, axis=0)[..., np.newaxis]  # Add C dimension
                logger.info(f"Extracted shape: {result.shape}")
                return result
                
            else:
                # Single channel, single z
                frame = reader[0]
                result = frame[np.newaxis, ..., np.newaxis]
                logger.info(f"Extracted shape: {result.shape}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to read ND2 data: {e}")
            raise ValueError(f"Could not read ND2 data: {e}")
    
    def _save_as_tiff(
        self, 
        image_data: np.ndarray, 
        output_path: Path, 
        metadata: Dict[str, Any]
    ) -> None:
        """Save image data as OME-TIFF with metadata.
        
        Args:
            image_data: Image array to save.
            output_path: Output file path.
            metadata: Metadata dictionary to embed.
        """
        # Prepare OME-TIFF metadata - match actual array shape
        axes_mapping = {
            4: 'ZYXC',  # Z-slices, Y, X, Channels (actual shape from nd2reader)
            3: 'ZYX'    # Z-slices, Y, X (single channel)
        }
        axes = axes_mapping.get(image_data.ndim, 'YX')
        
        # Get channel names properly
        channel_names = metadata.get('channel_names', [])
        if not channel_names:
            # Fallback to generic names
            num_channels = image_data.shape[-1] if image_data.ndim > 3 else 1
            channel_names = [f'Channel_{i}' for i in range(num_channels)]
        
        ome_metadata = {
            'axes': axes,
        }
        
        # Add channel information properly for OME-TIFF
        if len(channel_names) > 1:
            ome_metadata['Channel'] = [{'Name': name} for name in channel_names]
        elif len(channel_names) == 1:
            ome_metadata['Channel'] = {'Name': channel_names[0]}
        
        # Add pixel size information if available
        if metadata.get('pixel_size_um'):
            ome_metadata['PhysicalSizeX'] = metadata['pixel_size_um']
            ome_metadata['PhysicalSizeY'] = metadata['pixel_size_um']
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
        
        # Add z-spacing if available
        if metadata.get('pixel_info', {}).get('voxel_size_z'):
            ome_metadata['PhysicalSizeZ'] = metadata['pixel_info']['voxel_size_z']
            ome_metadata['PhysicalSizeZUnit'] = 'µm'
        
        # Debug logging
        logger.info(f"Saving image with shape: {image_data.shape}, axes: {axes}")
        logger.info(f"OME metadata: {ome_metadata}")
        
        # Save as OME-TIFF with explicit axes - this is critical for multichannel
        try:
            logger.info(f"Attempting OME-TIFF with explicit axes and metadata")
            
            # Create proper OME metadata with explicit axes
            ome_metadata = {
                'axes': 'ZYXC'  # Be explicit about axes interpretation
            }
            
            # Add physical pixel sizes
            if metadata.get('pixel_size_um'):
                ome_metadata['PhysicalSizeX'] = metadata['pixel_size_um']
                ome_metadata['PhysicalSizeY'] = metadata['pixel_size_um'] 
                ome_metadata['PhysicalSizeXUnit'] = 'µm'
                ome_metadata['PhysicalSizeYUnit'] = 'µm'
            
            tifffile.imwrite(
                output_path,
                image_data,
                ome=True,
                metadata=ome_metadata,
                compression='lzw'
            )
        except Exception as e:
            logger.error(f"OME-TIFF save failed: {e}, trying without OME metadata")
            # Fallback: save as regular TIFF with JSON description
            # Use LZW compression for better ImageJ compatibility
            tifffile.imwrite(
                output_path,
                image_data,
                description=json.dumps(metadata, indent=2, default=str),
                compression='lzw'
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
                output_file = output_dir / nd2_file.with_suffix('.ome.tiff').name
                output_path, _ = self.nd2_to_ome_tiff(nd2_file, output_file)
                conversions[nd2_file] = output_path
                logger.info(f"Converted: {nd2_file.name} -> {output_path.name}")
            except Exception as e:
                logger.error(f"Failed to convert {nd2_file}: {e}")
                conversions[nd2_file] = None
        
        return conversions