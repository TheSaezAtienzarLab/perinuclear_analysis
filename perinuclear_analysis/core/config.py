"""
Configuration management for the perinuclear analysis module.
Phase-aware configuration with progressive feature enablement.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import yaml


@dataclass
class Phase1Config:
    """Phase 1: Basic configuration for image loading and infrastructure."""
    
    # File handling
    supported_image_formats: List[str] = field(default_factory=lambda: ['.nd2', '.tif', '.tiff'])
    max_file_size_gb: float = 10.0
    
    # Image properties (no defaults - must come from metadata)
    
    # Validation
    min_image_dimension: int = 100  # Minimum width/height in pixels
    max_image_dimension: int = 10000  # Maximum width/height in pixels
    
    # Metadata handling
    extract_all_metadata: bool = True
    required_metadata_fields: List[str] = field(default_factory=lambda: ['pixel_size', 'channels'])
    
    # Logging
    verbose: bool = True
    log_level: str = 'INFO'


@dataclass
class Phase2Config:
    """Phase 2: Configuration for MIP creation and basic visualization."""
    
    # MIP creation
    projection_method: str = 'max'  # 'max', 'mean', 'sum'
    z_range: Optional[Tuple[int, int]] = None  # None means use all z-slices
    
    # Quality metrics
    calculate_quality_metrics: bool = True
    quality_threshold: float = 0.7
    
    # Visualization
    figure_dpi: int = 100
    default_colormap: str = 'viridis'
    # channel_colors will be automatically generated from actual channel names
    
    # Output
    save_intermediate_results: bool = True
    output_format: str = 'png'


@dataclass
class Phase3Config:
    """Phase 3: Configuration for cell and nuclei segmentation."""
    
    # Cell segmentation
    cell_model_type: str = 'cyto2'  # Cellpose model type
    cell_diameter: Optional[float] = None  # None means auto-estimate
    cell_flow_threshold: float = 0.4
    cell_cellprob_threshold: float = 0.0
    min_cell_area_pixels: int = 100  # Minimum cell area filter
    
    # Nuclei detection
    nuclei_model_type: str = 'nuclei'  # Cellpose model for nuclei
    nuclei_diameter: Optional[float] = None
    nuclei_flow_threshold: float = 0.4
    nuclei_cellprob_threshold: float = 0.0
    min_nuclei_area_pixels: int = 50
    
    # Association
    max_nuclei_per_cell: int = 1  # For quality control
    require_nuclei_in_cell: bool = True
    
    # GPU acceleration
    use_gpu: bool = False
    gpu_device: int = 0


@dataclass
class Phase4Config:
    """Phase 4: Configuration for ring analysis."""
    
    # Ring parameters (in micrometers)
    inner_ring_distance_um: float = 5.0  # Distance from nuclei edge
    outer_ring_distance_um: float = 10.0  # Distance from nuclei edge
    exclusion_zone_start_um: float = 5.0  # Start of exclusion zone
    exclusion_zone_end_um: float = 10.0  # End of exclusion zone
    
    # Ring generation
    use_distance_transform: bool = True
    smooth_ring_boundaries: bool = True
    smoothing_sigma: float = 1.0
    
    # Boundary handling
    respect_cell_boundaries: bool = True
    handle_edge_cells: str = 'exclude'  # 'exclude', 'partial', 'include'
    min_ring_coverage: float = 0.5  # Minimum fraction of ring that must be within cell
    
    # Validation
    validate_ring_geometry: bool = True
    visualize_rings: bool = True


@dataclass
class Phase5Config:
    """Phase 5: Configuration for signal quantification and complete pipeline."""
    
    # Signal quantification
    background_correction_method: str = 'local'  # 'none', 'global', 'local', 'rolling_ball'
    background_percentile: float = 5.0
    
    # Intensity measurements
    measurements: List[str] = field(default_factory=lambda: [
        'mean', 'median', 'sum', 'std', 'min', 'max'
    ])
    
    # Normalization
    normalize_intensities: bool = True
    normalization_method: str = 'per_cell'  # 'per_cell', 'global', 'z_score'
    
    # Statistical analysis
    calculate_ratios: bool = True
    ratio_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('perinuclear_5um', 'peripheral_10um'),
        ('nuclear', 'cytoplasmic'),
    ])
    
    # Batch processing
    batch_size: int = 10
    parallel_processing: bool = False
    n_workers: int = 4
    
    # Output
    save_individual_results: bool = True
    save_summary_statistics: bool = True
    output_formats: List[str] = field(default_factory=lambda: ['csv', 'xlsx', 'json'])
    
    # Visualization
    generate_plots: bool = True
    plot_types: List[str] = field(default_factory=lambda: [
        'violin', 'box', 'scatter', 'heatmap'
    ])


# Preprocessing Configuration Classes

@dataclass
class BackgroundSubtractionConfig:
    """Configuration for background subtraction preprocessing."""
    
    # Method selection
    method: str = 'rolling_ball'  # 'rolling_ball', 'gaussian', 'morphological'
    
    # Rolling ball parameters
    rolling_ball_radius_dapi: int = 50  # pixels
    rolling_ball_radius_phalloidin: int = 20  # pixels
    rolling_ball_radius_protein: int = 30  # pixels
    
    # Gaussian background parameters
    gaussian_sigma_dapi: float = 25.0
    gaussian_sigma_phalloidin: float = 10.0
    gaussian_sigma_protein: float = 15.0
    
    # Output handling
    clip_negative_values: bool = True
    normalize_output: bool = False


@dataclass
class DenoisingConfig:
    """Configuration for channel-specific denoising."""
    
    # DAPI denoising
    dapi_method: str = 'nl_means'  # 'nl_means', 'bilateral', 'gaussian'
    dapi_nl_means_patch_size: int = 7
    dapi_nl_means_patch_distance: int = 13
    dapi_nl_means_h: float = 0.1
    
    # Phalloidin denoising
    phalloidin_method: str = 'tv_chambolle'  # 'tv_chambolle', 'anisotropic', 'bilateral'
    phalloidin_tv_weight: float = 0.1
    phalloidin_tv_iterations: int = 100
    
    # Protein marker denoising
    protein_method: str = 'bilateral'  # 'bilateral', 'gaussian', 'nl_means'
    protein_bilateral_sigma_color: float = 0.1
    protein_bilateral_sigma_spatial: float = 2.0
    
    # Quality preservation
    preserve_edges: bool = True
    edge_threshold: float = 0.1


@dataclass
class DeconvolutionConfig:
    """Configuration for 3D deconvolution."""
    
    # Algorithm parameters
    method: str = 'richardson_lucy_tv'  # 'richardson_lucy', 'richardson_lucy_tv', 'wiener'
    max_iterations: int = 20
    convergence_threshold: float = 1e-4
    
    # Richardson-Lucy with Total Variation
    tv_regularization_weight: float = 0.002
    auto_stopping: bool = True
    stopping_patience: int = 5
    
    # Performance settings
    use_gpu: bool = False  # Set to True if cuCIM is available
    chunk_size: Tuple[int, int, int] = (256, 256, 16)  # For memory management
    
    # Quality control
    calculate_metrics: bool = True
    save_iterations: bool = False  # For debugging


@dataclass
class QualityControlConfig:
    """Configuration for quality control metrics and thresholds."""
    
    # Signal-to-noise ratio
    snr_threshold: float = 10.0
    snr_background_percentile: float = 5.0
    
    # Focus quality
    focus_method: str = 'power_log_log'  # 'power_log_log', 'laplacian_var', 'brenner'
    focus_threshold: float = 0.7
    
    # Field uniformity
    uniformity_cv_threshold: float = 5.0  # Coefficient of variation percentage
    uniformity_grid_size: int = 8  # For CLAHE-like analysis
    
    # Dynamic range
    max_saturated_pixels_percent: float = 1.0
    min_dynamic_range_bits: int = 8
    
    # Photobleaching detection
    check_photobleaching: bool = True
    bleaching_threshold_percent: float = 10.0
    
    # Quality gates
    enforce_quality_gates: bool = True
    fail_on_quality: bool = False  # Whether to stop processing on quality failure


@dataclass 
class ChannelProcessingConfig:
    """Configuration for channel-specific processing parameters."""
    
    # Channel identification
    dapi_channel_names: List[str] = field(default_factory=lambda: ['DAPI', 'Hoechst', 'DAPI - DAPI'])
    phalloidin_channel_names: List[str] = field(default_factory=lambda: ['Phalloidin', 'Actin', 'AF488'])
    lamp1_channel_names: List[str] = field(default_factory=lambda: ['LAMP1', 'Lysosome'])
    protein_channel_names: List[str] = field(default_factory=lambda: ['GAL3', 'ALIX', 'Protein'])
    
    # DAPI processing
    dapi_gamma_correction: float = 1.0  # 0.8-1.2 range
    dapi_segmentation_blur_sigma: float = 3.0
    dapi_overexposure_strategy: bool = True  # For mouse samples
    
    # Phalloidin processing
    phalloidin_edge_enhancement: bool = True
    phalloidin_clahe_clip_limit: float = 0.03
    phalloidin_clahe_tile_grid: Tuple[int, int] = (8, 8)
    phalloidin_unsharp_radius: float = 1.5
    phalloidin_unsharp_strength: float = 0.75
    
    # LAMP1 processing
    lamp1_log_sigma_range: Tuple[float, float] = (0.5, 2.0)  # in microns
    lamp1_vesicle_size_range: Tuple[float, float] = (0.5, 2.0)  # in microns
    lamp1_percentile_threshold: float = 95.0
    lamp1_circularity_threshold: float = 0.7
    
    # Multi-scale protein processing
    protein_scale_range: Tuple[float, float] = (0.5, 5.0)  # in microns
    protein_vesicle_detection: bool = True
    protein_diffuse_detection: bool = True
    protein_normalization_method: str = 'median'  # 'median', 'robust', 'percentile'


@dataclass
class PreprocessingConfig:
    """Master configuration for image preprocessing pipeline."""
    
    # Sub-configurations
    background_subtraction: BackgroundSubtractionConfig = field(default_factory=BackgroundSubtractionConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    channel_processing: ChannelProcessingConfig = field(default_factory=ChannelProcessingConfig)
    
    # Pipeline control
    processing_order: List[str] = field(default_factory=lambda: [
        'background_subtraction', 'denoising', 'deconvolution'
    ])
    skip_steps: List[str] = field(default_factory=list)
    
    # Memory management (optimized for M3 Pro)
    max_memory_usage_gb: float = 14.0  # Conservative for 18GB system
    chunk_processing: bool = True
    parallel_channels: bool = True
    max_parallel_channels: int = 2
    
    # Output control
    save_intermediate_steps: bool = False
    intermediate_format: str = 'tiff'
    compression: Optional[str] = 'lzw'
    
    # Quality control integration
    run_quality_checks: bool = True
    quality_report_format: str = 'json'
    
    def get_channel_config(self, channel_name: str) -> Dict[str, Any]:
        """Get processing parameters for a specific channel."""
        channel_name_lower = channel_name.lower()
        
        if any(name.lower() in channel_name_lower for name in self.channel_processing.dapi_channel_names):
            return self._get_dapi_config()
        elif any(name.lower() in channel_name_lower for name in self.channel_processing.phalloidin_channel_names):
            return self._get_phalloidin_config()
        elif any(name.lower() in channel_name_lower for name in self.channel_processing.lamp1_channel_names):
            return self._get_lamp1_config()
        else:
            return self._get_protein_config()
    
    def _get_dapi_config(self) -> Dict[str, Any]:
        """Get DAPI-specific processing parameters."""
        return {
            'background_method': 'rolling_ball',
            'background_radius': self.background_subtraction.rolling_ball_radius_dapi,
            'denoise_method': self.denoising.dapi_method,
            'denoise_params': {
                'patch_size': self.denoising.dapi_nl_means_patch_size,
                'patch_distance': self.denoising.dapi_nl_means_patch_distance,
                'h': self.denoising.dapi_nl_means_h,
            },
            'gamma_correction': self.channel_processing.dapi_gamma_correction,
            'segmentation_blur': self.channel_processing.dapi_segmentation_blur_sigma,
        }
    
    def _get_phalloidin_config(self) -> Dict[str, Any]:
        """Get Phalloidin-specific processing parameters."""
        return {
            'background_method': 'rolling_ball', 
            'background_radius': self.background_subtraction.rolling_ball_radius_phalloidin,
            'denoise_method': self.denoising.phalloidin_method,
            'denoise_params': {
                'weight': self.denoising.phalloidin_tv_weight,
                'iterations': self.denoising.phalloidin_tv_iterations,
            },
            'edge_enhancement': self.channel_processing.phalloidin_edge_enhancement,
            'clahe_params': {
                'clip_limit': self.channel_processing.phalloidin_clahe_clip_limit,
                'tile_grid': self.channel_processing.phalloidin_clahe_tile_grid,
            },
            'unsharp_params': {
                'radius': self.channel_processing.phalloidin_unsharp_radius,
                'strength': self.channel_processing.phalloidin_unsharp_strength,
            },
        }
    
    def _get_lamp1_config(self) -> Dict[str, Any]:
        """Get LAMP1-specific processing parameters."""
        return {
            'background_method': 'rolling_ball',
            'background_radius': self.background_subtraction.rolling_ball_radius_protein,
            'denoise_method': self.denoising.protein_method,
            'denoise_params': {
                'sigma_color': self.denoising.protein_bilateral_sigma_color,
                'sigma_spatial': self.denoising.protein_bilateral_sigma_spatial,
            },
            'log_sigma_range': self.channel_processing.lamp1_log_sigma_range,
            'vesicle_size_range': self.channel_processing.lamp1_vesicle_size_range,
            'percentile_threshold': self.channel_processing.lamp1_percentile_threshold,
            'circularity_threshold': self.channel_processing.lamp1_circularity_threshold,
        }
    
    def _get_protein_config(self) -> Dict[str, Any]:
        """Get generic protein marker processing parameters."""
        return {
            'background_method': 'rolling_ball',
            'background_radius': self.background_subtraction.rolling_ball_radius_protein,
            'denoise_method': self.denoising.protein_method,
            'denoise_params': {
                'sigma_color': self.denoising.protein_bilateral_sigma_color,
                'sigma_spatial': self.denoising.protein_bilateral_sigma_spatial,
            },
            'scale_range': self.channel_processing.protein_scale_range,
            'vesicle_detection': self.channel_processing.protein_vesicle_detection,
            'diffuse_detection': self.channel_processing.protein_diffuse_detection,
            'normalization_method': self.channel_processing.protein_normalization_method,
        }


@dataclass
class Config:
    """Main configuration class that combines all phase configurations."""
    
    # Phase configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    phase5: Phase5Config = field(default_factory=Phase5Config)
    
    # Preprocessing configuration
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Phase enablement flags
    enabled_phases: List[str] = field(default_factory=lambda: ['phase1'])
    
    # Global settings
    project_name: str = 'perinuclear_analysis'
    output_dir: Path = field(default_factory=lambda: Path('results'))
    temp_dir: Path = field(default_factory=lambda: Path('temp'))
    
    # Reproducibility
    random_seed: int = 42
    ensure_reproducibility: bool = True
    
    def enable_phase(self, phase: str) -> None:
        """Enable a specific phase.
        
        Args:
            phase: Phase identifier (e.g., 'phase2', 'phase3')
        """
        if phase not in self.enabled_phases:
            # Ensure prerequisites are met
            phase_num = int(phase[-1])
            for i in range(1, phase_num):
                prereq = f'phase{i}'
                if prereq not in self.enabled_phases:
                    raise ValueError(f"Cannot enable {phase} without {prereq}")
            self.enabled_phases.append(phase)
    
    def is_phase_enabled(self, phase: str) -> bool:
        """Check if a phase is enabled.
        
        Args:
            phase: Phase identifier to check.
            
        Returns:
            bool: True if phase is enabled.
        """
        return phase in self.enabled_phases
    
    def get_phase_config(self, phase: str) -> Any:
        """Get configuration for a specific phase.
        
        Args:
            phase: Phase identifier.
            
        Returns:
            Phase configuration object.
        """
        if not self.is_phase_enabled(phase):
            raise ValueError(f"Phase {phase} is not enabled")
        
        return getattr(self, phase)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary.
        """
        def config_to_dict(config_obj):
            """Recursively convert dataclass to dictionary."""
            result = {}
            for field_name, field_value in config_obj.__dict__.items():
                if hasattr(field_value, '__dict__'):
                    result[field_name] = config_to_dict(field_value)
                elif isinstance(field_value, Path):
                    result[field_name] = str(field_value)
                else:
                    result[field_name] = field_value
            return result
        
        return {
            'phase1': config_to_dict(self.phase1),
            'phase2': config_to_dict(self.phase2),
            'phase3': config_to_dict(self.phase3),
            'phase4': config_to_dict(self.phase4),
            'phase5': config_to_dict(self.phase5),
            'preprocessing': config_to_dict(self.preprocessing),
            'enabled_phases': self.enabled_phases,
            'project_name': self.project_name,
            'output_dir': str(self.output_dir),
            'temp_dir': str(self.temp_dir),
            'random_seed': self.random_seed,
            'ensure_reproducibility': self.ensure_reproducibility,
        }
    
    def save(self, filepath: Path, format: str = 'auto') -> None:
        """Save configuration to file.
        
        Args:
            filepath: Path to save configuration.
            format: File format ('json', 'yaml', or 'auto' to detect from extension).
        """
        filepath = Path(filepath)
        
        if format == 'auto':
            format = 'yaml' if filepath.suffix.lower() in ['.yml', '.yaml'] else 'json'
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            if format == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'Config':
        """Load configuration from JSON or YAML file.
        
        Args:
            filepath: Path to configuration file.
            
        Returns:
            Config: Loaded configuration object.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        config = cls()
        
        def load_nested_config(config_data: Dict, config_class):
            """Recursively load nested configuration objects."""
            if not isinstance(config_data, dict):
                return config_data
                
            result = {}
            for key, value in config_data.items():
                if isinstance(value, dict) and hasattr(config_class, '__dataclass_fields__'):
                    # Check if this field should be a nested config class
                    field_type = config_class.__dataclass_fields__.get(key)
                    if field_type and hasattr(field_type.type, '__dataclass_fields__'):
                        result[key] = load_nested_config(value, field_type.type)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            return config_class(**result)
        
        # Load phase configurations
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5']:
            if phase in data:
                phase_class = globals()[f'{phase.capitalize()}Config']
                setattr(config, phase, load_nested_config(data[phase], phase_class))
        
        # Load preprocessing configuration
        if 'preprocessing' in data:
            config.preprocessing = load_nested_config(data['preprocessing'], PreprocessingConfig)
        
        # Load other settings
        config.enabled_phases = data.get('enabled_phases', ['phase1'])
        config.project_name = data.get('project_name', 'perinuclear_analysis')
        config.output_dir = Path(data.get('output_dir', 'results'))
        config.temp_dir = Path(data.get('temp_dir', 'temp'))
        config.random_seed = data.get('random_seed', 42)
        config.ensure_reproducibility = data.get('ensure_reproducibility', True)
        
        return config


def create_default_config(phases: List[str] = None) -> Config:
    """Create a default configuration with specified phases enabled.
    
    Args:
        phases: List of phases to enable. If None, only phase1 is enabled.
        
    Returns:
        Config: Default configuration object.
    """
    config = Config()
    
    if phases:
        for phase in phases:
            config.enable_phase(phase)
    
    return config


def create_preprocessing_templates() -> Dict[str, Dict[str, Any]]:
    """Create preprocessing configuration templates for different analysis types.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of template configurations.
    """
    
    # Standard 4-channel template
    standard_4channel = {
        'background_subtraction': {
            'method': 'rolling_ball',
            'rolling_ball_radius_dapi': 50,
            'rolling_ball_radius_phalloidin': 20,
            'rolling_ball_radius_protein': 30,
            'clip_negative_values': True
        },
        'denoising': {
            'dapi_method': 'nl_means',
            'dapi_nl_means_h': 0.1,
            'phalloidin_method': 'tv_chambolle',
            'phalloidin_tv_weight': 0.1,
            'protein_method': 'bilateral',
            'protein_bilateral_sigma_color': 0.1,
            'protein_bilateral_sigma_spatial': 2.0
        },
        'deconvolution': {
            'method': 'richardson_lucy_tv',
            'max_iterations': 20,
            'tv_regularization_weight': 0.002,
            'auto_stopping': True
        },
        'quality_control': {
            'snr_threshold': 10.0,
            'focus_threshold': 0.7,
            'uniformity_cv_threshold': 5.0,
            'enforce_quality_gates': True
        },
        'channel_processing': {
            'dapi_channel_names': ['DAPI', 'Hoechst', 'DAPI - DAPI'],
            'phalloidin_channel_names': ['Phalloidin', 'Actin', 'AF488'],
            'lamp1_channel_names': ['LAMP1', 'Lysosome'],
            'protein_channel_names': ['GAL3', 'ALIX', 'Protein']
        }
    }
    
    # Colocalization-optimized template
    colocalization_optimized = {
        'background_subtraction': {
            'method': 'rolling_ball',
            'rolling_ball_radius_dapi': 50,
            'rolling_ball_radius_phalloidin': 20,
            'rolling_ball_radius_protein': 25,  # Slightly smaller for better quantification
            'clip_negative_values': True
        },
        'denoising': {
            'dapi_method': 'nl_means',
            'dapi_nl_means_h': 0.08,  # More conservative for quantitative accuracy
            'phalloidin_method': 'bilateral',  # More conservative than TV
            'phalloidin_tv_weight': 0.05,
            'protein_method': 'bilateral',
            'protein_bilateral_sigma_color': 0.08,
            'protein_bilateral_sigma_spatial': 1.5
        },
        'deconvolution': {
            'method': 'richardson_lucy_tv',
            'max_iterations': 15,  # Fewer iterations for quantitative work
            'tv_regularization_weight': 0.001,  # Lower regularization
            'auto_stopping': True
        },
        'quality_control': {
            'snr_threshold': 12.0,  # Higher SNR requirement
            'focus_threshold': 0.8,
            'uniformity_cv_threshold': 3.0,  # Stricter uniformity
            'enforce_quality_gates': True,
            'fail_on_quality': True  # Strict quality enforcement
        }
    }
    
    # High-throughput template (optimized for speed and memory)
    high_throughput = {
        'background_subtraction': {
            'method': 'gaussian',  # Faster than rolling ball
            'gaussian_sigma_dapi': 25.0,
            'gaussian_sigma_phalloidin': 10.0,
            'gaussian_sigma_protein': 15.0,
            'clip_negative_values': True
        },
        'denoising': {
            'dapi_method': 'gaussian',  # Fastest option
            'phalloidin_method': 'bilateral',
            'protein_method': 'bilateral',
            'protein_bilateral_sigma_color': 0.15,  # Slightly more aggressive
            'protein_bilateral_sigma_spatial': 3.0
        },
        'deconvolution': {
            'method': 'richardson_lucy',  # Skip TV regularization for speed
            'max_iterations': 10,  # Fewer iterations
            'auto_stopping': False  # Skip convergence checking
        },
        'quality_control': {
            'snr_threshold': 8.0,  # Relaxed thresholds
            'focus_threshold': 0.6,
            'uniformity_cv_threshold': 8.0,
            'enforce_quality_gates': False  # Skip for speed
        },
        'max_memory_usage_gb': 12.0,  # Conservative memory usage
        'chunk_processing': True,
        'parallel_channels': True,
        'max_parallel_channels': 4  # Use more cores
    }
    
    # Quality assessment template (focus on metrics)
    quality_assessment = {
        'background_subtraction': {
            'method': 'rolling_ball',
            'rolling_ball_radius_dapi': 50,
            'rolling_ball_radius_phalloidin': 20,
            'rolling_ball_radius_protein': 30
        },
        'denoising': {
            'dapi_method': 'nl_means',
            'dapi_nl_means_h': 0.05,  # Very conservative
            'phalloidin_method': 'anisotropic',
            'protein_method': 'nl_means'
        },
        'deconvolution': {
            'method': 'richardson_lucy_tv',
            'max_iterations': 30,  # More iterations for quality
            'tv_regularization_weight': 0.003,
            'calculate_metrics': True,
            'save_iterations': True  # For quality analysis
        },
        'quality_control': {
            'snr_threshold': 15.0,  # Very high standards
            'focus_threshold': 0.9,
            'uniformity_cv_threshold': 2.0,
            'check_photobleaching': True,
            'enforce_quality_gates': True,
            'fail_on_quality': True
        },
        'run_quality_checks': True,
        'save_intermediate_steps': True  # For detailed analysis
    }
    
    return {
        'standard_4channel': standard_4channel,
        'colocalization_optimized': colocalization_optimized, 
        'high_throughput': high_throughput,
        'quality_assessment': quality_assessment
    }


def save_preprocessing_template(template_name: str, output_dir: Path = Path('.')):
    """Save a preprocessing template as YAML file.
    
    Args:
        template_name: Name of template ('standard_4channel', 'colocalization_optimized', etc.)
        output_dir: Directory to save the template file.
    """
    templates = create_preprocessing_templates()
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
    
    template_config = PreprocessingConfig(**templates[template_name])
    full_config = Config(preprocessing=template_config)
    
    output_path = output_dir / f"{template_name}_preprocessing_config.yaml"
    full_config.save(output_path, format='yaml')
    
    print(f"Template '{template_name}' saved to {output_path}")


def create_all_preprocessing_templates(output_dir: Path = Path('config_templates')):
    """Create all preprocessing template files.
    
    Args:
        output_dir: Directory to save all template files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    templates = create_preprocessing_templates()
    
    for template_name in templates.keys():
        save_preprocessing_template(template_name, output_dir)
    
    print(f"All preprocessing templates created in {output_dir}")


def load_preprocessing_template(template_name: str, template_dir: Path = Path('config_templates')) -> PreprocessingConfig:
    """Load a preprocessing template configuration.
    
    Args:
        template_name: Name of the template to load.
        template_dir: Directory containing template files.
        
    Returns:
        PreprocessingConfig: Loaded preprocessing configuration.
    """
    template_path = template_dir / f"{template_name}_preprocessing_config.yaml"
    
    if not template_path.exists():
        # Fallback to creating from built-in templates
        templates = create_preprocessing_templates()
        if template_name in templates:
            return PreprocessingConfig(**templates[template_name])
        else:
            raise FileNotFoundError(f"Template file not found: {template_path}")
    
    full_config = Config.load(template_path)
    return full_config.preprocessing