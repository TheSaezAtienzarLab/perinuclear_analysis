"""
Configuration management for the perinuclear analysis module.
Phase-aware configuration with progressive feature enablement.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json


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
    channel_colors: Dict[str, str] = field(default_factory=lambda: {
        'DAPI': 'blue',
        'GFP': 'green',
        'RFP': 'red'
    })
    
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


@dataclass
class Config:
    """Main configuration class that combines all phase configurations."""
    
    # Phase configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    phase5: Phase5Config = field(default_factory=Phase5Config)
    
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
        return {
            'phase1': self.phase1.__dict__,
            'phase2': self.phase2.__dict__,
            'phase3': self.phase3.__dict__,
            'phase4': self.phase4.__dict__,
            'phase5': self.phase5.__dict__,
            'enabled_phases': self.enabled_phases,
            'project_name': self.project_name,
            'output_dir': str(self.output_dir),
            'temp_dir': str(self.temp_dir),
            'random_seed': self.random_seed,
            'ensure_reproducibility': self.ensure_reproducibility,
        }
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'Config':
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file.
            
        Returns:
            Config: Loaded configuration object.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load phase configurations
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5']:
            if phase in data:
                phase_class = globals()[f'{phase.capitalize()}Config']
                setattr(config, phase, phase_class(**data[phase]))
        
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