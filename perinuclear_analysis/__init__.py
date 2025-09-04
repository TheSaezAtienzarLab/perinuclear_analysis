"""
Perinuclear Analysis Module

A phased implementation for analyzing subcellular localization in microscopy images.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

import warnings
from typing import Dict, Any

# Phase status tracking
PHASE_STATUS = {
    "phase1": True,   # Infrastructure & Image Loading
    "phase2": True,   # MIP Creation & Basic Visualization
    "phase3": True,   # Cell & Nuclei Segmentation
    "phase4": True,   # Ring Analysis
    "phase5": True,   # Signal Quantification & Complete Pipeline
}

# Phase 1 imports - Always available
from .core import Config, Phase1Config
from .data_processing import ImageLoader
from .core import (
    validate_file_path,
    get_pixel_size_from_metadata,
    convert_pixels_to_microns,
    convert_microns_to_pixels,
    get_fluorophore_color,
    create_channel_color_mapping,
    get_colormap_from_fluorophore,
)

# Phase 2 imports
if PHASE_STATUS["phase2"]:
    from .data_processing import MIPCreator
    try:
        from .mip_preprocessor import MIPPreprocessor
    except ImportError:
        warnings.warn("MIPPreprocessor not available yet")
        MIPPreprocessor = None
    from .visualization import Visualizer, plot_mip, plot_channels

# Phase 3 imports
if PHASE_STATUS["phase3"]:
    try:
        from .analysis import CellSegmenter, NucleiDetector
    except ImportError as e:
        warnings.warn(
            f"Phase 3 dependencies not installed: {e}\n"
            "Install with: pip install .[phase3]"
        )
        PHASE_STATUS["phase3"] = False

# Phase 4 imports
if PHASE_STATUS["phase4"]:
    try:
        from .ring_analysis import RingAnalyzer
    except ImportError as e:
        warnings.warn(
            f"Phase 4 dependencies not installed: {e}\n"
            "Install with: pip install .[phase4]"
        )
        PHASE_STATUS["phase4"] = False
        RingAnalyzer = None

# Phase 5 imports
if PHASE_STATUS["phase5"]:
    try:
        from .signal_quantification import SignalQuantifier
        from .core import PerinuclearAnalyzer
    except ImportError as e:
        warnings.warn(
            f"Phase 5 dependencies not installed: {e}\n"
            "Install with: pip install .[phase5]"
        )
        PHASE_STATUS["phase5"] = False
        SignalQuantifier = None
        PerinuclearAnalyzer = None


def check_phase_status() -> Dict[str, bool]:
    """Check which phases are currently available.
    
    Returns:
        Dict[str, bool]: Dictionary indicating availability of each phase.
    """
    return PHASE_STATUS.copy()


def get_phase_info(phase: str) -> Dict[str, Any]:
    """Get information about a specific phase.
    
    Args:
        phase: Phase identifier (e.g., 'phase1', 'phase2', etc.)
        
    Returns:
        Dict[str, Any]: Information about the phase including status and features.
    """
    phase_info = {
        "phase1": {
            "name": "Infrastructure & Image Loading",
            "status": PHASE_STATUS.get("phase1", False),
            "features": [
                "ND2 file loading",
                "TIF mask loading",
                "Metadata extraction",
                "Pixel calibration",
            ],
            "modules": ["config", "image_loader", "utils"],
        },
        "phase2": {
            "name": "MIP Creation & Basic Visualization",
            "status": PHASE_STATUS.get("phase2", False),
            "features": [
                "Maximum Intensity Projection",
                "Multi-channel handling",
                "Basic visualization",
                "Quality metrics",
            ],
            "modules": ["mip_creator", "visualization"],
        },
        "phase3": {
            "name": "Cell & Nuclei Segmentation",
            "status": PHASE_STATUS.get("phase3", False),
            "features": [
                "Cell detection with Cellpose",
                "Nuclei detection from DAPI",
                "Area-based filtering",
                "Segmentation validation",
            ],
            "modules": ["cell_segmentation", "nuclei_detection"],
        },
        "phase4": {
            "name": "Ring Analysis",
            "status": PHASE_STATUS.get("phase4", False),
            "features": [
                "5μm perinuclear rings",
                "10μm rings with exclusion zone",
                "Pixel-to-micron conversion",
                "Boundary management",
            ],
            "modules": ["ring_analysis"],
        },
        "phase5": {
            "name": "Signal Quantification & Complete Pipeline",
            "status": PHASE_STATUS.get("phase5", False),
            "features": [
                "Regional signal quantification",
                "Statistical analysis",
                "Complete workflow integration",
                "Advanced visualizations",
            ],
            "modules": ["signal_quantification", "core"],
        },
    }
    
    if phase not in phase_info:
        raise ValueError(f"Unknown phase: {phase}. Valid phases: {list(phase_info.keys())}")
    
    return phase_info[phase]


# Convenience function for phase validation
def ensure_phase_available(phase: str) -> None:
    """Ensure that a specific phase is available.
    
    Args:
        phase: Phase identifier to check.
        
    Raises:
        RuntimeError: If the phase is not available.
    """
    if not PHASE_STATUS.get(phase, False):
        info = get_phase_info(phase)
        raise RuntimeError(
            f"Phase '{phase}' ({info['name']}) is not available.\n"
            f"Install with: pip install .[{phase}]"
        )


# Export all public components
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Phase management
    "check_phase_status",
    "get_phase_info",
    "ensure_phase_available",
    "PHASE_STATUS",
    
    # Phase 1 - Always available
    "Config",
    "Phase1Config",
    "ImageLoader",
    "validate_file_path",
    "get_pixel_size_from_metadata",
    "convert_pixels_to_microns",
    "convert_microns_to_pixels",
    "get_fluorophore_color",
    "create_channel_color_mapping",
    "get_colormap_from_fluorophore",
]

# Add phase-specific exports
if PHASE_STATUS["phase2"]:
    phase2_exports = ["MIPCreator", "Visualizer", "plot_mip", "plot_channels"]
    if MIPPreprocessor is not None:
        phase2_exports.append("MIPPreprocessor")
    __all__.extend(phase2_exports)

if PHASE_STATUS["phase3"]:
    __all__.extend([
        "CellSegmenter",
        "NucleiDetector",
    ])

if PHASE_STATUS["phase4"] and RingAnalyzer is not None:
    __all__.extend([
        "RingAnalyzer",
    ])

if PHASE_STATUS["phase5"]:
    phase5_exports = []
    if SignalQuantifier is not None:
        phase5_exports.append("SignalQuantifier")
    if PerinuclearAnalyzer is not None:
        phase5_exports.append("PerinuclearAnalyzer")
    __all__.extend(phase5_exports)