# Perinuclear Analysis Module

A comprehensive Python module for analyzing subcellular localization changes in microscopy images, with specific focus on perinuclear signal detection and quantification.

## Overview

This module processes .nd2 microscopy images with .tif masks to study signal distribution in different cellular regions, particularly changes in subcellular localization after treatment. The analysis includes:

- Maximum Intensity Projection (MIP) creation from z-stacks
- Cell and nuclei detection using state-of-the-art segmentation models
- Concentric ring analysis (5μm and 10μm from nuclei)
- Signal quantification in different cellular regions
- Statistical analysis and visualization

## Phased Implementation Approach

The module is implemented in 5 distinct phases, allowing incremental testing and validation:

### Phase 1: Infrastructure & Image Loading ✓
- Basic package structure and configuration
- .nd2 and .tif file loading
- Metadata extraction and validation
- Pixel calibration and coordinate management

### Phase 2: MIP Creation & Basic Visualization
- Maximum Intensity Projection generation
- Multi-channel handling
- Basic visualization tools
- Quality control metrics

### Phase 3: Cell & Nuclei Segmentation
- Cell detection using Cellpose
- Nuclei detection from DAPI signal
- Area-based filtering
- Segmentation validation

### Phase 4: Ring Analysis
- 5μm perinuclear ring creation
- 10μm ring with exclusion zone
- Pixel-to-micron conversions
- Boundary management

### Phase 5: Signal Quantification & Complete Pipeline
- Regional signal quantification
- Statistical analysis
- Complete workflow integration
- Advanced visualizations

## Installation

### Basic Installation (Phase 1)
```bash
pip install -r requirements.txt
```

### Phase-Specific Installation
```bash
# Install with phase-specific dependencies
pip install .[phase1]  # Basic infrastructure
pip install .[phase2]  # MIP creation
pip install .[phase3]  # Segmentation
pip install .[phase4]  # Ring analysis
pip install .[phase5]  # Complete pipeline
```

### Development Installation
```bash
pip install -e .[dev]
```

## Quick Start

### Phase 1: Loading Images
```python
from perinuclear_analysis import ImageLoader

# Load .nd2 microscopy image
loader = ImageLoader()
image_data = loader.load_nd2("path/to/image.nd2")
print(f"Image shape: {image_data.shape}")
print(f"Pixel size: {loader.get_pixel_size()} μm")

# Load .tif mask
mask = loader.load_tif_mask("path/to/mask.tif")
```

### Phase 2: Creating MIP
```python
from perinuclear_analysis import MIPCreator

# Create Maximum Intensity Projection
mip_creator = MIPCreator()
mip = mip_creator.create_mip(image_data, method='max')

# Visualize MIP
mip_creator.visualize_mip(mip, channels=['DAPI', 'GFP', 'RFP'])
```

### Phase 3: Segmentation
```python
from perinuclear_analysis import CellSegmenter, NucleiDetector

# Segment cells
cell_segmenter = CellSegmenter()
cell_masks = cell_segmenter.segment(mip, min_area=100)

# Detect nuclei
nuclei_detector = NucleiDetector()
nuclei_masks = nuclei_detector.detect(mip[:,:,0])  # DAPI channel
```

### Phase 4: Ring Analysis
```python
from perinuclear_analysis import RingAnalyzer

# Create concentric rings
ring_analyzer = RingAnalyzer(pixel_size=0.325)  # μm/pixel
rings_5um = ring_analyzer.create_5um_rings(nuclei_masks)
rings_10um = ring_analyzer.create_10um_rings(nuclei_masks, cell_masks)
```

### Phase 5: Complete Analysis
```python
from perinuclear_analysis import PerinuclearAnalyzer

# Run complete analysis pipeline
analyzer = PerinuclearAnalyzer()
results = analyzer.analyze(
    nd2_path="path/to/image.nd2",
    mask_path="path/to/mask.tif",
    output_dir="results/"
)

# Access quantification results
print(results.get_statistics())
analyzer.plot_results()
```

## Workflow Overview

```
Input Files (.nd2, .tif)
        ↓
[Phase 1] Load & Validate
        ↓
[Phase 2] Create MIP
        ↓
[Phase 3] Segment Cells & Nuclei
        ↓
[Phase 4] Generate Ring Masks
        ↓
[Phase 5] Quantify Signals
        ↓
Statistical Analysis & Visualization
```

## Testing

Each phase has its own test suite:

```bash
# Test specific phase
pytest tests/test_phase1.py -v
pytest tests/test_phase2.py -v
pytest tests/test_phase3.py -v
pytest tests/test_phase4.py -v
pytest tests/test_phase5.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=perinuclear_analysis
```

## Examples

See the `examples/` directory for phase-specific example scripts:

- `phase1_image_loading.py` - Basic image loading and metadata extraction
- `phase2_mip_creation.py` - MIP generation and visualization
- `phase3_segmentation.py` - Cell and nuclei segmentation
- `phase4_ring_analysis.py` - Ring mask creation and validation
- `phase5_complete_analysis.py` - Full pipeline demonstration

## Output Structure

```
results/
├── mip/                    # Maximum intensity projections
├── segmentation/           # Cell and nuclei masks
├── ring_masks/            # Concentric ring masks
├── quantification/        # Signal measurements
├── statistics/            # Statistical analysis results
└── visualizations/        # Plots and figures
```

## Configuration

Key parameters can be configured in `perinuclear_analysis/config.py`:

- Pixel size calibration
- Segmentation thresholds
- Ring distances (5μm, 10μm)
- Minimum cell area
- Background correction methods
- Output formats

## Requirements

- Python 3.8+
- nd2reader for .nd2 file support
- scikit-image for image processing
- numpy for numerical operations
- matplotlib for visualization
- cellpose for cell segmentation (Phase 3+)
- pandas for data management (Phase 5)

## Citation

If you use this module in your research, please cite:
```
```

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions, please open an issue on GitHub or contact the maintainers.