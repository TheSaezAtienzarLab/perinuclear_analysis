# Confocal Microscopy Preprocessing Implementation Tasks

## Project Structure Integration

The preprocessing pipeline will be organized into specific modules within the existing structure:

```
perinuclear_analysis/
├── core/                    # ✅ Existing - shared utilities, config, format_converter
├── data_processing/         # ✅ Existing - image_loader, mip_creator
├── imaging_preprocessing/   # 🆕 New preprocessing modules
│   ├── background_subtraction/
│   ├── denoising/
│   ├── deconvolution/
│   ├── quality_control/
│   ├── channel_processors/
│   └── pipeline_orchestration/
├── analysis/               # ✅ Existing - cell_segmentation, nuclei_detection
└── visualization/          # ✅ Existing - visualization tools
```

## Phase 1: Pre-Development Assessment & Setup

### 1.1 Hardware & Software Environment Assessment
- [ ] **Verify system specifications (M3 Pro MacBook - 18GB)**
  - ✅ RAM sufficient (18GB > 16GB minimum)
  - [ ] Check available storage for intermediate files (recommend 50GB+)
  - [ ] Verify CPU core utilization (M3 Pro - 11 cores)
  - [ ] Test memory pressure during large nd2 loading

- [ ] **Leverage existing infrastructure**
  - ✅ Use existing `ImageLoader` from `data_processing/`
  - ✅ Use existing `Config` system from `core/`
  - ✅ Use existing `utils` and `format_converter` from `core/`
  - [ ] Extend existing config classes for preprocessing parameters

- [ ] **Evaluate sample datasets**
  - [ ] Test with existing nd2 files in the project
  - [ ] Verify channel configurations match existing `ImageLoader`
  - [ ] Test memory usage with current `MIPCreator`
  - [ ] Validate metadata compatibility with existing utilities

### 1.2 Configuration System Integration
- [x] **Extend existing Config classes**
  - [x] Add `PreprocessingConfig` to `core/config.py`
  - [x] Create channel-specific config classes inheriting from existing structure  
  - [x] Integrate with existing phase-based configuration system
  - [x] Add YAML template support to existing config system

- [x] **Created specialized configuration classes:**
  - [x] `BackgroundSubtractionConfig` - Rolling ball, Gaussian, morphological methods
  - [x] `DenoisingConfig` - Channel-specific denoising (NL-means, TV, bilateral)
  - [x] `DeconvolutionConfig` - Richardson-Lucy with Total Variation, GPU support
  - [x] `QualityControlConfig` - SNR, focus, uniformity, photobleaching metrics
  - [x] `ChannelProcessingConfig` - DAPI, Phalloidin, LAMP1, protein-specific parameters

- [x] **Channel-specific parameter management**
  - [x] Intelligent channel detection based on channel names
  - [x] DAPI: γ-correction, segmentation blur, overexposure strategy
  - [x] Phalloidin: edge enhancement, CLAHE, unsharp masking
  - [x] LAMP1: LoG filtering, vesicle detection, percentile thresholds
  - [x] Proteins: multi-scale filtering, vesicular/diffuse detection

- [x] **Memory optimization for M3 Pro (18GB)**
  - [x] Conservative 14GB max usage setting
  - [x] Chunk processing configuration
  - [x] Parallel channel processing with limits
  - [x] Memory-efficient parameter defaults

- [x] **YAML template system implementation**
  - [x] Built-in template creation: `create_preprocessing_templates()`
  - [x] Template saving: `save_preprocessing_template()`
  - [x] Batch template generation: `create_all_preprocessing_templates()`
  - [x] Template loading: `load_preprocessing_template()`

- [x] **Four analysis-specific templates created:**
  - [x] `standard_4channel` - General purpose DAPI/Phalloidin/LAMP1/GAL3
  - [x] `colocalization_optimized` - Conservative parameters for quantitative accuracy
  - [x] `high_throughput` - Speed/memory optimized for batch processing
  - [x] `quality_assessment` - Maximum quality with detailed metrics

- [x] **Enhanced Config class functionality**
  - [x] Recursive dictionary conversion with nested config support
  - [x] YAML/JSON auto-detection based on file extension
  - [x] Nested configuration loading with proper dataclass instantiation
  - [x] Backward compatibility with existing phase system

- [x] **Updated core module exports**
  - [x] All preprocessing config classes exported from `core/__init__.py`
  - [x] Template management functions available package-wide
  - [x] Maintained existing API compatibility

- [ ] **Configuration validation using existing utilities**
  - [ ] Extend existing `validate_file_path` functionality
  - [ ] Use existing metadata validation approaches
  - [ ] Integrate with existing phase management system

## Phase 2: Modular Preprocessing Components

### 2.1 Background Subtraction Module (`imaging_preprocessing/background_subtraction/`)
- [x] **Create `BackgroundSubtractor` class**
  - [x] Extend existing `ImageLoader` functionality for preprocessing
  - [x] Rolling ball algorithm implementation (radius: 50px DAPI, 20px Phalloidin)
  - [x] Gaussian background estimation with configurable sigma
  - [x] Integration with existing `Phase1Config` parameter system

- [x] **Memory-efficient implementation optimized for 3D ome.tiff**
  - [x] Chunk-based z-stack processing for M3 Pro (18GB memory)
  - [x] Automatic optimal chunk size calculation based on image dimensions
  - [x] Memory usage monitoring with conservative 2GB chunk limits
  - [x] Garbage collection after each chunk to free memory

- [x] **3D-specific processing methods implemented:**
  - [x] `_rolling_ball_subtraction_3d()` - Slice-by-slice with chunked processing
  - [x] `_gaussian_subtraction_3d()` - True 3D Gaussian filtering
  - [x] `_morphological_subtraction_3d()` - Chunked morphological operations
  - [x] Background statistics tracking across z-slices

- [x] **Channel-specific parameter integration:**
  - [x] DAPI: 50px radius rolling ball for nuclear regions  
  - [x] Phalloidin: 20px radius to preserve filament structures
  - [x] LAMP1/Proteins: 30px radius for punctate structures
  - [x] Automatic parameter selection based on channel names

- [x] **Advanced features completed:**
  - [x] `process_from_loader()` - Direct integration with ImageLoader
  - [x] `batch_process()` - Multiple channels/images processing
  - [x] `from_preprocessing_config()` - Factory method from config
  - [x] `get_recommended_parameters()` - Channel-specific guidance
  - [x] Comprehensive metadata tracking with processing statistics

### 2.2 Quality Control Module (`imaging_preprocessing/quality_control/`)
- [ ] **Create `QualityController` class**
  - Signal-to-noise ratio calculation (target SNR > 10)
  - Focus quality assessment (Power Log-Log Slope)
  - Field uniformity checks (CV < 5%)
  - Dynamic range utilization monitoring

- [ ] **Integration with existing visualization**
  - Use existing `Visualizer` class from `visualization/`
  - Quality report generation compatible with existing plotting
  - Automated flagging system with existing logging patterns

### 2.3 Denoising Module (`imaging_preprocessing/denoising/`)
- [ ] **Create channel-specific denoisers**
  - `DAPIDenoiser` - Non-local means for nuclear texture
  - `PhalloidinDenoiser` - Anisotropic diffusion for edge preservation  
  - `ProteinMarkerDenoiser` - Bilateral filter for punctate structures
  - Base `Denoiser` class with common functionality

- [ ] **Parameter management**
  - Extend existing config system for denoising parameters
  - Channel-specific parameter inheritance
  - Validation using existing utilities

## Phase 3: Channel Processors Module (`imaging_preprocessing/channel_processors/`)

### 3.1 DAPI Processor (`DAPIProcessor` class)
- [ ] **Implement DAPI-specific processing pipeline**
  - Extend base channel processor class
  - Gamma correction (γ = 0.8-1.2) with parameter optimization
  - Gaussian blur with σ = 3 pixels for segmentation compatibility
  - Integration with existing `NucleiDetector` from `analysis/`

- [ ] **Chromatin texture handling**
  - Overexposure strategy for mouse samples
  - Heterogeneous staining normalization using existing utilities
  - Nuclear boundary preservation validation
  - Use existing pixel calibration functions from `core/utils`

### 3.2 Phalloidin Processor (`PhalloidinProcessor` class)  
- [ ] **Edge-preserving algorithm implementation**
  - Anisotropic diffusion filtering
  - Total variation denoising from denoising module
  - Structure tensor-based enhancement
  - Filament continuity preservation metrics

- [ ] **Contrast enhancement integration**
  - CLAHE implementation (8×8 tile size)
  - Unsharp masking (1-2px radius, 50-100% strength)
  - Integration with existing visualization tools for validation

### 3.3 LAMP1 Processor (`LAMP1Processor` class)
- [ ] **Punctate structure detection**
  - Laplacian of Gaussian filtering using existing pixel-to-micron conversion
  - Morphological top-hat filtering  
  - Size-based vesicle detection (0.5-2 μm) using existing calibration
  - Integration with existing analysis pipeline

- [ ] **Adaptive processing**
  - Percentile-based thresholds (95th percentile)
  - Variable expression level handling
  - Cell-by-cell normalization compatible with existing segmentation

### 3.4 GAL3/ALIX Processor (`ProteinMarkerProcessor` class)
- [ ] **Multi-localization pattern handling** 
  - Multi-scale filtering (0.5-5 μm scales)
  - Vesicular vs diffuse pattern separation
  - Circularity filtering (> 0.7) using existing morphological tools
  - Integration with existing colocalization analysis

- [ ] **Robust statistical processing**
  - Median-based intensity normalization
  - Expression variability handling using existing statistical utilities
  - Multi-pattern detection algorithms

## Phase 4: Deconvolution Module (`imaging_preprocessing/deconvolution/`)

### 4.1 3D Deconvolution Implementation (`Deconvolver` class)
- [ ] **Richardson-Lucy with Total Variation (RLTV)**
  - Standalone deconvolution class for z-stack processing
  - Parameter optimization (λ = 0.002 initial) using config system
  - Iteration control (15-30 with auto-stopping)
  - Memory-efficient 3D processing for M3 Pro (18GB constraint)

- [ ] **Performance optimization**
  - CPU-optimized implementation for 11 cores
  - Chunk-based 3D processing with existing memory patterns
  - Progress monitoring using existing logging
  - Integration with existing quality control metrics

### 4.2 MIPCreator Enhancement (modify existing `data_processing/mip_creator.py`)
- [ ] **Add Smooth Manifold Extraction (SME) to existing MIPCreator**
  - Implement SME as new projection method in current class
  - 2.5D object handling while maintaining existing API
  - Continuity preservation algorithms
  - Maintain compatibility with existing visualization

- [ ] **Add weighted z-projection to MIPCreator**  
  - Focus quality metric calculation
  - Weight computation algorithms integrated into existing methods
  - Signal-to-noise improvement validation
  - Preserve existing MIP functionality

## Phase 5: Colocalization & Quantitative Accuracy

### 5.1 Thresholding & Normalization
- [ ] **Implement Otsu thresholding with pre-filtering**
  - Gaussian pre-filtering (σ = 1)
  - Stability improvements
  - Comparison with Costes method

- [ ] **Add percentile normalization**
  - Channel-wise normalization (1st-99th percentiles)
  - Quantitative relationship preservation
  - Fluorophore brightness compensation

### 5.2 Spectral Unmixing & Crosstalk Correction
- [ ] **Implement crosstalk detection**
  - Single-labeled control analysis
  - Compensation matrix generation
  - < 5% bleed-through validation

- [ ] **Add spectral unmixing pipeline**
  - Linear unmixing implementation
  - Non-negativity constraint enforcement
  - Quality validation metrics

## Phase 6: Performance & Scalability

### 6.1 Parallel Processing Implementation
- [ ] **Multi-core processing setup**
  - Multiprocessing pool implementation
  - Optimal worker count determination
  - Load balancing optimization

- [ ] **GPU acceleration integration**
  - cuCIM integration for deconvolution
  - Asynchronous processing streams
  - CPU-GPU transfer optimization

### 6.2 Caching & Memory Management
- [ ] **Multi-level caching system**
  - In-memory caching for frequent data
  - SSD-based intermediate result caching
  - Cache invalidation strategies

- [ ] **Memory usage optimization**
  - Chunk size optimization
  - Memory monitoring and alerts
  - Garbage collection optimization

## Phase 7: Integration & Testing

### 7.1 Pipeline Integration
- [ ] **Integrate with existing perinuclear_analysis package**
  - Import structure updates
  - Configuration system integration
  - API compatibility maintenance

- [ ] **Workflow orchestration**
  - Nextflow/Snakemake integration consideration
  - Microservices architecture planning
  - Containerization setup

### 7.2 Validation & Testing
- [ ] **Create comprehensive test suite**
  - Unit tests for each preprocessing step
  - Integration tests for full pipeline
  - Performance benchmarks

- [ ] **Reproducibility validation**
  - ICC analysis implementation (target > 0.75)
  - Parameter logging system
  - Environment standardization

### 7.3 Documentation & Examples
- [ ] **Create usage documentation**
  - Configuration examples
  - Best practices guide
  - Troubleshooting documentation

- [ ] **Develop example workflows**
  - Standard 4-channel processing
  - Custom parameter optimization
  - Quality control interpretation

## Phase 5: Pipeline Orchestration Module (`imaging_preprocessing/pipeline_orchestration/`)

### 5.1 Preprocessing Pipeline (`PreprocessingPipeline` class)
- [ ] **Create main orchestration class**
  - Sequential processing: background subtraction → denoising → deconvolution
  - Integration with existing `ImageLoader` and `MIPCreator`
  - Channel-specific processing routing to appropriate processors
  - Memory management and progress tracking

- [ ] **Pipeline configuration management**
  - Extend existing config system for pipeline-wide settings
  - Template-based configurations for different experimental setups
  - Parameter validation and optimization suggestions
  - Integration with existing phase management system

### 5.2 Script Generation for Analysis Types
- [ ] **Create analysis-specific scripts**
  - `standard_4channel_preprocessing.py` - DAPI, Phalloidin, LAMP1, GAL3
  - `colocalization_preprocessing.py` - Optimized for quantitative analysis
  - `high_throughput_preprocessing.py` - Batch processing with memory optimization
  - `quality_assessment_preprocessing.py` - Focus on QC metrics and validation

- [ ] **Script templates and customization**
  - Parameter template generation
  - Custom pipeline creation tools
  - Integration with existing analysis workflows
  - Batch processing capabilities

## Phase 6: Integration & Testing

### 6.1 Package Integration
- [ ] **Update main package structure**
  - Add preprocessing phase to existing `PHASE_STATUS`
  - Update `__init__.py` imports for new modules
  - Maintain backward compatibility with existing API
  - Integration with existing configuration and logging systems

- [ ] **Create preprocessing-specific configurations**
  - Extend existing `Phase1Config`, `Phase2Config` pattern
  - Add `PreprocessingConfig` class with all module parameters
  - YAML template support for different analysis types
  - Parameter inheritance and validation

### 6.2 Testing & Validation
- [ ] **Comprehensive test suite**
  - Unit tests for each preprocessing module
  - Integration tests with existing `ImageLoader` and `MIPCreator`
  - Memory usage testing on M3 Pro architecture
  - Quality metric validation against reference datasets

- [ ] **Performance benchmarking**
  - Processing time measurements for different file sizes
  - Memory usage profiling for 18GB constraint validation
  - CPU utilization optimization for 11-core M3 Pro
  - Comparison with existing processing pipeline

## Pre-Development Checklist

Before starting implementation, verify:

- [ ] **Dataset characteristics understood**
  - File formats and sizes
  - Channel configurations
  - Expected processing volumes
  - Quality requirements

- [ ] **Hardware requirements met**
  - Sufficient RAM and storage
  - GPU availability assessed
  - Processing time expectations set

- [ ] **Software dependencies planned**
  - Version compatibility matrix
  - Optional vs required dependencies
  - Installation complexity assessed

- [ ] **Integration requirements defined**
  - Existing codebase compatibility
  - API design considerations
  - Configuration system requirements

## Success Metrics

- **Quality**: SNR > 10, CV < 5%, ICC > 0.75
- **Performance**: 10-25x speedup with GPU acceleration
- **Accuracy**: < 5% crosstalk, sub-pixel registration
- **Usability**: Automated processing without manual intervention
- **Reproducibility**: Consistent results across different systems