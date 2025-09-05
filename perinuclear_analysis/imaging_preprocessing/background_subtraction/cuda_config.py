"""
CUDA configuration and GPU detection utilities for background subtraction.

This module provides configuration management for CUDA-accelerated operations,
including GPU detection, memory management, and fallback mechanisms.
"""

import logging
import warnings
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import os

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


@dataclass
class CUDABackgroundConfig:
    """Configuration for CUDA-accelerated background subtraction."""
    
    # GPU usage settings
    use_cuda: bool = True
    auto_detect_gpu: bool = True
    force_cpu: bool = False
    
    # Memory management
    max_gpu_memory_fraction: float = 0.8  # Use 80% of available GPU memory
    min_gpu_memory_gb: float = 2.0  # Minimum GPU memory required
    enable_memory_pool: bool = True
    
    # Performance settings
    enable_async_transfer: bool = True
    chunk_processing_threshold_gb: float = 4.0  # Use chunking for images larger than this
    
    # Fallback settings
    auto_fallback_to_cpu: bool = True
    log_fallback_reasons: bool = True
    
    # Method-specific CUDA optimizations
    rolling_ball_parallel_slices: bool = True
    gaussian_3d_optimization: bool = True
    morphological_batch_processing: bool = True
    
    # Debugging and monitoring
    enable_performance_logging: bool = False
    log_memory_usage: bool = False
    benchmark_on_init: bool = False


class GPUDetector:
    """GPU detection and capability assessment."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._gpu_info = None
        self._capabilities = None
    
    def detect_gpu(self) -> Dict[str, Any]:
        """Detect available GPU and its capabilities."""
        if not CUDA_AVAILABLE:
            return {
                'available': False,
                'reason': 'CUDA libraries not installed',
                'cupy_available': False
            }
        
        try:
            # Get GPU count
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count == 0:
                return {
                    'available': False,
                    'reason': 'No CUDA-capable GPUs found',
                    'gpu_count': 0
                }
            
            # Get primary GPU info
            device = cp.cuda.Device(0)
            device.use()
            
            # Get memory info
            mem_info = device.mem_info
            total_mem = mem_info[1] / (1024**3)  # Convert to GB
            free_mem = mem_info[0] / (1024**3)
            
            # Get GPU properties
            props = cp.cuda.runtime.getDeviceProperties(0)
            
            gpu_info = {
                'available': True,
                'gpu_count': gpu_count,
                'primary_gpu': {
                    'name': props['name'].decode(),
                    'compute_capability': f"{props['major']}.{props['minor']}",
                    'total_memory_gb': total_mem,
                    'free_memory_gb': free_mem,
                    'multiprocessor_count': props['multiProcessorCount'],
                    'max_threads_per_block': props['maxThreadsPerBlock'],
                    'max_threads_per_multiprocessor': props['maxThreadsPerMultiprocessor'],
                    'shared_memory_per_block': props['sharedMemPerBlock'],
                    'warp_size': props['warpSize']
                },
                'cupy_version': cp.__version__,
                'cuda_version': cp.cuda.runtime.runtimeGetVersion()
            }
            
            self._gpu_info = gpu_info
            return gpu_info
            
        except Exception as e:
            return {
                'available': False,
                'reason': f'GPU detection failed: {str(e)}',
                'error': str(e)
            }
    
    def assess_capabilities(self, config: CUDABackgroundConfig) -> Dict[str, Any]:
        """Assess GPU capabilities for background subtraction tasks."""
        if not self._gpu_info or not self._gpu_info['available']:
            return {
                'suitable': False,
                'reason': 'No suitable GPU detected'
            }
        
        gpu = self._gpu_info['primary_gpu']
        
        # Check memory requirements
        if gpu['free_memory_gb'] < config.min_gpu_memory_gb:
            return {
                'suitable': False,
                'reason': f'Insufficient GPU memory: {gpu["free_memory_gb"]:.1f}GB < {config.min_gpu_memory_gb}GB required'
            }
        
        # Check compute capability (require 6.0+ for good performance)
        major, minor = map(int, gpu['compute_capability'].split('.'))
        if major < 6:
            return {
                'suitable': False,
                'reason': f'Insufficient compute capability: {gpu["compute_capability"]} < 6.0 required'
            }
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(gpu)
        
        capabilities = {
            'suitable': True,
            'performance_score': performance_score,
            'recommended_max_image_size_gb': gpu['free_memory_gb'] * config.max_gpu_memory_fraction,
            'estimated_speedup': self._estimate_speedup(gpu),
            'memory_bandwidth_gb_s': self._estimate_memory_bandwidth(gpu),
            'recommendations': self._get_recommendations(gpu, config)
        }
        
        self._capabilities = capabilities
        return capabilities
    
    def _calculate_performance_score(self, gpu: Dict[str, Any]) -> float:
        """Calculate a performance score for the GPU (0-100)."""
        score = 0.0
        
        # Memory score (40% weight)
        memory_gb = gpu['total_memory_gb']
        if memory_gb >= 16:
            score += 40
        elif memory_gb >= 8:
            score += 30
        elif memory_gb >= 4:
            score += 20
        else:
            score += 10
        
        # Compute capability score (30% weight)
        major, minor = map(int, gpu['compute_capability'].split('.'))
        if major >= 8:  # Ampere or newer
            score += 30
        elif major >= 7:  # Volta or Turing
            score += 25
        elif major >= 6:  # Pascal
            score += 20
        else:
            score += 10
        
        # Multiprocessor count score (30% weight)
        mp_count = gpu['multiprocessor_count']
        if mp_count >= 80:
            score += 30
        elif mp_count >= 60:
            score += 25
        elif mp_count >= 40:
            score += 20
        elif mp_count >= 20:
            score += 15
        else:
            score += 10
        
        return min(score, 100.0)
    
    def _estimate_speedup(self, gpu: Dict[str, Any]) -> Dict[str, float]:
        """Estimate speedup for different operations."""
        major, minor = map(int, gpu['compute_capability'].split('.'))
        memory_gb = gpu['total_memory_gb']
        
        # Base speedup estimates
        base_speedup = {
            'rolling_ball': 15.0,
            'gaussian_3d': 8.0,
            'morphological': 20.0,
            'statistical_ops': 5.0
        }
        
        # Adjust based on GPU capabilities
        if major >= 8:  # Ampere or newer
            multiplier = 1.2
        elif major >= 7:  # Volta or Turing
            multiplier = 1.0
        else:  # Pascal or older
            multiplier = 0.8
        
        # Adjust for memory size
        if memory_gb >= 16:
            memory_multiplier = 1.1
        elif memory_gb >= 8:
            memory_multiplier = 1.0
        else:
            memory_multiplier = 0.9
        
        final_multiplier = multiplier * memory_multiplier
        
        return {op: speedup * final_multiplier for op, speedup in base_speedup.items()}
    
    def _estimate_memory_bandwidth(self, gpu: Dict[str, Any]) -> float:
        """Estimate memory bandwidth in GB/s."""
        # Rough estimates based on GPU architecture
        major, minor = map(int, gpu['compute_capability'].split('.'))
        
        if major >= 8:  # Ampere
            return 900.0
        elif major >= 7:  # Volta/Turing
            return 700.0
        elif major >= 6:  # Pascal
            return 500.0
        else:
            return 300.0
    
    def _get_recommendations(self, gpu: Dict[str, Any], config: CUDABackgroundConfig) -> List[str]:
        """Get recommendations for optimal GPU usage."""
        recommendations = []
        
        memory_gb = gpu['free_memory_gb']
        
        if memory_gb < 4:
            recommendations.append("Consider using smaller image chunks or CPU processing for large images")
        elif memory_gb >= 8:
            recommendations.append("GPU has sufficient memory for most image processing tasks")
        
        major, minor = map(int, gpu['compute_capability'].split('.'))
        if major >= 7:
            recommendations.append("GPU supports advanced CUDA features - enable all optimizations")
        elif major == 6:
            recommendations.append("GPU supports basic CUDA acceleration - good for most operations")
        
        if config.max_gpu_memory_fraction > 0.9:
            recommendations.append("Consider reducing max_gpu_memory_fraction to leave room for system operations")
        
        return recommendations


class CUDAManager:
    """Manager for CUDA operations and memory."""
    
    def __init__(self, config: CUDABackgroundConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.detector = GPUDetector()
        self.gpu_info = None
        self.capabilities = None
        self.memory_pool = None
        
    def initialize(self) -> bool:
        """Initialize CUDA environment and return success status."""
        if not CUDA_AVAILABLE:
            if self.config.log_fallback_reasons:
                self.logger.warning("CUDA libraries not available")
            return False
        
        if self.config.force_cpu:
            if self.config.log_fallback_reasons:
                self.logger.info("CPU processing forced by configuration")
            return False
        
        # Detect GPU
        self.gpu_info = self.detector.detect_gpu()
        if not self.gpu_info['available']:
            if self.config.log_fallback_reasons:
                self.logger.warning(f"GPU not available: {self.gpu_info.get('reason', 'Unknown')}")
            return False
        
        # Assess capabilities
        self.capabilities = self.detector.assess_capabilities(self.config)
        if not self.capabilities['suitable']:
            if self.config.log_fallback_reasons:
                self.logger.warning(f"GPU not suitable: {self.capabilities.get('reason', 'Unknown')}")
            return False
        
        # Initialize memory management
        if self.config.enable_memory_pool:
            self._setup_memory_pool()
        
        # Log GPU info
        gpu = self.gpu_info['primary_gpu']
        self.logger.info(f"Initialized CUDA on {gpu['name']} "
                        f"({gpu['total_memory_gb']:.1f}GB, "
                        f"Compute {gpu['compute_capability']})")
        
        if self.config.enable_performance_logging:
            self.logger.info(f"Performance score: {self.capabilities['performance_score']:.1f}/100")
            self.logger.info(f"Estimated speedup: {self.capabilities['estimated_speedup']}")
        
        return True
    
    def _setup_memory_pool(self):
        """Setup CuPy memory pool for efficient memory management."""
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Set memory pool limits
            gpu = self.gpu_info['primary_gpu']
            max_memory = int(gpu['free_memory_gb'] * self.config.max_gpu_memory_fraction * 1024**3)
            mempool.set_limit(size=max_memory)
            
            self.memory_pool = mempool
            self.logger.debug(f"Memory pool initialized with {max_memory / 1024**3:.1f}GB limit")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup memory pool: {e}")
            self.memory_pool = None
    
    def cleanup(self):
        """Cleanup CUDA resources."""
        if self.memory_pool:
            try:
                self.memory_pool.free_all_blocks()
                self.logger.debug("CUDA memory pool cleaned up")
            except Exception as e:
                self.logger.warning(f"Error cleaning up memory pool: {e}")
    
    def get_optimal_chunk_size(self, image_shape: Tuple[int, ...], dtype: str = 'float32') -> int:
        """Calculate optimal chunk size for processing large images."""
        if not self.capabilities:
            return 1
        
        # Calculate memory per slice
        bytes_per_element = 4 if dtype == 'float32' else 8  # Assume float32 or float64
        slice_memory = image_shape[1] * image_shape[2] * bytes_per_element
        
        # Calculate how many slices fit in available memory
        available_memory = self.capabilities['recommended_max_image_size_gb'] * 1024**3
        max_slices = int(available_memory / slice_memory)
        
        # Use conservative estimate (50% of max)
        optimal_slices = max(1, max_slices // 2)
        
        return min(optimal_slices, image_shape[0])
    
    def should_use_gpu(self, image_shape: Tuple[int, ...], dtype: str = 'float32') -> bool:
        """Determine if GPU should be used for given image."""
        if not self.capabilities or not self.capabilities['suitable']:
            return False
        
        # Calculate image memory requirement
        bytes_per_element = 4 if dtype == 'float32' else 8
        image_memory_gb = (image_shape[0] * image_shape[1] * image_shape[2] * bytes_per_element) / (1024**3)
        
        # Check if image fits in available memory
        max_memory_gb = self.capabilities['recommended_max_image_size_gb']
        
        if image_memory_gb > max_memory_gb:
            if self.config.log_fallback_reasons:
                self.logger.info(f"Image too large for GPU ({image_memory_gb:.1f}GB > {max_memory_gb:.1f}GB)")
            return False
        
        return True


def create_cuda_config(
    use_cuda: Optional[bool] = None,
    max_memory_fraction: float = 0.8,
    min_memory_gb: float = 2.0,
    **kwargs
) -> CUDABackgroundConfig:
    """Create a CUDA configuration with sensible defaults."""
    
    # Auto-detect CUDA usage if not specified
    if use_cuda is None:
        use_cuda = CUDA_AVAILABLE
    
    config = CUDABackgroundConfig(
        use_cuda=use_cuda,
        max_gpu_memory_fraction=max_memory_fraction,
        min_gpu_memory_gb=min_memory_gb,
        **kwargs
    )
    
    return config


def get_gpu_recommendations() -> Dict[str, Any]:
    """Get GPU recommendations for optimal performance."""
    detector = GPUDetector()
    gpu_info = detector.detect_gpu()
    
    if not gpu_info['available']:
        return {
            'recommendation': 'install_cuda',
            'message': 'Install CUDA libraries and compatible GPU for acceleration',
            'requirements': {
                'cuda_version': '11.0+',
                'gpu_memory': '4GB+',
                'compute_capability': '6.0+'
            }
        }
    
    capabilities = detector.assess_capabilities(CUDABackgroundConfig())
    
    if not capabilities['suitable']:
        return {
            'recommendation': 'upgrade_gpu',
            'message': capabilities.get('reason', 'GPU not suitable'),
            'current_gpu': gpu_info['primary_gpu']['name'],
            'requirements': {
                'min_memory_gb': 4.0,
                'min_compute_capability': '6.0'
            }
        }
    
    return {
        'recommendation': 'optimal',
        'message': 'GPU is suitable for CUDA acceleration',
        'gpu_info': gpu_info,
        'capabilities': capabilities,
        'estimated_speedup': capabilities['estimated_speedup']
    }
