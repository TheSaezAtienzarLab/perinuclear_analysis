"""
Phase 2: Basic visualization functions for image display and analysis.
Provides tools for displaying MIPs, channel information, and quality assessment.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec

from ..core.config import Phase2Config


logger = logging.getLogger(__name__)


class Visualizer:
    """Basic visualization tools for microscopy images."""
    
    def __init__(self, config: Optional[Phase2Config] = None):
        """Initialize the visualizer.
        
        Args:
            config: Phase 2 configuration. If None, uses defaults.
        """
        self.config = config or Phase2Config()
        plt.rcParams['figure.dpi'] = self.config.figure_dpi
    
    def plot_image(self,
                   image: np.ndarray,
                   title: Optional[str] = None,
                   colormap: Optional[str] = None,
                   figsize: Tuple[int, int] = (8, 8),
                   show_colorbar: bool = True,
                   ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
        """Plot a single image.
        
        Args:
            image: Image array (Y, X) or (Y, X, 3/4).
            title: Plot title.
            colormap: Colormap for grayscale images.
            figsize: Figure size.
            show_colorbar: Whether to show colorbar for grayscale.
            ax: Existing axes to plot on.
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes objects.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        colormap = colormap or self.config.default_colormap
        
        if image.ndim == 2:
            # Grayscale image
            im = ax.imshow(image, cmap=colormap)
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif image.ndim == 3 and image.shape[2] in [3, 4]:
            # RGB or RGBA image
            # Ensure values are in [0, 1] range
            if image.max() > 1:
                image = image / image.max()
            ax.imshow(image)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
        
        ax.set_title(title or "Image")
        ax.axis('off')
        
        return fig, ax
    
    def plot_channels(self,
                      image: np.ndarray,
                      channel_names: Optional[List[str]] = None,
                      figsize: Optional[Tuple[int, int]] = None,
                      colormap: str = 'gray') -> Figure:
        """Plot individual channels in a grid.
        
        Args:
            image: Multi-channel image (Y, X, C) or (Z, Y, X, C).
            channel_names: Names for each channel.
            figsize: Figure size. Auto-calculated if None.
            colormap: Colormap for individual channels.
            
        Returns:
            Figure: Matplotlib figure object.
        """
        # Handle different dimensions
        if image.ndim == 4:
            # Take MIP if z-stack
            image = np.max(image, axis=0)
        elif image.ndim != 3:
            raise ValueError(f"Image must be 3D or 4D, got {image.ndim}D")
        
        n_channels = image.shape[2]
        
        # Setup channel names
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(n_channels)]
        
        # Calculate grid layout
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (n_cols * 4, n_rows * 4)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i in range(n_channels):
            ax = axes[i]
            channel_data = image[:, :, i]
            
            im = ax.imshow(channel_data, cmap=colormap)
            ax.set_title(channel_names[i])
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Individual Channels", fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_composite(self,
                       image: np.ndarray,
                       channel_colors: Optional[Dict[int, str]] = None,
                       channel_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (10, 10),
                       show_individual: bool = True) -> Figure:
        """Plot composite image with individual channels.
        
        Args:
            image: Multi-channel image (Y, X, C).
            channel_colors: Colors for each channel.
            channel_names: Names for each channel.
            figsize: Figure size.
            show_individual: Whether to show individual channels.
            
        Returns:
            Figure: Matplotlib figure object.
        """
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D (Y, X, C), got {image.ndim}D")
        
        n_channels = image.shape[2]
        
        # Setup colors
        if channel_colors is None:
            default_colors = {
                0: 'blue',
                1: 'green',
                2: 'red',
                3: 'cyan',
                4: 'magenta',
                5: 'yellow',
            }
            channel_colors = {i: default_colors.get(i, 'gray') 
                            for i in range(n_channels)}
        
        # Setup channel names
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(n_channels)]
        
        # Create composite image
        composite = self._create_composite_image(image, channel_colors)
        
        if show_individual:
            # Create subplot layout
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, n_channels, height_ratios=[2, 1])
            
            # Plot composite
            ax_composite = fig.add_subplot(gs[0, :])
            ax_composite.imshow(composite)
            ax_composite.set_title("Composite Image")
            ax_composite.axis('off')
            
            # Plot individual channels
            for i in range(n_channels):
                ax = fig.add_subplot(gs[1, i])
                
                # Get colormap for channel
                color = channel_colors.get(i, 'gray')
                if color in ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']:
                    cmap = self._get_color_colormap(color)
                else:
                    cmap = 'gray'
                
                ax.imshow(image[:, :, i], cmap=cmap)
                ax.set_title(channel_names[i], fontsize=10)
                ax.axis('off')
            
            plt.suptitle("Multi-channel Visualization", fontsize=14)
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(composite)
            ax.set_title("Composite Image")
            ax.axis('off')
        
        return fig
    
    def _create_composite_image(self,
                               image: np.ndarray,
                               channel_colors: Dict[int, str]) -> np.ndarray:
        """Create a composite RGB image from multi-channel data.
        
        Args:
            image: Multi-channel image (Y, X, C).
            channel_colors: Colors for each channel.
            
        Returns:
            np.ndarray: RGB composite image.
        """
        height, width, n_channels = image.shape
        composite = np.zeros((height, width, 3), dtype=np.float32)
        
        color_map = {
            'red': (1, 0, 0),
            'green': (0, 1, 0),
            'blue': (0, 0, 1),
            'cyan': (0, 1, 1),
            'magenta': (1, 0, 1),
            'yellow': (1, 1, 0),
            'gray': (1, 1, 1),
            'white': (1, 1, 1),
        }
        
        for c in range(n_channels):
            # Get color
            color_name = channel_colors.get(c, 'gray')
            if color_name in color_map:
                color = color_map[color_name]
            else:
                # Try to parse as matplotlib color
                try:
                    import matplotlib.colors as mcolors
                    color = mcolors.to_rgb(color_name)
                except:
                    color = (1, 1, 1)  # Default to white
            
            # Normalize channel
            channel_data = image[:, :, c].astype(np.float32)
            if channel_data.max() > 0:
                channel_data = channel_data / channel_data.max()
            
            # Add to composite
            for i in range(3):
                composite[:, :, i] += channel_data * color[i]
        
        # Normalize composite
        composite = np.clip(composite, 0, 1)
        
        return composite
    
    def _get_color_colormap(self, color: str) -> str:
        """Get a colormap for a specific color.
        
        Args:
            color: Color name.
            
        Returns:
            str: Colormap name.
        """
        colormaps = {
            'red': 'Reds',
            'green': 'Greens',
            'blue': 'Blues',
            'cyan': 'GnBu',
            'magenta': 'RdPu',
            'yellow': 'YlOrBr',
        }
        return colormaps.get(color, 'gray')
    
    def plot_intensity_histogram(self,
                                 image: np.ndarray,
                                 channel_names: Optional[List[str]] = None,
                                 bins: int = 256,
                                 figsize: Tuple[int, int] = (10, 6),
                                 log_scale: bool = True) -> Figure:
        """Plot intensity histogram for each channel.
        
        Args:
            image: Input image (Y, X) or (Y, X, C).
            channel_names: Names for each channel.
            bins: Number of histogram bins.
            figsize: Figure size.
            log_scale: Whether to use log scale for y-axis.
            
        Returns:
            Figure: Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        n_channels = image.shape[2]
        
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(n_channels)]
        
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
        
        for c in range(n_channels):
            channel_data = image[:, :, c].flatten()
            
            # Remove zeros for better visualization
            channel_data = channel_data[channel_data > 0]
            
            if len(channel_data) > 0:
                color = colors[c % len(colors)]
                ax.hist(channel_data, bins=bins, alpha=0.5, 
                       label=channel_names[c], color=color, 
                       density=True, histtype='stepfilled')
        
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency (normalized)")
        ax.set_title("Intensity Distribution")
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_z_profile(self,
                       z_stack: np.ndarray,
                       position: Optional[Tuple[int, int]] = None,
                       channel: int = 0,
                       figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """Plot intensity profile along z-axis.
        
        Args:
            z_stack: Z-stack image (Z, Y, X) or (Z, Y, X, C).
            position: (y, x) position to sample. If None, uses center.
            channel: Channel to plot if multi-channel.
            figsize: Figure size.
            
        Returns:
            Figure: Matplotlib figure object.
        """
        if z_stack.ndim == 3:
            z_stack = z_stack[:, :, :, np.newaxis]
        
        n_slices, height, width, n_channels = z_stack.shape
        
        if position is None:
            position = (height // 2, width // 2)
        
        y, x = position
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot z-profile at position
        z_profile = z_stack[:, y, x, channel]
        ax1.plot(range(n_slices), z_profile, 'b-', linewidth=2)
        ax1.set_xlabel("Z-slice")
        ax1.set_ylabel("Intensity")
        ax1.set_title(f"Z-profile at position ({y}, {x})")
        ax1.grid(True, alpha=0.3)
        
        # Show position on MIP
        mip = np.max(z_stack[:, :, :, channel], axis=0)
        ax2.imshow(mip, cmap='gray')
        ax2.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
        ax2.set_title("Position on MIP")
        ax2.axis('off')
        
        plt.tight_layout()
        
        return fig


def plot_mip(mip: np.ndarray,
             title: str = "Maximum Intensity Projection",
             colormap: str = 'viridis',
             figsize: Tuple[int, int] = (8, 8)) -> Figure:
    """Convenience function to plot a MIP.
    
    Args:
        mip: MIP image array.
        title: Plot title.
        colormap: Colormap name.
        figsize: Figure size.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    visualizer = Visualizer()
    fig, ax = visualizer.plot_image(mip, title=title, colormap=colormap, figsize=figsize)
    return fig


def plot_channels(image: np.ndarray,
                  channel_names: Optional[List[str]] = None) -> Figure:
    """Convenience function to plot channels.
    
    Args:
        image: Multi-channel image.
        channel_names: Channel names.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    visualizer = Visualizer()
    return visualizer.plot_channels(image, channel_names)