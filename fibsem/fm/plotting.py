import matplotlib.pyplot as plt
from fibsem.fm.structures import FluorescenceImage
from typing import Optional
import numpy as np

def plot_histogram(image: FluorescenceImage, channel: Optional[int] = None, bins: int = 256, 
                  show: bool = True, range_values: Optional[tuple] = None):
    """Plot histograms for specified channel or all channels using plt.hist()
    
    Args:
        image: FluorescenceImage object
        channel: Optional channel index (0-based). If None, plots all channels
        bins: Number of histogram bins (default: 256)
        show: Whether to display the plot (default: True)
        range_values: Optional tuple (min, max) to set histogram range. If None, uses data range.
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """

    # Temporarily reshape data to 4D for consistent processing
    if image.data.ndim == 2:
        data = image.data.reshape(1, 1, *image.data.shape)
    elif image.data.ndim == 3:
        data = image.data.reshape(1, *image.data.shape)
    else:
        data = image.data

    nc = data.shape[0]

    # Validate channel index if specified
    if channel is not None:
        if channel < 0 or channel >= nc:
            raise ValueError(f"Channel index {channel} out of range [0, {nc-1}]")

    fig = plt.figure(figsize=(10, 6))

    if channel is not None:
        # Plot specific channel
        channel_data = data[channel, :, :, :].flatten()
        channel_name = image.metadata.channels[channel].name if len(image.metadata.channels) > channel else f"Channel {channel+1}"
        plt.hist(channel_data, bins=bins, alpha=0.7, label=channel_name, range=range_values)
        plt.title(f'Histogram - {channel_name}')
    else:
        # Plot all channels
        for ch_idx in range(nc):
            channel_data = data[ch_idx, :, :, :].flatten()
            channel_name = image.metadata.channels[ch_idx].name if len(image.metadata.channels) > ch_idx else f"Channel {ch_idx+1}"
            plt.hist(channel_data, bins=bins, alpha=0.5, label=channel_name, range=range_values)
        plt.title('Histogram - All Channels')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig


def plot_histogram_data(image: FluorescenceImage, channel: Optional[int] = None, bins: int = 256, 
                       figsize: tuple = (10, 6), show: bool = True, 
                       density: bool = False, range_values: Optional[tuple] = None):
    """Plot histogram for FluorescenceImage using pre-calculated histogram data
    
    Args:
        image: FluorescenceImage object
        channel: Optional channel index (0-based). If None, plots all channels
        bins: Number of histogram bins (default: 256)
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        show: Whether to display the plot (default: True)
        density: If True, return probability density instead of counts (default: False)
        range_values: Optional tuple (min, max) to set histogram range. If None, uses data range.
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    hist_data = image.calculate_histogram(channel=channel, bins=bins, density=density, range_values=range_values)

    fig = plt.figure(figsize=figsize)

    if channel is not None:
        # Single channel
        plt.bar(hist_data['bin_edges'][:-1], hist_data['counts'],
                width=np.diff(hist_data['bin_edges']), alpha=0.7,
                edgecolor='black', linewidth=0.5)
        plt.title(f'Histogram - {hist_data["channel_name"]}')
    else:
        # Multiple channels
        for channel_key, channel_data in hist_data.items():
            bin_centers = (channel_data['bin_edges'][:-1] + channel_data['bin_edges'][1:]) / 2
            plt.plot(bin_centers, channel_data['counts'],
                    label=channel_data['channel_name'], linewidth=2)
        plt.legend()
        plt.title('Histogram - All Channels')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability Density' if density else 'Count')
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig