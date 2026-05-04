from __future__ import annotations
from datetime import datetime
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar

from fibsem.fm.structures import FluorescenceImage


def plot_histogram(
    image: FluorescenceImage,
    channel: Optional[int] = None,
    bins: int = 256,
    show: bool = True,
    range_values: Optional[tuple] = None,
):
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
            raise ValueError(f"Channel index {channel} out of range [0, {nc - 1}]")

    fig = plt.figure(figsize=(10, 6))

    if channel is not None:
        # Plot specific channel
        channel_data = data[channel, :, :, :].flatten()
        channel_name = (
            image.metadata.channels[channel].name
            if len(image.metadata.channels) > channel
            else f"Channel {channel + 1}"
        )
        plt.hist(
            channel_data, bins=bins, alpha=0.7, label=channel_name, range=range_values
        )
        plt.title(f"Histogram - {channel_name}")
    else:
        # Plot all channels
        for ch_idx in range(nc):
            channel_data = data[ch_idx, :, :, :].flatten()
            channel_name = (
                image.metadata.channels[ch_idx].name
                if len(image.metadata.channels) > ch_idx
                else f"Channel {ch_idx + 1}"
            )
            plt.hist(
                channel_data,
                bins=bins,
                alpha=0.5,
                label=channel_name,
                range=range_values,
            )
        plt.title("Histogram - All Channels")

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig


def plot_histogram_data(
    image: FluorescenceImage,
    channel: Optional[int] = None,
    bins: int = 256,
    figsize: tuple = (10, 6),
    show: bool = True,
    density: bool = False,
    range_values: Optional[tuple] = None,
):
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
    hist_data = image.calculate_histogram(
        channel=channel, bins=bins, density=density, range_values=range_values
    )

    fig = plt.figure(figsize=figsize)

    if channel is not None:
        # Single channel
        plt.bar(
            hist_data["bin_edges"][:-1],
            hist_data["counts"],
            width=np.diff(hist_data["bin_edges"]),
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        plt.title(f"Histogram - {hist_data['channel_name']}")
    else:
        # Multiple channels
        for channel_key, channel_data in hist_data.items():
            bin_centers = (
                channel_data["bin_edges"][:-1] + channel_data["bin_edges"][1:]
            ) / 2
            plt.plot(
                bin_centers,
                channel_data["counts"],
                label=channel_data["channel_name"],
                linewidth=2,
            )
        plt.legend()
        plt.title("Histogram - All Channels")

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Probability Density" if density else "Count")
    plt.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig

def plot_fluorescence_image(
    image: FluorescenceImage,
    filename: Optional[str] = None,
    dpi: int = 300,
    metadata_height: int = 100,
    display_metadata: bool = False,
    selected_channels: list[int] | None = None,
    z_index: int | None = None,
    contrast_limits: list[tuple[float, float]] | None = None,
    alpha: list[float] | float | None = None,
    fontsize: int = 4,
    show: bool = False,
    projection_method: str = "focus_stack_sobel",
) -> tuple[Figure, Axes]:
    """
    Plot a FluorescenceImage using matplotlib with alpha blending and metadata bar.
    Optionally save as PNG.

    Args:
        image: FluorescenceImage to export
        filename: Path to save the PNG file
        dpi: DPI for the output image
        metadata_height: Height in pixels for the metadata bar at bottom
        display_metadata: Whether to display metadata bar
        selected_channels: List of channel indices to display (None for all)
        z_index: Z index to display for z-stacks (None for projection). Overrides projection_method.
        contrast_limits: Per-channel contrast limits as list of (min, max) tuples with values between 0-1.
                        If None, auto-scales each channel. If provided, must have one tuple per displayed channel
                        (length matches selected_channels if specified, or total channels if not).
        alpha: Per-channel opacity values. Can be a single float (applied to all channels) or a list of floats
               (one per displayed channel). Values must be between 0 (transparent) and 1 (opaque). Default is 0.8.
               If list is provided, length must match number of displayed channels.
        fontsize: Font size for metadata text and scalebar (default: 4)
        show: Whether to display the plot (default: False)
        projection_method: Method for z-stack projection (only used if z_index is None). Options:
                          - "max_intensity": Maximum intensity projection
                          - "focus_stack": Focus stacking using default method
                          - "focus_stack_sobel": Focus stacking using Sobel operator (default)
                          - "focus_stack_tenengrad": Focus stacking using Tenengrad method

    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    if image.data is None:
        raise ValueError("Image data is None")

    if image.metadata is None:
        raise ValueError("Image metadata is None")
    if image.metadata.channels is None:
        raise ValueError("Image metadata channels are None")

    # Determine expected number of channels
    if len(image.data.shape) == 4:  # CZYX format
        total_channels = image.data.shape[0]
    elif len(image.data.shape) == 3:  # CYX format
        total_channels = image.data.shape[0]
    else:
        total_channels = 1

    # Determine which channels will be displayed
    if selected_channels is not None:
        display_channels = selected_channels
        num_display_channels = len(selected_channels)
        # Validate selected_channels
        for ch in selected_channels:
            if ch < 0 or ch >= total_channels:
                raise ValueError(
                    f"selected_channels contains invalid index {ch}. "
                    f"Valid range is [0, {total_channels - 1}]"
                )
    else:
        display_channels = list(range(total_channels))
        num_display_channels = total_channels

    # Validate contrast_limits if provided
    # contrast_limits should match the number of channels to display
    if contrast_limits is not None:
        if len(contrast_limits) != num_display_channels:
            raise ValueError(
                f"contrast_limits must have one tuple per displayed channel. "
                f"Expected {num_display_channels}, got {len(contrast_limits)}"
            )

        for i, (vmin, vmax) in enumerate(contrast_limits):
            if not (0 <= vmin <= 1 and 0 <= vmax <= 1):
                raise ValueError(
                    f"contrast_limits values must be between 0 and 1. "
                    f"Channel {i} has limits ({vmin}, {vmax})"
                )
            if vmin >= vmax:
                raise ValueError(
                    f"contrast_limits min must be less than max. "
                    f"Channel {i} has limits ({vmin}, {vmax})"
                )

    # Validate and process alpha parameter
    # alpha should match the number of channels to display
    if alpha is None:
        # Default alpha value for all displayed channels
        alpha_values = [0.8] * num_display_channels
    elif isinstance(alpha, (int, float)):
        # Single alpha value applied to all displayed channels
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1. Got {alpha}")
        alpha_values = [float(alpha)] * num_display_channels
    elif isinstance(alpha, list):
        # Per-channel alpha values
        if len(alpha) != num_display_channels:
            raise ValueError(
                f"alpha list must have one value per displayed channel. "
                f"Expected {num_display_channels}, got {len(alpha)}"
            )
        for i, a in enumerate(alpha):
            if not (0 <= a <= 1):
                raise ValueError(
                    f"alpha values must be between 0 and 1. "
                    f"Channel {i} has alpha={a}"
                )
        alpha_values = alpha
    else:
        raise TypeError(f"alpha must be a float or list of floats. Got {type(alpha)}")

    # Get image dimensions and handle z-stack projection
    if z_index is not None and len(image.data.shape) == 4:  # CZYX format with specific slice
        num_channels, total_z_slices, height, width = image.data.shape
        projected_data = image.data[:, z_index, :, :]  # Select specific Z slice
        projection_type = f"Z-Slice: {z_index}/{total_z_slices}"
    elif len(image.data.shape) == 4:  # CZYX format - apply projection
        num_channels, _, height, width = image.data.shape

        # Validate projection_method
        valid_methods = ["max_intensity", "focus_stack", "focus_stack_sobel", "focus_stack_tenengrad"]
        if projection_method not in valid_methods:
            raise ValueError(
                f"Invalid projection_method: {projection_method}. "
                f"Valid options are: {valid_methods}"
            )

        # Apply the selected projection method
        if projection_method == "max_intensity":
            projected_data = np.max(image.data, axis=1)  # Max intensity projection
            projection_type = "Max Intensity"
        elif projection_method == "focus_stack":
            projected_data = image.focus_stack().data.squeeze()
            projection_type = "Focus Stack"
        elif projection_method == "focus_stack_sobel":
            projected_data = image.focus_stack(method="sobel").data.squeeze()
            projection_type = "Focus Stack (Sobel)"
        elif projection_method == "focus_stack_tenengrad":
            projected_data = image.focus_stack(method="tenengrad").data.squeeze()
            projection_type = "Focus Stack (Tenengrad)"
    elif len(image.data.shape) == 3:  # CYX format - no projection needed
        num_channels, height, width = image.data.shape
        projected_data = image.data
        projection_type = None  # 2D image, no projection
    else:
        raise ValueError(f"Unsupported image shape: {image.data.shape}")

    if not display_metadata:
        metadata_height = 0

    # Create figure with space for metadata bar
    fig_height = height + metadata_height
    fig, ax = plt.subplots(1, 1, figsize=(width/dpi, fig_height/dpi), dpi=dpi)

    # Remove axes and margins
    ax.set_xlim(0, width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    # Display channels using imshow with alpha and collect legend info
    legend_handles = []

    # Iterate through the channels we want to display
    # idx is the index into contrast_limits/alpha_values arrays
    # c is the actual channel number in the image data
    for idx, c in enumerate(display_channels):
        channel_data = projected_data[c]

        # Normalize channel data to 0-1
        if contrast_limits is not None:
            # Apply per-channel contrast limits using mapped index
            vmin, vmax = contrast_limits[idx]
            # Convert relative limits (0-1) to absolute intensity values
            data_min = channel_data.min()
            data_max = channel_data.max()
            data_range = data_max - data_min

            abs_min = data_min + vmin * data_range
            abs_max = data_min + vmax * data_range

            # Clip and normalize to 0-1
            normalized = np.clip(channel_data, abs_min, abs_max)
            if abs_max > abs_min:
                normalized = (normalized - abs_min) / (abs_max - abs_min)
            else:
                normalized = np.zeros_like(normalized)
        else:
            # Auto-scale: normalize to channel max
            if channel_data.max() > 0:
                normalized = channel_data / channel_data.max()
            else:
                normalized = channel_data

        # Get channel color and name from metadata
        if c < len(image.metadata.channels):
            channel_meta = image.metadata.channels[c]
            color_name = channel_meta.color
            channel_name = channel_meta.name
        else:
            color_name = 'gray'
            channel_name = f'Channel-{c+1:02d}'

        # Convert color name to RGB
        try:
            color_rgb = mcolors.to_rgb(color_name)
        except ValueError:
            color_rgb = (0.5, 0.5, 0.5)  # Default to gray

        # Create colored channel using colormap
        colors = ['black', color_name]
        cmap = LinearSegmentedColormap.from_list(f'channel_{c}', colors)

        # Display channel with alpha blending (use per-channel alpha with mapped index)
        ax.imshow(normalized, cmap=cmap, alpha=alpha_values[idx],
                 extent=(0, width, metadata_height, height + metadata_height),
                 )

        # Create legend handle for this channel
        legend_handles.append(mpatches.Patch(color=color_rgb, label=channel_name))
    
    if display_metadata:
        # Add empty black metadata bar at bottom
        metadata_bar = np.zeros((metadata_height, width, 3))
        ax.imshow(metadata_bar, extent=(0, width, 0, metadata_height), origin='lower')

        # Left column: General metadata (filename, acquisition date, pixel size)
        left_column_lines = []

        # add image description if available
        if hasattr(image.metadata, 'description') and image.metadata.description:
            left_column_lines.append(f"Filename: {image.metadata.description}")

        # Add acquisition date
        try:
            acquisition_date = image.metadata.acquisition_date
            dt = datetime.fromisoformat(acquisition_date)
            # use the current time zone
            dt = dt.astimezone()
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        except (ValueError, TypeError, AttributeError):
            date_str = str(acquisition_date)
        left_column_lines.append(f"Acquired: {date_str}")

        # Add pixel size information
        pixel_size_parts = []
        if hasattr(image.metadata, 'pixel_size_x') and image.metadata.pixel_size_x is not None:
            # Convert from meters to nanometers for display
            px_xy_nm = image.metadata.pixel_size_x * 1e9
            pixel_size_parts.append(f"XY:{px_xy_nm:.1f}nm")
        if hasattr(image.metadata, 'pixel_size_z') and image.metadata.pixel_size_z is not None:
            px_z_nm = image.metadata.pixel_size_z * 1e9
            pixel_size_parts.append(f"Z:{px_z_nm:.1f}nm")

        if pixel_size_parts:
            left_column_lines.append(f"Pixel Size: {' '.join(pixel_size_parts)}")

        # Add z-stack/projection information
        if projection_type is not None:
            left_column_lines.append(f"Projection: {projection_type}")

        # Right column: Channel information
        right_column_lines = []
        for c in range(num_channels):
            if c < len(image.metadata.channels):
                channel_meta = image.metadata.channels[c]
                channel_name = channel_meta.name

                # Get channel settings from metadata
                ex_wl = getattr(channel_meta, 'excitation_wavelength', 'N/A')
                em_wl = getattr(channel_meta, 'emission_wavelength', 'N/A')
                power = getattr(channel_meta, 'power', 'N/A')
                exposure = getattr(channel_meta, 'exposure_time', 'N/A')

                # Format the values
                ex_str = f"{ex_wl:.0f}nm" if isinstance(ex_wl, (int, float)) else str(ex_wl)
                em_str = f"{em_wl:.0f}nm" if isinstance(em_wl, (int, float)) else str(em_wl)
                power_str = f"{power*100:.1f}%" if isinstance(power, (int, float)) else str(power)
                exp_str = f"{exposure*1000:.0f}ms" if isinstance(exposure, (int, float)) else str(exposure)
                # Create info string for channel
                info_str = f"{channel_name}: Ex:{ex_str} Em:{em_str} P:{power_str} Exp:{exp_str}"
                right_column_lines.append(info_str)

        # Display left column text
        if left_column_lines:
            left_text = '\n'.join(left_column_lines)
            ax.text(10, metadata_height - 10, left_text,
                    color='white', ha='left', va='top',
                    fontsize=fontsize, linespacing=1.2)

        # Display right column text (starting at midpoint of width)
        if right_column_lines:
            right_text = '\n'.join(right_column_lines)
            ax.text(width / 2 - 120, metadata_height - 10, right_text,
                    color='white', ha='left', va='top',
                    fontsize=fontsize, linespacing=1.2)

    # Add scalebar to bottom right of image (if pixel size is available)
    if hasattr(image.metadata, 'pixel_size_x') and image.metadata.pixel_size_x is not None:
        pixel_size_m = image.metadata.pixel_size_x  # in meters
        scalebar = ScaleBar(pixel_size_m, 'm', location='lower right',
                            box_color='black', box_alpha=0.7,
                            color='white', font_properties={'size': fontsize})
        ax.add_artist(scalebar)

    # Add legend for channels
    if legend_handles:
        ax.legend(handles=legend_handles, loc='best',
                 facecolor='black', edgecolor='white',
                 fontsize=fontsize, labelcolor='white')

    # Save the figure
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0,
                    facecolor='black', edgecolor='none')

    # Show the figure if requested
    if show:
        plt.show()

    return fig, ax
