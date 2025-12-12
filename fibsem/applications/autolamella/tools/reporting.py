import glob
import io
import logging
import os
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.ndimage import median_filter
from skimage.transform import resize

from fibsem.applications.autolamella.structures import (
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.tools.data import (
    format_pretty_dataframes,
    parse_logfile,
)
from fibsem.imaging.tiled import plot_stage_positions_on_image
from fibsem.milling import plot_milling_patterns
from fibsem.structures import FibsemImage


class PDFReportGenerator:
    def __init__(self, output_filename: str):
        self.output_filename = output_filename
        self.doc = SimpleDocTemplate(
            output_filename,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        self.styles = getSampleStyleSheet()
        self.story = []

        # Create custom styles
        self.styles.add(ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.grey,
            alignment=1,
            spaceAfter=20
        ))

    def add_title(self, title: str, subtitle: Optional[str] = None):
        """Add a title and optional subtitle to the document"""
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['Subtitle']))
        self.story.append(Spacer(1, 20))

    def add_heading(self, text: str, level: int = 2):
        """Add a heading with specified level"""
        style = self.styles[f'Heading{level}']
        self.story.append(Paragraph(text, style))
        self.story.append(Spacer(1, 12))

    def add_paragraph(self, text: str):
        """Add a paragraph of text"""
        self.story.append(Paragraph(text, self.styles['Normal']))
        self.story.append(Spacer(1, 12))

    def add_page_break(self):
        """Add a page break"""
        self.story.append(PageBreak())

    def add_image(self, path: str, width=6*inch, height=4*inch):
        """Add an image to the PDF"""
        img = Image(path, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 20))

    def add_dataframe(self, df: pd.DataFrame, title: Optional[str] = None, includes_totals: bool = False):
        """Add a pandas DataFrame as a table"""
        if title:
            self.add_heading(title, 3)
        
        # Convert DataFrame to list of lists
        data = [df.columns.tolist()] + df.values.tolist()
        
        # Create table style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2F314F")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ])
        
        if includes_totals:
            style.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E8E8E8'))
            style.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
        
        table = Table(data)
        table.setStyle(style)
        self.story.append(table)
        self.story.append(Spacer(1, 20))

    def add_plot(self, plot_function, title=None, *args, **kwargs):
        """Add a matplotlib plot
        plot_function should be a function that creates and returns a matplotlib figure
        """
        if title:
            self.add_heading(title, 3)
        
        # Create plot and save to bytes buffer
        fig = plot_function(*args, **kwargs)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        
        # Add plot to story
        img = Image(img_buffer, width=6*inch, height=4*inch)
        self.story.append(img)
        self.story.append(Spacer(1, 20))
        plt.close(fig)

    def add_mpl_figure(self, fig: Figure, width: float = 6*inch, height: float = 4*inch) -> None:
        """Add a matplotlib figure to the PDF using in-memory buffer

        Args:
            fig: Matplotlib figure to add
            width: Width in inches (default: 6 inches)
            height: Height in inches (default: 4 inches)
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close(fig)
        self.story.append(Image(buf, width=width, height=height))
        self.story.append(Spacer(1, 20))

    def add_plotly_figure(self, fig: go.Figure, title=None, width=6.5*inch, height=4*inch) -> None:
        """Add a Plotly figure to the PDF"""
        if title:
            self.add_heading(title, 3)

        # Convert Plotly figure to static image
        img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)

        # Create BytesIO object
        img_buffer = io.BytesIO(img_bytes)

        # Add image to story
        img = Image(img_buffer, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 20))

    def generate(self):
        """Generate the PDF document"""
        self.doc.build(self.story)




###### NEW REPORTING FOR NEW TASK-WORKFLOW ######
def _add_lamella_section(pdf: PDFReportGenerator,
                         lamella: Lamella,
                         df_task_history: pd.DataFrame,
                         df_milling: pd.DataFrame,
                         include_workflow: bool = True,
                         include_images: bool = True,
                         include_milling: bool = True) -> None:
    """Add a complete lamella section to the report.

    Args:
        pdf: The PDF generator instance
        lamella: The lamella to generate a section for
        df_task_history: Task history dataframe for all lamellae
        df_milling: Milling data dataframe for all lamellae
        include_workflow: Whether to include workflow duration table
        include_images: Whether to include workflow summary images
        include_milling: Whether to include milling patterns and data
    """
    pdf.add_page_break()
    pdf.add_heading(f"Lamella: {lamella.name}")

    # Workflow table
    if include_workflow:
        df = df_task_history[df_task_history["Lamella Name"] == lamella.name]
        if not df.empty:
            pdf.add_dataframe(df, 'Workflow')

    # Workflow images
    if include_images:
        fig = plot_lamella_task_workflow_summary(lamella)
        if fig is not None:
            pdf.add_mpl_figure(fig, width=6*inch, height=4*inch)

    # Milling data and patterns
    if include_milling:
        df = df_milling[df_milling["Lamella Name"] == lamella.name]
        if not df.empty:
            pdf.add_dataframe(df, 'Milling Data')

        figs = plot_task_milling_summary(lamella)
        if figs:
            for fig in figs:
                pdf.add_mpl_figure(fig, width=6*inch, height=4*inch)


def generate_report_data2(experiment: Experiment, encoding: str = "utf-8") -> Dict[str, any]:
    """Generate report data from an experiment by parsing the logfile directly.

    Args:
        experiment: The experiment to extract data from
        encoding: Text encoding for log file parsing (default: "utf-8")

    Returns:
        Dictionary containing parsed report data with keys:
            - "experiment_name": Name of the experiment
            - "experiment_summary_dataframe": Summary statistics for the experiment
            - "workflow_dataframe": Workflow statistics
            - "task_history_dataframe": History of all tasks
            - "milling_dataframe": Milling operation details
            - "detection_dataframe": Detection results (if available)
            - "detection_summary_dataframe": Detection summary statistics (if available)
    """
    REPORT_DATA: Dict[str, Any] = {}

    dfs = parse_logfile(str(experiment.path), encoding=encoding)
    dfs = format_pretty_dataframes(dfs)

    REPORT_DATA["experiment_name"] = experiment.name
    REPORT_DATA["experiment_summary_dataframe"] = dfs["experiment"]
    REPORT_DATA["workflow_dataframe"] = dfs["workflow"]
    REPORT_DATA["task_history_dataframe"] = dfs["task_history"]
    REPORT_DATA["milling_dataframe"] = dfs["milling"]
    REPORT_DATA["detection_dataframe"] = dfs["detection"]
    REPORT_DATA["detection_summary_dataframe"] = dfs["detection_summary"]

    return REPORT_DATA


def generate_report2(experiment: Experiment,
                    output_filename: str = "autolamella.pdf",
                    sections: Optional[Dict[str, bool]] = None,
                    encoding: str = "utf-8") -> None:
    """Generate a comprehensive AutoLamella experiment report.

    This function generates a PDF report from an experiment by parsing the logfile directly.
    Used for the new task-based workflow system.

    Args:
        experiment: The experiment to generate a report for
        output_filename: Path where the PDF report will be saved (default: "autolamella.pdf")
        sections: Optional dict to control which sections to include. Available keys:
            - "overview": Overview image with positions (default: True)
            - "task_history": Task history table (default: True)
            - "detection": Detection data tables (default: True)
            - "lamella_workflow": Per-lamella workflow tables (default: True)
            - "lamella_workflow_images": Per-lamella workflow images (default: True)
            - "lamella_milling": Per-lamella milling data and patterns (default: True)
        encoding: Text encoding for log file parsing (default: "utf-8")

    Returns:
        None. The PDF report is saved to output_filename.

    Raises:
        FileNotFoundError: If the experiment path doesn't exist
        ValueError: If the experiment has no positions

    Example:
        >>> exp = Experiment.load("path/to/experiment.yaml")
        >>> generate_report2(exp, "my_report.pdf", sections={"overview": True, "detection": False})
    """
    # Set up default sections
    default_sections = {
        "overview": True,
        "task_history": True,
        "detection": True,
        "lamella_workflow": True,
        "lamella_workflow_images": True,
        "lamella_milling": True
    }
    sections = {**default_sections, **(sections or {})}

    report_data = generate_report_data2(experiment, encoding=encoding)

    # Create PDF generator
    pdf = PDFReportGenerator(output_filename=output_filename)

    # Add content - Header and summary tables
    pdf.add_title(f"AutoLamella Report: {report_data['experiment_name']}",
                  f'Generated on {datetime.now().strftime("%B %d, %Y")}')
    pdf.add_paragraph('This report summarises the results of the AutoLamella experiment.')
    pdf.add_dataframe(report_data["workflow_dataframe"], 'Workflow Summary')
    pdf.add_dataframe(report_data["experiment_summary_dataframe"], 'Experiment Summary')

    # Overview image with positions
    if sections["overview"]:
        try:
            filenames = glob.glob(os.path.join(experiment.path, "*overview*.tif"))
            filenames = [f for f in filenames if "autogamma" not in f]
            if len(filenames) > 0:
                pdf.add_page_break()
                pdf.add_heading("Overview (Positions)")
                for filename in filenames:
                    image = FibsemImage.load(filename)
                    fig = generate_final_overview_image(exp=experiment, image=image)
                    pdf.add_mpl_figure(fig, width=4.5*inch, height=4.5*inch)
        except Exception as e:
            logging.warning(f"Error generating overview image: {e}")

    # Task history
    if sections["task_history"]:
        pdf.add_page_break()
        pdf.add_heading("Task History")
        pdf.add_dataframe(report_data["task_history_dataframe"])

    # Detection data
    if sections["detection"]:
        if report_data["detection_dataframe"] is not None and not report_data["detection_dataframe"].empty:
            pdf.add_page_break()
            pdf.add_heading("Detection Data")
            pdf.add_dataframe(report_data["detection_dataframe"])
            pdf.add_dataframe(report_data["detection_summary_dataframe"], "Detection Summary")

    # Per-lamella sections
    df_task_history = report_data["task_history_dataframe"]
    df_milling = report_data["milling_dataframe"]

    for p in experiment.positions:
        _add_lamella_section(
            pdf=pdf,
            lamella=p,
            df_task_history=df_task_history,
            df_milling=df_milling,
            include_workflow=sections["lamella_workflow"],
            include_images=sections["lamella_workflow_images"],
            include_milling=sections["lamella_milling"]
        )

    # Generate PDF
    pdf.generate()

def plot_lamella_task_workflow_summary(p: Lamella,
                         show_title: bool = False,
                         show_scalebar: bool = True,
                         figsize: Tuple[int, int] = (30, 5),
                         target_size: int = 256,
                         fontsize: int = 12,
                         mode: str = "light",
                         show: bool = False) -> Optional[Figure]:
    """Plot the final images for each task of the lamella workflow.

    Creates a grid of images showing SEM and FIB views at different resolutions
    for each completed task in the lamella workflow.

    Args:
        p: The lamella to plot workflow summary for
        show_title: Whether to add a title to the figure (default: False)
        figsize: Base size for the figure as (width, height) tuple (default: (30, 5))
        target_size: Target size for image resize (default: 256)
        fontsize: Font size for labels and title (default: 12)
        mode: Display mode - "light" (black text) or "dark" (white text) (default: "light")
        show: Whether to display the figure immediately (default: False)

    Returns:
        Matplotlib Figure object if images are found, None otherwise
    """

    # Determine text color based on mode
    text_color = "white" if mode == "dark" else "black"

    # get completed tasks
    completed_tasks = [t.name for t in p.task_history]
    if not completed_tasks:
        logging.info(f"No completed tasks found for {p.name}")
        return None

    task_filenames = {}
    for task_name in completed_tasks:
        filenames = sorted(glob.glob(os.path.join(p.path, f"ref_{task_name}*_final_*res*.tif*")))
        if len(filenames) == 0:
            continue
        task_filenames[task_name] = filenames

    if not task_filenames:
        logging.info(f"No valid images found for {p.name}")
        return None

    # only keep tasks with valid images
    completed_tasks = list(task_filenames.keys())
    nrows = len(completed_tasks)
    ncols = max(len(task_filenames[task]) for task in completed_tasks)

    # Load first image to determine aspect ratio for proper figure sizing
    first_filename = list(task_filenames.values())[0][0]
    first_img = FibsemImage.load(first_filename)
    img_shape = first_img.data.shape
    resize_shape = (int(img_shape[0] * (target_size / img_shape[1])), target_size)
    aspect_ratio = resize_shape[0] / resize_shape[1]

    # Calculate figure size based on target image size and number of rows
    # 4 columns of images, each target_size wide
    fig_width = figsize[0]
    # Height should accommodate nrows of images with aspect_ratio, minimal spacing
    fig_height = fig_width * aspect_ratio * nrows / ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    for i, task_name in enumerate(completed_tasks):

        if nrows == 1:
            ax = axes
        else:
            ax = axes[i]

        if ncols == 1:
            ax = np.expand_dims(ax, axis=0)

        # Set y-axis label for this row (only on the leftmost subplot)
        ax[0].set_ylabel(task_name, fontsize=fontsize, rotation=90, ha='center', va='center', color=text_color)

        filenames = task_filenames[task_name]
        try:
            for j, fname in enumerate(filenames):
                img = FibsemImage.load(fname)

                # resize image, maintain aspect ratio
                shape = img.data.shape
                resize_shape = (int(shape[0] * (target_size / shape[1])), target_size)
                arr = resize(img.data, resize_shape, preserve_range=True).astype(img.data.dtype)
                arr = median_filter(arr, size=3)

                ax[j].imshow(arr, cmap="gray")
                ax[j].set_xticks([])
                ax[j].set_yticks([])
                for spine in ax[j].spines.values():
                    spine.set_visible(False)

                # add scalebar
                if show_scalebar:
                    ax[j].add_artist(ScaleBar(
                        dx=img.metadata.pixel_size.x * (shape[1] / target_size),
                        color="black",
                        box_color="white",
                        box_alpha=0.5,
                    location="lower right",
                ))

            if j < ncols - 1:
                # fill remaining subplots with blank images
                for k in range(j + 1, ncols):
                    ax[k].imshow(np.zeros(resize_shape, dtype=np.uint8), cmap="gray")
                    ax[k].set_xticks([])
                    ax[k].set_yticks([])
                    for spine in ax[k].spines.values():
                        spine.set_visible(False)
                    ax[k].text(0.5, 0.5, "No Data", color="white", fontsize=fontsize,
                               ha='center', va='center', transform=ax[k].transAxes)

        except Exception as e:
            logging.error(f"Error plotting {p.name} - {task_name}: {e}")
            continue

    if show_title:
        fig.suptitle(f"Lamella {p.name}", fontsize=int(fontsize * 1.5), color=text_color)

    # Minimize spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.02, hspace=0.02)

    if show:
        plt.show()

    return fig


def plot_experiment_task_summary(exp: Experiment,
                                  task_name: str,
                                  show_title: bool = False,
                                  show_scalebar: bool = True,
                                  figsize: Tuple[int, int] = (30, 5),
                                  target_size: int = 256,
                                  fontsize: int = 12,
                                  mode: str = "light",
                                  show: bool = False) -> Optional[Figure]:
    """Plot the final images for a specific task across all lamellae in an experiment.

    Creates a grid of images showing SEM and FIB views at different resolutions
    for a specific task across all lamellae that have completed that task.

    Args:
        exp: The experiment containing lamellae to plot
        task_name: Name of the task to plot across all lamellae
        show_title: Whether to add a title to the figure (default: False)
        figsize: Base size for the figure as (width, height) tuple (default: (30, 5))
        target_size: Target size for image resize (default: 256)
        fontsize: Font size for labels and title (default: 12)
        mode: Display mode - "light" (black text) or "dark" (white text) (default: "light")
        show: Whether to display the figure immediately (default: False)

    Returns:
        Matplotlib Figure object if images are found, None otherwise
    """

    # Determine text color based on mode
    text_color = "white" if mode == "dark" else "black"

    # Find lamellae that have completed the specified task
    lamella_filenames = {}
    for lamella in exp.positions:
        # Check if lamella has completed the task
        if not lamella.has_completed_task(task_name):
            continue

        # Look for task images
        filenames = sorted(glob.glob(os.path.join(lamella.path, f"ref_{task_name}*_final_*res*.tif*")))
        if len(filenames) == 0:
            continue
        lamella_filenames[lamella.name] = filenames

    if not lamella_filenames:
        logging.info(f"No valid images found for task '{task_name}' in experiment '{exp.name}'")
        return None

    # only keep lamellae with valid images
    lamella_names = list(lamella_filenames.keys())
    nrows = len(lamella_names)
    ncols = max(len(lamella_filenames[lamella]) for lamella in lamella_names)

    # Load first image to determine aspect ratio for proper figure sizing
    first_filename = list(lamella_filenames.values())[0][0]
    first_img = FibsemImage.load(first_filename)
    img_shape = first_img.data.shape
    resize_shape = (int(img_shape[0] * (target_size / img_shape[1])), target_size)
    aspect_ratio = resize_shape[0] / resize_shape[1]

    # Calculate figure size based on target image size and number of rows
    # 4 columns of images, each target_size wide
    fig_width = figsize[0]
    # Height should accommodate nrows of images with aspect_ratio, minimal spacing
    fig_height = fig_width * aspect_ratio * nrows / ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    for i, lamella_name in enumerate(lamella_names):

        if nrows == 1:
            ax = axes
        else:
            ax = axes[i]

        if ncols == 1:
            ax = np.expand_dims(ax, axis=0)

        # Set y-axis label for this row (lamella name)
        ax[0].set_ylabel(lamella_name, fontsize=fontsize, rotation=90, ha='center', va='center', color=text_color)

        filenames = lamella_filenames[lamella_name]
        try:
            for j, fname in enumerate(filenames):
                img = FibsemImage.load(fname)

                # resize image, maintain aspect ratio
                shape = img.data.shape
                resize_shape = (int(shape[0] * (target_size / shape[1])), target_size)
                arr = resize(img.data, resize_shape, preserve_range=True).astype(img.data.dtype)
                arr = median_filter(arr, size=3)

                ax[j].imshow(arr, cmap="gray")
                ax[j].set_xticks([])
                ax[j].set_yticks([])
                for spine in ax[j].spines.values():
                    spine.set_visible(False)

                # add scalebar
                if show_scalebar:
                    ax[j].add_artist(ScaleBar(
                        dx=img.metadata.pixel_size.x * (shape[1] / target_size),
                        color="black",
                        box_color="white",
                        box_alpha=0.5,
                        location="lower right",
                    ))

            if j < ncols - 1:
                # fill remaining subplots with blank images
                for k in range(j + 1, ncols):
                    ax[k].imshow(np.zeros(resize_shape, dtype=np.uint8), cmap="gray")
                    ax[k].set_xticks([])
                    ax[k].set_yticks([])
                    for spine in ax[k].spines.values():
                        spine.set_visible(False)
                    ax[k].text(0.5, 0.5, "No Data", color="white", fontsize=fontsize,
                               ha='center', va='center', transform=ax[k].transAxes)

        except Exception as e:
            logging.error(f"Error plotting {lamella_name} - {task_name}: {e}")
            continue

    if show_title:
        fig.suptitle(f"Task: {task_name}", fontsize=int(fontsize * 1.5), color=text_color)

    # Minimize spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.02, hspace=0.02)

    if show:
        plt.show()

    return fig


def plot_task_milling_summary(p: Lamella, show: bool = False) -> List[Figure]:
    """Plot the milling patterns for each task of the lamella workflow.

    Creates figures showing the milling patterns overlaid on the final high-resolution
    ion beam images for each task that includes milling operations.

    Args:
        p: The lamella to plot milling patterns for
        show: Whether to display the figures immediately (default: False)

    Returns:
        List of matplotlib Figure objects, one per task with milling operations.
        Returns empty list if no milling tasks are found or images are missing.
    """
    figures = []
     
    for k, v in p.task_config.items():

        # check if the task was completed
        if k not in [t.name for t in p.task_history]:
            continue

        # check if the task has milling operations
        if v.milling:

            filenames = sorted(glob.glob(os.path.join(p.path, f"ref_{k}_final_*_ib.tif")))
            if len(filenames) == 0:
                logging.info(f"No final high-res ion beam image found for {p.name} - {k}")
                continue
            image = FibsemImage.load(filenames[0])
            milling_stages = []
            for mtask in v.milling.values():
                milling_stages.extend(mtask.stages)

            fig, ax = plot_milling_patterns(image, milling_stages, title=f"Lamella {p.name} - {k}")
            figures.append(fig)
            if show:
                plt.show()
    return figures

def generate_final_overview_image(exp: Experiment,
                                  image: FibsemImage,
                                  state: str = "MILLING") -> Figure:
    """Generate an overview image with all the final lamellae positions marked.

    Creates a figure showing the overview image with colored markers indicating
    the final positions of all lamellae in the experiment.

    Args:
        exp: The experiment containing lamella positions to plot
        image: The overview image to use as background
        state: The workflow state to retrieve positions from (default: "MILLING")

    Returns:
        Matplotlib Figure object with the overview image and position markers
    """

    sem_positions = []
    for p in exp.positions:
        pstate = p.poses.get(state, p.state.microscope_state)
        if pstate is None or pstate.stage_position is None:
            continue
        pos = pstate.stage_position
        pos.name = p.name
        sem_positions.append(pos)

    fig = plot_stage_positions_on_image(image, sem_positions,
                                        show=False,
                                        color="cyan",
                                        show_scalebar=True,
                                        figsize=None)

    # plot details
    fig.suptitle(f"Experiment: {exp.name}")
    fig.tight_layout()

    return fig

def save_final_overview_image(exp: Experiment, 
                        image: FibsemImage, 
                        output_path: str) -> Figure:
    """Save the final overview image with all the final lamellae positions.
    Args: 
        exp (Experiment): The experiment to plot.
        image (FibsemImage): The overview image.
        output_path (str): The path to save the image to.
    Returns:
        plt.Figure: The figure with the overview image and the positions.
    Note: The figure is saved with a dpi of 300.
    """

    fig = generate_final_overview_image(exp, image)

    # save the figure with dpi=300
    fig.savefig(output_path, dpi=300)

    return fig



######## legacy reporting - for old protocol – to be deleted ######


def plot_multi_gantt(df: pd.DataFrame, color_by='piece_id', barmode='group') -> go.Figure:
    """
    Create a Gantt chart for multiple pieces/processes
    
    Parameters:
    - df: DataFrame with columns [piece_id, step, timestamp, end_time]
    - color_by: Column to use for color coding ('piece_id' or 'step')
    - barmode: 'group' or 'overlay' for how bars should be displayed
    """
    fig = px.timeline(
        df, 
        x_start='start_time',
        x_end='end_time',
        y='step',
        color=color_by,
        # title='Multi-Process Timeline',
        # hover_data=['duration']  # Uncomment to show duration in hover
    )

    # Update layout
    fig.update_layout(
        title_x=0.5,
        xaxis_title='Time',
        yaxis_title='Workflow Step',
        height=400,
        barmode=barmode,  # 'group' or 'overlay'
        yaxis={'categoryorder': 'array', 
               'categoryarray': df['step'].unique()},
        showlegend=True,
        # legend_title_text='Piece ID'
    )

    # Reverse y-axis so first step is at top
    fig.update_yaxes(autorange="reversed")
    
    return fig

def generate_workflow_steps_timeline(df: pd.DataFrame) -> Dict[str, go.Figure]:

    timezone = datetime.now().astimezone().tzinfo

    df["start_time"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert(timezone)
    df['end_time'] = df['start_time'] + pd.to_timedelta(df['duration'], unit='s')

    # # drop step in Created, Finished
    df = df[~df["stage"].isin(["Created", "PreSetupLamella", "SetupLamella", "PositionReady", "Finished"])]
    # drop step in [STARTED, FINISHED, NULL_END]
    df = df[~df["step"].isin(["STARTED", "FINISHED", "NULL_END"])]

    WORKFLOW_STEPS_FIGURES = {}

    for stage_name in df["stage"].unique():
        df1 = df[df["stage"] == stage_name]
        fig = plot_multi_gantt(df1, color_by='step', barmode='overlay')
        
        WORKFLOW_STEPS_FIGURES[stage_name] = fig

    return WORKFLOW_STEPS_FIGURES    


def generate_workflow_timeline(df: pd.DataFrame) -> go.Figure:

    # drop rows with duration over 1 day
    df = df[df["duration"] < 86400]

    timezone = datetime.now().astimezone().tzinfo
    df["start_time"] = pd.to_datetime(df["start"], unit="s").dt.tz_localize("UTC").dt.tz_convert(timezone)
    df["end_time"] = pd.to_datetime(df["end"], unit="s").dt.tz_localize("UTC").dt.tz_convert(timezone)

    df.rename({"stage": "step"}, axis=1, inplace=True)

    # drop step in Created, Finished
    df = df[~df["step"].isin(["Created", "Finished"])]

    fig = plot_multi_gantt(df, color_by='step', barmode='overlay')
    
    return fig

def generate_report_timeline(df: pd.DataFrame):
    # plot time series with x= step_n and y = timestamp with step  as hover text
    df.dropna(inplace=True)
    df.duration = df.duration.astype(int)

    # convert timestamp to datetime, aus timezone 
    df.timestamp = pd.to_datetime(df.timestamp, unit="s")

    # convert timestamp to current timezone
     # get current timezone?
    timezone = datetime.now().astimezone().tzinfo
    df.timestamp = df.timestamp.dt.tz_localize("UTC").dt.tz_convert(timezone)

    df.rename(columns={"stage": "Workflow"}, inplace=True)

    fig_timeline = px.scatter(df, x="step_n", y="timestamp", color="Workflow", symbol="lamella",
        # title="AutoLamella Timeline", 
        hover_name="Workflow", hover_data=df.columns)
        # size = "duration", size_max=20)
    return fig_timeline

def generate_interaction_timeline(df: pd.DataFrame) -> Optional[go.Figure]:

    if len(df) == 0:
        return None
    
    df.dropna(inplace=True)

    # convert timestamp to datetime, aus timezone 
    df.timestamp = pd.to_datetime(df.timestamp, unit="s")

    # convert timestamp to australian timezone
    timezone = datetime.now().astimezone().tzinfo
    df.timestamp = df.timestamp.dt.tz_localize("UTC").dt.tz_convert(timezone)

    df["magnitude"] = np.sqrt(df["dm_x"]**2 + df["dm_y"]**2)

    fig_timeline = px.scatter(df, x="timestamp", y="magnitude", color="stage", symbol="type",
        # title="AutoLamella Interaction Timeline", 
        hover_name="stage", hover_data=df.columns,)
        # size = "duration", size_max=20)

    return fig_timeline

def generate_duration_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    df = df.copy()
    df.rename(columns={"petname": "Name", "stage": "Workflow"}, inplace=True)

    # convert duration to hr;min;sec
    df["duration"] = pd.to_timedelta(df["duration"], unit='s')
    df["Duration"] = df["duration"].apply(lambda x: f"{x.components.hours:02d}:{x.components.minutes:02d}:{x.components.seconds:02d}")

    # drop Workflow in ["Created", "SetupLamella", "Finished"]
    # TODO: better handling of SetupLamella
    columns_to_drop = ["Created", "PositionReady","Finished"]
    # if "ReadyLamella" in df["Workflow"].unique():
        # print("DROPPING OLD STAGES")
        # columns_to_drop = ["PreSetupLamella", "SetupLamella", "ReadyTrench", "Finished"]
    df = df[~df["Workflow"].isin(columns_to_drop)]


    fig_duration = px.bar(df, x="Name", y="duration", 
                        color="Workflow", barmode="group")
    
    return df[["Name", "Workflow", "Duration"]], fig_duration
