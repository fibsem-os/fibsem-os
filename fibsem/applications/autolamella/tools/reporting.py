import glob
import io
import logging
import os
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fibsem.imaging.tiled import plot_stage_positions_on_image
from fibsem.milling import get_milling_stages
from fibsem.milling.patterning.plotting import draw_milling_patterns
from fibsem.structures import FibsemImage
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

from autolamella.protocol.validation import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    convert_old_milling_protocol_to_new_protocol,
)
from autolamella.structures import (
    AutoLamellaMethod,
    AutoLamellaProtocol,
    AutoLamellaStage,
    Experiment,
    Lamella,
    get_completed_stages,
)
from autolamella.tools.data import calculate_statistics_dataframe


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

    def add_title(self, title, subtitle=None):
        """Add a title and optional subtitle to the document"""
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['Subtitle']))
        self.story.append(Spacer(1, 20))

    def add_heading(self, text, level=2):
        """Add a heading with specified level"""
        style = self.styles[f'Heading{level}']
        self.story.append(Paragraph(text, style))
        self.story.append(Spacer(1, 12))

    def add_paragraph(self, text):
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

    def add_dataframe(self, df, title=None, includes_totals=False):
        """Add a pandas DataFrame as a table"""
        if title:
            self.add_heading(title, 3)
        
        # Convert DataFrame to list of lists
        data = [df.columns.tolist()] + df.values.tolist()
        
        # Create table style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2F4F4F')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
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

    def add_mpl_figure(self, fig):
        fig.savefig('temp.png', format='png', bbox_inches='tight', dpi=300)
        self.story.append(Image('temp.png'))

    def add_plotly_figure(self, fig, title=None, width=6.5*inch, height=4*inch):
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

def plot_lamella_milling_workflow(p: Lamella) -> plt.Figure:
    
    # DRAW MILLING PATTERNS
    milling_workflows = [MILL_ROUGH_KEY, MILL_POLISHING_KEY, MICROEXPANSION_KEY, FIDUCIAL_KEY]
    milling_stages = []
    for mw in milling_workflows:
        if mw not in p.protocol.keys():
            continue
        milling_stages.extend(get_milling_stages(key=mw, protocol=p.protocol))

    filenames = sorted(glob.glob(os.path.join(p.path, "ref_MillPolishing*_final_high_res_ib.tif*")))

    if len(filenames) == 0:
        logging.info(f"No images found for {p.name}")
        return None

    # sem_image = FibsemImage.load(filenames[0])
    fib_image = FibsemImage.load(filenames[0])

    fig, ax = draw_milling_patterns(fib_image, milling_stages, title=f"{p.name}")

    return fig

def plot_lamella_summary(p: Lamella, 
                         method: AutoLamellaMethod = AutoLamellaMethod.ON_GRID, 
                         show_title: bool = False, 
                         figsize: Tuple[int, int] = (30, 5), 
                         show: bool = False) -> plt.Figure:
    """Plot the final images for each stage of the lamella workflow."""

    # get completed stages
    completed_stages = get_completed_stages(p, method=method)
    if not completed_stages:
        logging.info(f"No completed stages found for {p.name}")
        return None

    figsize = (figsize[0], figsize[1] * len(completed_stages))

    nrows = len(completed_stages)
    fig, axes = plt.subplots(nrows, 4, figsize=figsize)
    for i, s in enumerate(completed_stages):

        if nrows == 1:
            ax = axes
        else:
            ax = axes[i]

        stage_name = s.name
        filenames = sorted(glob.glob(os.path.join(p.path, f"ref_{stage_name}*_final_*_res*.tif*")))

        # for backwards compatibility
        if stage_name == "SetupLamella":
            tmp_filenames = sorted(glob.glob(os.path.join(p.path, "ref_ReadyLamella*_final_*_res*.tif*")))
            if len(tmp_filenames) > len(filenames):
                filenames = tmp_filenames

        if len(filenames) == 0:
            logging.info(f"No images found for {p.name} - {s.name}")
            continue

        if len(filenames) < 4:
            logging.info(f"Only {len(filenames)} images found for {p.name} - {s.name}, expected 4")
            continue
        try:
            for j, fname in enumerate(filenames):
                img = FibsemImage.load(fname)

                # resize image, maintain aspect ratio
                from PIL import Image as PILImage
                shape = img.data.shape
                target_size = 256
                resize_shape = (int(shape[0] * (target_size / shape[1])), target_size)
                arr = np.asarray(PILImage.fromarray(img.data).resize(resize_shape[::-1]))

                ax[j].imshow(arr, cmap="gray")
                ax[j].axis("off")

                # add scalebar
                from matplotlib_scalebar.scalebar import ScaleBar
                scalebar = ScaleBar(
                    dx=img.metadata.pixel_size.x * (shape[1] / target_size),
                    color="black",
                    box_color="white",
                    box_alpha=0.5,
                    location="lower right",
                )
                ax[j].add_artist(scalebar)

                if j == 0:
                    # add the stage_name to the bottom left corner
                    ax[j].text(0.01, 0.01, stage_name, 
                               fontsize=24, color="lime", 
                               transform=ax[j].transAxes)

                    if i == 0:
                        # add the lamella name to the top left corner
                        ax[j].text(0.01, 0.85, p.name, 
                                   fontsize=36, color="cyan", alpha=0.9, 
                                transform=ax[j].transAxes)
        except Exception as e:
            logging.error(f"Error plotting {p.name} - {s.name}: {e}")
            continue
    if show_title:
        fig.suptitle(f"Lamella {p.name}", fontsize=24)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    if show:
        plt.show()

    return fig

def get_lamella_figures(p: Lamella, exp_path: str) -> dict:

    p.path = os.path.join(exp_path, p.name)

    # get plot of milling patterns on final image
    fig_milling = plot_lamella_milling_workflow(p)

    filenames = sorted(glob.glob(os.path.join(p.path, "ref_MillPolishing*_final_high_res*.tif*")))
    sem_image = FibsemImage.load(filenames[0])
    fib_image = FibsemImage.load(filenames[1])

    fig_images, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sem_image.data, cmap="gray")
    ax[1].imshow(fib_image.data, cmap="gray")
    # plt.show()

    FIGURES = {"milling": deepcopy(fig_milling), "images": deepcopy(fig_images)}
    return FIGURES



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

def generate_interaction_timeline(df: pd.DataFrame) -> go.Figure:

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

    # # display df_experiment dataframe
    # st.subheader("Experiment Data")
    # df_lamella = df_experiment[["petname", "current_stage", "failure", "failure_note", "failure_timestamp"]].copy()
    # # rename petname to lamella
    # df_lamella.rename(columns={"petname": "lamella"}, inplace=True)
    # # convert timestamp to datetime, aus timezone
    # df_lamella.failure_timestamp = pd.to_datetime(df_lamella.failure_timestamp, unit="s")
    # st.dataframe(df_lamella)


def generate_experiment_summary(df: pd.DataFrame) -> go.Figure:
    pass


def generate_report_data(experiment: Experiment, encoding: str = "cp1252") -> dict:

    REPORT_DATA = {}

    # Load experiment data
    dfs = calculate_statistics_dataframe(experiment.path, encoding=encoding)
    df_experiment, df_history, df_beam_shift, df_steps, df_stage, df_det, df_click, df_milling = dfs

    df, fig_duration = generate_duration_data(df_history)

    REPORT_DATA["experiment_name"] = experiment.name
    REPORT_DATA["experiment_summary_dataframe"] = experiment.to_summary_dataframe()

    # timeline
    REPORT_DATA["workflow_timeline_plot"] = generate_workflow_timeline(df_history)
    REPORT_DATA["step_timeline_plots"] = generate_workflow_steps_timeline(df_steps)
    # REPORT_DATA["step_timeline_plot"] = generate_report_timeline(df_steps)
    # REPORT_DATA["interactions_timeline_plot"] = generate_interaction_timeline(df_click)

    # duration
    REPORT_DATA["duration_dataframe"] = df
    REPORT_DATA["duration_plot"] = fig_duration

    # lamella figures
    REPORT_DATA["lamella_data"] = {}
    for p in experiment.positions:
        # figs = get_lamella_figures(p, exp.path)
        # REPORT_DATA["lamella_data"][p.name] = figs
        REPORT_DATA["lamella_data"][p.name] = "TODO"

    return REPORT_DATA

# report generation
def generate_report(experiment: Experiment, 
                    output_filename: str = "autolamella.pdf", 
                    encoding="cp1252"):

    report_data = generate_report_data(experiment, encoding=encoding)

    # Create PDF generator
    pdf = PDFReportGenerator(output_filename=output_filename)
    
    # Add content
    pdf.add_title(f"AutoLamella Report: {report_data['experiment_name']}",
                  f'Generated on {datetime.now().strftime("%B %d, %Y")}')
    pdf.add_paragraph('This report summarises the results of the AutoLamella experiment.')
    pdf.add_dataframe(report_data["experiment_summary_dataframe"], 'Experiment Summary')

    # timeline
    pdf.add_page_break()
    pdf.add_plotly_figure(report_data["workflow_timeline_plot"], "Workflow Timeline")
    for stage_name, fig in report_data["step_timeline_plots"].items():
        pdf.add_plotly_figure(fig, f"{stage_name} Timeline")

    # pdf.add_plotly_figure(report_data["interactions_timeline_plot"], "Interaction Timeline")

    # duration
    # pdf.add_dataframe(report_data["duration_dataframe"], 'Workflow Duration')
    pdf.add_plotly_figure(report_data["duration_plot"], "Workflow Duration by Lamella")

    # TODO: 
    # show overall summary
    # show overview image with positions
    # show individual lamella data
    # show final images for each lamella
    # show milling patterns for each lamella
    # show milling data

    # method = AutoLamellaMethod.ON_GRID
    # if "Waffle" in experiment.name:
        # method = AutoLamellaMethod.WAFFLE

    os.makedirs("tmp", exist_ok=True)
    df_history = experiment.history_dataframe()

    for p in experiment.positions:
        print(f"exporting: {p.name}")
        p.path = os.path.join(experiment.path, p.name)
        pdf.add_page_break()
        pdf.add_heading(f"Lamella: {p.name}")

        df = df_history[df_history["petname"] == p.name]
        df, fig = generate_duration_data(df)

        pdf.add_dataframe(df, 'Workflow Duration')

        # display final images for each workflow stage
        fig = plot_lamella_summary(p, method=experiment.method)
        if fig is None:
            continue
        # save figure to temp file
        fname = f'tmp/tmp-{p.name}.png'
        fig.savefig(fname, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        pdf.story.append(Image(fname, width=6*inch, height=4*inch))

        # add milling patterns
        fig = plot_lamella_milling_workflow(p)
        if fig is None:
            continue
        # save figure to temp file
        fname = f'tmp/tmp-milling-{p.name}.png'
        fig.savefig(fname, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        pdf.story.append(Image(fname, width=6*inch, height=4*inch))


    # Generate PDF
    pdf.generate()

def generate_final_overview_image(exp: Experiment, 
                                  image: FibsemImage, 
                                  state: AutoLamellaStage = AutoLamellaStage.PositionReady) -> plt.Figure:
    """Generate an overview image with all the final lamellae positions.
    Args:
        exp (Experiment): The experiment to plot.
        image (FibsemImage): The overview image.
        state (AutoLamellaStage): The state to plot.
    Returns:
        plt.Figure: The figure with the overview image and the positions."""

    sem_positions = []
    for p in exp.positions:
        pstate = p.states.get(state, None)
        if pstate is None:
            continue
        pos = pstate.microscope_state.stage_position
        pos.name = p.name
        sem_positions.append(pos)

    fig = plot_stage_positions_on_image(image, sem_positions, show=False)

    # plot details
    fig.suptitle(f"Experiment: {exp.name}")
    fig.tight_layout()

    return fig

def save_final_overview_image(exp: Experiment, 
                        image: FibsemImage, 
                        output_path: str) -> None:
    """Save the final overview image with all the final lamellae positions.
    Args: 
        exp (Experiment): The experiment to plot.
        image (FibsemImage): The overview image.
        output_path (str): The path to save the image to.
    Returns:
        None
    """

    fig = generate_final_overview_image(exp, image)

    # save the figure with dpi=300
    fig.savefig(output_path, dpi=300)

    return fig
