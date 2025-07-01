
import argparse
import os

import streamlit as st
from fibsem.milling.patterning.plotting import draw_milling_patterns
from fibsem.structures import FibsemImage

import autolamella.config as cfg
from autolamella.structures import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Streamlit Review App')
    parser.add_argument(
        '--experiment_path',
        type=str,
        default=cfg.LOG_PATH,
        dest='experiment_path',
        help='Path to the experiment folder containing AutoLamella data.'
    )
    
    # Parse only known args to avoid conflicts with streamlit's own args
    args, unknown = parser.parse_known_args()
    return args

# Parse arguments at the top level
args = parse_args()

st.set_page_config(page_title="AutoLamella Review", page_icon=':snowflake:', layout="wide")
page_title = st.empty()

# add filepath selector ui
st.sidebar.title("File Selection")
st.sidebar.write("Select the folder containing the AutoLamella data.")
experiment_path = st.sidebar.text_input("Folder Path", value=args.experiment_path)
exp = Experiment.load(os.path.join(experiment_path, "experiment.yaml"))

page_title.title(f"AutoLamella Review - {exp.name}")

basenames = [
    "ref_SetupLamella_final_high_res_eb.tif",
    "ref_SetupLamella_final_high_res_ib.tif",
    "ref_MillRough_final_high_res_eb.tif",
    "ref_MillRough_final_high_res_ib.tif",
    "ref_MillPolishing_final_high_res_eb.tif",
    "ref_MillPolishing_final_high_res_ib.tif",
]

st.sidebar.write("Select the images to display.")
selected_images = st.sidebar.multiselect(
    "Select images to display",
    basenames,
    default=basenames,
)

# checkbox to display milling patterns
display_milling_patterns = st.sidebar.checkbox(
    "Display milling patterns",
    value=False,
    help="Select this option to display the milling patterns.",
)

# select milling patterns to display
milling_patterns = ["mill_rough", "mill_polishing", "microexpansion", "fiducial"]
selected_patterns = st.sidebar.multiselect(
    "Select milling patterns to display",
    milling_patterns,
    default=milling_patterns,
)

# select milling image
milling_images = [
    "ref_SetupLamella_final_high_res_ib.tif",
    "ref_MillRough_final_high_res_ib.tif",
    "ref_MillPolishing_final_high_res_ib.tif",
]

for pos in exp.positions:

    st.subheader(f"Position: {pos.name}")
    cols = st.columns(len(selected_images))

    for i, basename in enumerate(selected_images):
        image_path = os.path.join(exp.path, pos.name, basename)

        if not os.path.exists(image_path):
            st.warning(f"Image {basename} not found in {pos.name}.")
            continue

        remove_keys = ["_ib.tif", "_eb.tif", "_high_res", "ref_"]
        name = basename
        for key in remove_keys:
            name = name.replace(key, "")
            @st.cache_data
            def cached_load_image(image_path):
                return FibsemImage.load(image_path)
        
        image = cached_load_image(image_path)
        cols[i].image(image.data, clamp=True, caption=name)


    if not display_milling_patterns:
        continue


    milling_stages = []
    for k in selected_patterns:
        if k not in pos.milling_workflows.keys():
            st.warning(f"Pattern {k} not found in {pos.name}.")
            continue
        milling_stages.extend(pos.milling_workflows[k])
        
    # craete a slider to select the image to display
    selected_image = st.slider("Select image to display", 0, len(milling_images)-1, 0, key=f"pos_{pos.name}_image_slider")
    
    image_path = os.path.join(exp.path, pos.name, milling_images[selected_image])

    image = cached_load_image(image_path)
    fig, ax = draw_milling_patterns(image, milling_stages, title=f"{pos.name} - {milling_images[selected_image]}")
    st.pyplot(fig, use_container_width=True)

