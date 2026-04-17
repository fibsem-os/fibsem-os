import os
import glob
from pathlib import Path

from fibsem import utils
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task


def test_mill_stages_with_acquisitions(tmp_path: Path) -> None:
    """Test milling stages with image acquisitions (alignment, reference images, etc.).
    Smoke test to ensure images are being acquired and saved correctly.
    """
    # setup a microscope session
    microscope, _ = utils.setup_session(manufacturer="Demo")

    milling_stage = FibsemMillingStage(name="test-stage")

    config = FibsemMillingTaskConfig.from_stages(stages=[milling_stage], name="test-task")
    config.acquisition.acquire_sem = True
    config.acquisition.acquire_fib = True
    config.acquisition.imaging.path = tmp_path

    run_milling_task(microscope, config)

    # check for the alignment reference image
    files = glob.glob(os.path.join(tmp_path, "**", "*alignment_reference*.tif"), recursive=True)
    assert len(files) > 0, f"No alignment reference files found in {tmp_path}"

    # check for beam shift alignment files
    files = glob.glob(os.path.join(tmp_path, "**", "*beam_shift_alignment*.tif"), recursive=True)
    assert len(files) > 0, f"No beam shift alignment files found in {tmp_path}"

    # check for post-milling reference images (SEM + FIB = 2 files)
    files = glob.glob(os.path.join(tmp_path, "**", "*_finished_*.tif"), recursive=True)
    assert len(files) == 2, f"Expected 2 post-milling files, found {len(files)} in {tmp_path}"
