# Demos

Small scripts that open one widget, canvas or plot on its own, with mock or
sample data, so it can be looked at and clicked without running the app. They
are developer tools: nothing imports them, pytest does not collect them, and
they are not part of the installed package.

Run one from the repository root, in an environment with fibsem-os installed
(`pip install -e ".[ui]"`):

```bash
python demos/widgets/demo_toast.py
```

A demo that needs data says so at the top of its file; several take a path to
an experiment or a project folder. The milling, image-settings, dual-beam and
overview demos start the simulated Demo microscope.
`imaging/demo_alignment_plot_v2.py` is the exception: it connects to whichever
microscope is configured, so check that before running it on a machine wired to
one.

The `demo_` prefix is deliberate: a script's own folder goes first on the import
path, so a demo named after a package it imports would shadow it.

## autolamella

| Demo | Shows |
|---|---|
| `demo_confirm_run_workflow_dialog.py` | the dialog that confirms a workflow run |
| `demo_experiment_replay.py` | `ExperimentReplayWidget`, stepping through a recorded experiment |
| `demo_lamella_card.py` | `LamellaCardContainer` |
| `demo_lamella_list.py` | `LamellaListWidget` |
| `demo_lamella_name_list_widget.py` | `LamellaNameListWidget`, on real experiment data |
| `demo_lamella_task_image_widget.py` | the History tab (`LamellaTaskImageWidget`) on an experiment |
| `demo_lamella_workflow.py` | `LamellaWorkflowWidget` |
| `demo_selected_lamella.py` | `SelectedLamellaWidget` (objective and pose list) |
| `demo_workflow_config.py` | `WorkflowConfigWidget` |
| `demo_workflow_task_editor_widget.py` | `WorkflowTaskEditorWidget` |
| `demo_workflow_timeline_animation.py` | `WorkflowProgressWidget`, animated through two levels |

## canvas

| Demo | Shows |
|---|---|
| `demo_active_overlay_mode.py` | the active-overlay input model and the toolbar's mode toggle |
| `demo_alignment_overlay.py` | `AlignmentAreaOverlay` |
| `demo_canvas_modifiers.py` | `FibsemImageCanvas`'s modifier-aware mouse signals |
| `demo_detection_overlay.py` | detections: `MaskOverlay` and a `PointOverlay` per feature |
| `demo_quad_view.py` | the quad-view microscope display |
| `demo_real_space_canvas.py` | `FibsemRealSpaceCanvas` |

## correlation

| Demo | Shows |
|---|---|
| `demo_correlation_canvas.py` | `CorrelationCanvasWidget`, the FIB correlation surface |
| `demo_correlation_fm_canvas.py` | `CorrelationFMCanvasWidget`, the FM correlation surface |
| `demo_correlation_setup_section.py` | the correlation window's lamella Setup tab, on mock data |
| `demo_correlation_tab_widget.py` | `CorrelationTabWidget`, loaded from a project folder |
| `demo_correlation_v2.py` | `run_correlation_from_data` on test images, printing the result |

## fm

| Demo | Shows |
|---|---|
| `demo_autofocus_widget.py` | the multi-pass `AutofocusWidget` |
| `demo_channel_list_widget.py` | `FluorescenceMultiChannelWidget` |
| `demo_fm_canvas.py` | the multi-channel FM canvas with per-channel layer controls |
| `demo_fm_overview_widget.py` | `FMOverviewWidget`, on the Demo at the FM |
| `demo_fm_sparse_selection_dialog.py` | `FMSparseSelectionDialog`, on Demo beam overviews |
| `demo_line_plot.py` | `LinePlotWidget` with random data |
| `demo_plot_autofocus.py` | `plot_autofocus_result`'s figure |

## imaging

| Demo | Shows |
|---|---|
| `demo_alignment_comparison.py` | the two cross-correlation shift functions side by side |
| `demo_alignment_plot_v2.py` | `plot_multi_step_alignment`'s figure |
| `demo_drawing_widget.py` | the image drawing overlays, on QLabels |
| `demo_tile_order_strategies.py` | every `TileOrderStrategy` on a 5 × 5 grid |

## measure_tools

Prototype measure tools; the widgets live in `fibsem/ui/widgets/` and are not in
the app yet.

| Demo | Shows |
|---|---|
| `demo_drag_distance.py` | `DragDistanceOverlay` |
| `demo_radial_menu.py` | the radial menu (`QuadMenuOverlay`) |
| `demo_radial_ruler.py` | the radial menu with every measure tool: ruler, rectangle, angle, profile, pins, loupe |

## milling

| Demo | Shows |
|---|---|
| `demo_milling_overlay.py` | `MillingPatternOverlay` |
| `demo_milling_settings_widget.py` | `FibsemMillingSettingsWidget` |
| `demo_milling_stage_list.py` | `MillingStageListWidget` |
| `demo_milling_stages_widget.py` | `FibsemMillingStagesWidget` |
| `demo_milling_task_config_widget2.py` | `MillingTaskConfigWidget2` |
| `demo_milling_task_viewer_widget.py` | `MillingTaskViewerWidget` |
| `demo_pattern_settings_widget.py` | `FibsemPatternSettingsWidget` |

## widgets

| Demo | Shows |
|---|---|
| `demo_countdown_progress.py` | a countdown progress bar |
| `demo_dual_beam_widget.py` | `FibsemDualBeamWidget`, on the Demo |
| `demo_hooks_manual.py` | the hook system, with toasts |
| `demo_icon_tool_button.py` | every way to use `IconToolButton` |
| `demo_image_settings_widget.py` | `FibsemImageSettingsWidget` |
| `demo_overview_widget.py` | `FibsemOverviewWidget`, on the Demo |
| `demo_progress_bar.py` | the indeterminate progress bar style |
| `demo_progress_widget.py` | `FibsemProgressWidget` |
| `demo_reference_image_parameters_widget.py` | `ReferenceImageParametersWidget` |
| `demo_saved_position_widget.py` | `SavedPositionListWidget` |
| `demo_spinner_icon.py` | spinner icons |
| `demo_toast.py` | toast notifications |
| `demo_vertical_tab_widget.py` | the icon-only vertical tab rail |
