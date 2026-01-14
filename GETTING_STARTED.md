# Getting Started

WIP: add images, better descriptions

Walkthrough for on-grid lamella milling

This walkthrough assumes you have installed the application, and configured your microscope. Please see the relevant pages here:

### Initial Setup
### Running the Application

Open Anaconda Prompt:
```bash
conda activate fibsem
fibsem-autolamella-ui
```

If you used a different environment manager (e.g. venv, uv), replace with the relevant instructions for activate the environment.

### Connecting to the Microscope
- To connect to the microscope, first navigate to the 'Connection' tab. 
- Ensure the correct microscope configuration is selected. See microscope configuration page for more information on configurations. We provide configurations for most supported systems.
- Press 'Connect to Microscope'. The button should turn green, and other microscope control tabs (Imaging, Movement, Milling) should appear. 

## Creating an Experiment
- An experiment is the collection of data in a single 'session'. Typically this is the loading of a grid / set of grids.
- To create a new experiment, Select File -> Create Experiment. The create experiment dialog will open. 
- The experiment requires a unique name and directory. You can also add additional optional metadata to the experiment for further analysis (project, institute, user, sample). 
- You also need to select a protocol to associate with the experiment. Protocols define the task configuration and workflow. More details on protocols below. 
- A copy of the protocol file will be saved into the experiment directory, and automatically updated when editing the experiment/protocol in the program. 

### Protocol Files
 - Protocols are stored in yaml files
 - Protocols contain three sections: tasks, workflow, and options
 - Tasks: The configuration of individuail tasks in the workflow, e.g. Rough Milling Task -> contains milling patterns for rough milling, Select Milling Position -> milling angle
 - Workflow: The ordering of tasks in the workflow, and any restrictions on the execution. E.g. requiring a task to run before another
 - Options: Global set of options that apply to the entire experiment
 - For more information on editing the protocol see Protocol Editor section below
 - For more information on Tasks and Workflow see the workflow section below. 


### Loading an Experiment
- You can reload experiments to continue with a previous experiment. All experiment and protocol data is automatically saved so you can reload / restart easily.
- To reload an experiment: File -> Load Experiment
- Reloading an experiment will also reload the related protocol (from the experiment directory).


## User Interface
### Connection Tab
- The Connection tab is used to select the microscope configuration and connect / disconnect from the microscope. 
- Most actions are restricted until you connect to the microscope. 

### Experiment Tab
- 
- Lamella Controls
- Lamella Status
- Workflow Controls:

### Imaging Tab
- The imaging tab is used to acquire images (SEM and FIB) and set beam parameters (e.g. current, voltage, detectors)

### Movement Tab
- The movement tab is used to control the stage movement of the system. 
- You can control each axis individiually, move to specified orientations (SEM, FIB, MILLING) and save known positions.

### Milling Tab
- The milling tab is used to control milling patterns and the milling execution.
- Milling is a complicated topic, please see the dedicated milling page for details.

## Tools
### Minimap
#### Overview Acquisition
- Move to SEM Orientation
- Move to centre of the grid
- Ensure SEM/FIB coincidence
#### Tiled Image Acquisition
- Select imaging parameters, beam type, rows, columns, etc
- Acquire
- Wait for stitch
#### Selecting Lamella Positions
From the minimap, you can add lamella positions by clicking on the image.
- Press Alt + Left Click to add a new position. 
- The position will be added at the orientation the image was acquired at.
- You can only add positions inside the image area.

From the main UI, you can add lamella positions in the Experiment tab.
- Press 'Add Lamella' in the Setup section
- The position will be added at the current stage position.


### Protocol Editor

The protocol defines the configuration for the tasks and workflow. Each lamella receives a copy of the task configuration, that can be edited individually. The Protocol Editor enables editing the task configurations, workflow and each lamella's individual configuration. 

#### Tasks



Defines the complete task-based workflow protocol for an AutoLamella experiment. Contains task configurations (milling patterns, imaging parameters), workflow definition (task order, dependencies, supervision requirements), and workflow options. Can be saved/loaded from YAML files and shared across experiments. Supports conversion from legacy protocols.



#### **Workflow**

Defines the execution order and dependencies between tasks in an AutoLamella workflow. Specifies which tasks are required, which need user supervision, and their prerequisite relationships. Validates task dependencies and tracks completion status for individual lamellas. Provides methods to query remaining tasks and workflow completion status.


**Workflow Tasks**
    - Task Name: The name of the task. Must match the corresponding task in the task-config.
    - Required: Flag whether the task is required for the workflow to be complete. Allows for 'optional' tasks that may only be required for some specific workflows or samples.
    - Supervised: When a task is run in 'supervised' mode the execution will pause at pre-determined points (before milling, before/after stage movements) allowing you to control the microscope to modify. For example, you can adjust the stage position, or re-run milling. 
    - Requirements: Specified which tasks must be run before this task can be run. For example, 'Rough Milling' requires 'Setup Lamella' which means that must be run beforehand. 

You can edit the workflow tasks in workflow tab of the Protocol Editor.

#### **Options**

Contains optional workflow-level settings that apply globally to all tasks. Currently includes options like whether to turn beams off after workflow completion. Extensible for adding future global workflow parameters such as safety settings or default behavior configurations.

You can edit the options in the workflow tab of the Protocol Editor.
---

#### Lamella


## Running the Workflow
The workflow tasks can be run individually or sequentially.

- To run the workflow, go to the Experiment Tab.
- Press 'Run AutoLamella'. The Workflow dialog will open.
- You can select the lamella and task combinations you want to run.

The on-grid lamella milling workflow consists of the following tasks:
Select Milling Position -> Setup Lamella -> Rough Milling -> Polishing

When first starting a new protocol, it is recommended you run each step supervised for a lamella to confirm it works as expected. Once you have adjusted the protocol to your sample, you can run everything except Select Milling Position automated, and adjust pattern positions in the protocol editor.

These are included in the default protocol file (task-protocol.yaml).

### **Select Milling Position**

**Summary:**
Selects and validates the milling position for the lamella by moving to the specified milling angle and allowing user confirmation. This task prepares the lamella for subsequent setup and milling operations by establishing the correct geometric orientation.

**Steps:**
1. Move to the lamella's stage position
2. Check if microscope is at the specified milling angle, optionally auto-align or ask user to tilt
3. Acquire reference image at the milling position
4. Prompt user to fine-tune position in microscope UI (if supervision enabled)
5. Save the milling angle and microscope state.

---

## **Acquire Reference Image**

**Summary:**
Acquires high-resolution reference images at the lamella position for documentation and tracking purposes. 

**Steps:**
1. Move to the lamella's stage position
2. Prompt user to confirm readiness for image acquisition (if supervision enabled)
3. Generate filename incorporating the last completed task name and timestamp
4. Acquire reference images at multiple field-of-views using configured imaging settings

---

### **Setup Lamella**

**Summary:**
Prepares the lamella for milling by optionally milling a fiducial marker, establishing an alignment area, and acquiring a reference image for beam-shift alignment. This task sets up all necessary references and parameters for subsequent automated milling operations.

**Steps:**
1. Move to milling position and optionally align to previous reference image
2. Optionally mill a fiducial marker for alignment purposes
3. Define and update alignment area around the fiducial (or use default area)
4. Acquire reference image within the reduced alignment area for beam-shift alignment
5. Synchronize alignment area to rough and polishing milling tasks

---

### **Rough Milling**

**Summary:**
Performs the initial high-current milling to create the rough lamella structure by removing bulk material. This task aligns to the fiducial marker and executes the rough milling patterns with optional stress relief features. Rough Milling includes what other software refer to as 'Rough Milling', 'Fine Milling', etc as 'Rough Milling 01', 'Rough Milling 02'.

**Steps:**
1. Move to the stored lamella milling pose
2. Align to the reference alignment image using beam-shift
3. Execute rough milling patterns using the configured milling stages
4. Synchronize the milling position to the polishing task configuration
5. Acquire final reference images documenting the milled lamella

---

### **Polishing**

**Summary:**
Performs fine polishing milling to thin the lamella to the final target thickness using low-current milling. This task uses the alignment reference to precisely position polishing patterns relative to the rough-milled lamella.

**Steps:**
1. Move to the stored lamella milling pose
2. Align to the reference alignment image using beam-shift
3. Acquire reference images at the polishing field-of-view
4. Execute polishing milling patterns using configured low-current stages
5. Acquire final reference images documenting the polished lamella




# Cheat Sheet (Controls)


Stage Movement
- Double Click: Move stage to selected position (along sample plane)
- Alt + Double Click: Move stage vertically in chamber (correct coincidence)
- Acquire image after stage movement: Toggle in Movement Tab

Milling Patterns
- Shift + Left Click: Move Selected Pattern
- Ctrl + Shift + Left Click: Move all Patterns (maintains orientation)

Image Acquisition
- F6: Acquire Selected Image
- F7: AutoFocus

Fluorescence Imaging
- Shift + Scroll: Move Objective by Step Size
