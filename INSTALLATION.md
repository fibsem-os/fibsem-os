# Installation Guide

## Dependencies
* Python 3.9+
* FIB/SEM microscope (a commercial product by ThermoFisher FEI or TESACN)
* Autoscript software (a commercial product by ThermoFisher FEI) OR
* tescanautomation software (a commercial product by TESCAN)

### Python
Python 3.9+ is required.
We recommend installing Python via [Miniforge](https://conda-forge.org/download/),
a minimal installer that uses the community-maintained
[conda-forge](https://conda-forge.org/) channel by default. It provides the same
`conda` (and `mamba`) commands as Anaconda/Miniconda while staying lightweight and
avoiding the licensing terms that can apply to Anaconda's default channels for larger
organizations.

## Setting up your python virtual environment
It is also highly recommended to use virtual environments for development,
see [Managing Conda Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for more information.
(Optionally, you could use `virtualenv` if you prefer.)

Create a new virtual environment from your conda terminal (the Miniforge Prompt on
Windows, or any terminal on macOS/Linux):
```
cd fibsem
conda env create -n fibsem python=3.11 pip
conda activate fibsem
python -m pip install -e .
```

### Installation through Python virtualenv

Alternatively to using Conda, you may use the Python virtualenv tool to create a virtual environment for the project.

Firstly, install python 3.9+ on your system.
In a terminal window, move to a directory where you would like to place the virtual environment and then create a virtual environment using the following command
```
python -m venv fibsem
```
Once the environment is created, activate the environment using the following command
```
fibsem\Scripts\activate.bat
```
Once activated, move to the fibsem root directory and install fibsem like so
```
pip install -e .
```

### Installation with uv

[uv](https://docs.astral.sh/uv/) is a fast, drop-in replacement for `pip`.

First, install uv (see the [official install guide](https://docs.astral.sh/uv/getting-started/installation/)
for all options):
```
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Alternatively, install it into an existing environment with `pip install uv`.

uv works inside either environment above (conda or virtualenv) with nothing extra to
set up. Once your environment is created and activated, install fibsem with:
```
uv pip install -e .
```
To include the GUI dependencies, install the `ui` extra:
```
uv pip install -e '.[ui]'
```
This resolves the same dependencies from `pyproject.toml` as `pip`, just faster.
No lockfile or additional configuration is required.

## Installing Microscope Hardware APIs

## Installing Autoscript
Autoscript provides an API (application programming interface) for scripting
control of compatible FEI microscope systems.
This is a commercial product by Thermo Fisher FEI, please visit their website
at https://www.thermofisher.com/au/en/home/electron-microscopy.html for information on pricing and installation.

We use Autoscript version 4.6.+

The version numbers of the python packages Autoscript installs were:
* autoscript-core 5.12.0
* autoscript-sdb-microscope-client 4.6.0
* autoscript-sdb-microscope-client-tests 4.6.0
* autoscript-toolkit 4.6.0
* thermoscientific-logging 5.12.0

#### Add the autoscript python packages to your `site-packages`

To add the AutoScript python packages to your new conda environment, follow these three steps:

1. Find the python environment that was created with your AutoScript installation.
Typically, you can expect the environment is named 'Autoscript', and its installed packages should be found at:
`C:\Program Files\Python35\envs\AutoScript\Lib\site-packages\`

***Troubleshooting:** If you're having trouble finding the location AutoScript chose to install its python packages into,*
*you can open the *default terminal* on your machine (eg: `cmd` for Windows) and type `where python` (Windows) or `which python` (Unix).*
*The result will be something like `C:\Program Files\Python35\envs\AutoScript\python.exe`.*
*Navigate to the environment location (in the example here, that's `C:\Program Files\Python35\envs\AutoScript\` *
*then change directories into `Lib`, and then the `site-packages` directory. This is where the python packages live.*

2. Find the conda environment location you just made called `fibsem`.
`...conda/envs/fibsem/Lib/site-packages/`

*Note: if you used python virtual env to create a virtual environment, the location of the fibsem/Lib/site-packages will be where the virtual environment was created. Where this document mentions the site-packages directory, it is referring to the site-packages directory of the virtual environment.*

***Troubleshooting:** If you're having trouble finding the conda environment location for `fibsem`*
*you can open the *Anaconda terminal* on your machine and type `where python` (Windows) or `which python` (Unix).*
*The result will be something like `C:\Users\yourusername\.conda\envs\fibsem\python.exe`*
*Navigate to the environment location (in the example here, that's `C:\Users\yourusername\.conda\envs\fibsem\` *
*then change directories into `Lib`, and then the `site-packages` directory.*
*This is where you want to add copies of the AutoScript python packages.*

3. Make a copy of the relevant AutoScript python packages into the conda environment.
You will need to copy:

* autoscript_core
* autoscript_core-5.12.0.dist-info
* autoscript_sdb_microscope_client
* autoscript_sdb_microscope_client_tests
* autoscript_sdb_microscope_client_tests-4.6.0.dist-info
* autoscript_sdb_microscope_client-4.6.0.dist-info
* autoscript_toolkit
* autoscript_toolkit-4.6.0.dist-info
* thermoscientific_logging
* thermoscientific_logging-5.12.1.dist-info


#### Having problems?
* Check to see if Autoscript is correctly installed and configured.
* Check to see if your python environment contains all packages listed in
the requirements.txt
* Check that when you call python from the terminal, you get the python
environment containing the dependencies listed above
(i.e. you are not using a different python environment)
* Try cloning the repository and running the unit tests,
you may want to try installing from the source code.

## Installing Tescanautomation

Tescanautomation is a hardware API for controlling TESCAN microscopes. This is a commercially available product from TESCAN. The SDK is available in an .exe file format

### Prerequisites

Before beginning this install, please ensure the following

- FIBSEM conda environment is installed and setup
- tescan-automation-sdk-install exe file is ready to go

### Installing the SDK

Run the installer exe file. When it asks for the python interpreter, select the one that is on the conda environment and proceed with the install.

The package should now be installed successfully

### ***Common Issue with Python Interpreter***

If the conda python interpreter cannot be selected from the drop down options, proceed with the install and take note of the path of installed python interpreter. 

(If no python interpreter can be found in the drop down, install python 3.9+ seperately and run the installation exe again)

Once the installation has been completed, navigate to where python is installed on which the SDK has been installed.

In there, navigate to


`...\python\lib\site-packages`

from this folder, find and copy the following folders:

- All folders beginning with `PySide6`
- All folders beginning with `shiboken`
- All folders beginning with `tescan`

Copy these into the python folder that is set up in the conda environment

`...\Anaconda3\envs\fibsem\lib\site-packages`

The package should now be installed successfully

### Checking Install

To check if the module has been installed properly and can be imported, run the following python code in FIBSEM:

```python
import sys
from tescanautomation import Automation

print("Tescan Imported Successfully") if "tescanautomation" in sys.modules else print("Tescan Import was unsuccessful")

```

If the import or install is unsuccessful, check to see if all the packages have been copied to the right directory. 