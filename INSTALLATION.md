# Installation

By the end of this page fibsemOS is installed in its own Python environment,
the application opens from a desktop shortcut, and you know which manufacturer
software your microscope needs on top.

This file is the copy that ships with the code, for machines without a
browser. The same guide, with screenshots, is at
https://www.fibsemos.org/docs/installation/. Change both when you change one.

## Before you start

- A computer running Windows, Linux or macOS. On a microscope, this is
  usually the support PC on the same network as the instrument, not the
  microscope's own PC.
- Python 3.8 or newer. If you do not have Python, install
  [Miniforge](https://conda-forge.org/download/): it gives you `conda` from
  the community-maintained conda-forge channel with no licensing conditions.
- No microscope is needed. Everything here also runs against the built-in
  simulator, so you can install and try fibsemOS on a laptop first.

## 1. Create an environment

Open a terminal (on Windows, the Miniforge Prompt) and create an environment
for fibsemOS. Keeping it separate from everything else on the computer means
an upgrade of fibsemOS cannot disturb other software, and vice versa.

```bash
conda create -n fibsem python=3.11 pip
conda activate fibsem
```

## 2. Install fibsemOS

Install the package with its user-interface dependencies:

```bash
pip install "fibsem[ui]"
```

To install from a checkout of the source instead, for example to run a
development branch, run the same command from the repository directory with
`-e .` in place of the package name:

```bash
pip install -e ".[ui]"
```

Either way, the `[ui]` extra is what installs the application windows. Leave
it off only for a scripting-only install with no graphical interface. Image
labelling (`fibsem_label`) draws in napari, which is not part of `[ui]`; add
the `[labelling]` extra for it.

### Offline installation

For a computer with no internet connection, download the packages on a
connected computer with the same Python version and carry them over, for
example on a USB stick.

On the connected computer:

```bash
mkdir pkg
cd pkg
pip download "fibsem[ui]"
```

On the offline computer, from the copied `pkg` directory, with the `fibsem`
environment active:

```bash
pip install --no-index --find-links . "fibsem[ui]"
```

## 3. Open the application

With the environment active:

```bash
fibsem-autolamella-ui
```

The window opens on the Microscope tab with nothing connected yet. That is
the whole test: if you see this, the install worked.

## 4. Make a desktop shortcut

You will not want to open a terminal every session. From the menu bar choose
**Tools → Create Desktop Shortcut...**, pick where to put it (the Desktop is
offered), and fibsemOS writes a shortcut that activates the right environment
and starts the application. It makes a `.lnk` on Windows, a `.desktop` file
on Linux and a `.command` file on macOS.

From now on, double-click the shortcut to start fibsemOS.

## 5. Microscope software

fibsemOS talks to a real instrument through the manufacturer's own control
software, which is licensed by them and is not on PyPI. This step only applies
on the computer that will connect to the microscope; skip it on a laptop, and
skip it entirely for the simulator.

### Thermo Fisher (Aquilos, Hydra, Arctis)

fibsemOS controls Thermo Fisher instruments through AutoScript, Thermo
Fisher's scripting interface. It is a licensed product; contact Thermo Fisher
for pricing and installation. fibsemOS works with AutoScript 4.7 and later.

If AutoScript is already installed on this computer, fibsemOS will most likely
find it on its own: the first step of Guided Setup reports whether it was
found. If it reports that AutoScript was not found, its Python packages are in
AutoScript's own environment and need copying into yours.

1. **Find AutoScript's packages.** AutoScript installs into its own Python
   environment, usually named `AutoScript`, with packages at a path like
   `C:\Program Files\Python35\envs\AutoScript\Lib\site-packages\`. If it is
   not there, open a terminal and run `where python`: the result is something
   like `C:\Program Files\Python35\envs\AutoScript\python.exe`, and
   `Lib\site-packages` sits beside that `python.exe`.
2. **Find the fibsem environment's packages.** With the `fibsem` environment
   active, run `where python` (Windows) or `which python`. The result is
   something like `C:\Users\yourname\.conda\envs\fibsem\python.exe`, and
   again `Lib\site-packages` sits beside it. If you made the environment with
   `venv` or `uv`, it is wherever you created that environment.
3. **Copy the packages across.** Copy these folders from AutoScript's
   `site-packages` into the fibsem environment's `site-packages`:

   - `autoscript_core` and `autoscript_core-<version>.dist-info`
   - `autoscript_sdb_microscope_client` and
     `autoscript_sdb_microscope_client-<version>.dist-info`
   - `autoscript_sdb_microscope_client_tests` and
     `autoscript_sdb_microscope_client_tests-<version>.dist-info`
   - `autoscript_toolkit` and `autoscript_toolkit-<version>.dist-info`
   - `thermoscientific_logging` and
     `thermoscientific_logging-<version>.dist-info`

   With AutoScript 4.7 the versions were 4.7.0 for the `autoscript_*`
   packages and 5.12.0 for `autoscript_core` and `thermoscientific_logging`.

Restart fibsemOS afterwards; Guided Setup should now report AutoScript as
installed.

**Having problems?**

- Check that AutoScript itself is installed and configured, and that its own
  environment can `import autoscript_sdb_microscope_client`.
- Check that the terminal's `python` is the `fibsem` environment's, not
  another one: `where python` (Windows) or `which python`.
- Check that the copied folders sit directly inside `site-packages`, not in a
  subfolder.
- Try `pip install -e ".[ui]"` from a clone of the repository and run the
  tests, to separate an environment problem from an AutoScript one.

### Tescan

Run the Tescan Automation SDK installer from Tescan. When it asks which
Python interpreter to install into, choose the one in the `fibsem`
environment (`where python` with the environment active shows its path).

If the `fibsem` interpreter is not offered in the installer's dropdown,
proceed with whichever interpreter it offers and note its path. (If it offers
none, install Python 3.8 or newer separately and run the installer again.)
Then copy every folder beginning with `tescan`, `PySide6` and `shiboken` from
that interpreter's `site-packages` (`...\python\lib\site-packages`) into
the `fibsem` environment's.

To check, with the environment active:

```bash
python -c "from tescanautomation import Automation; print('Tescan SDK found')"
```

If connecting later fails with a message about a missing module, this is the
step to come back to.

## Alternatives

### venv instead of conda

Python's own virtual environments work too. With Python 3.8 or newer
installed, in the directory where the environment should live:

```bash
python -m venv fibsem
```

Activate it, then run the `pip install` from step 2:

```bash
# Windows
fibsem\Scripts\activate
```

```bash
# macOS / Linux
source fibsem/bin/activate
```

### uv instead of pip

[uv](https://docs.astral.sh/uv/) is a fast, drop-in replacement for `pip`.
Install it (see the [uv install guide](https://docs.astral.sh/uv/getting-started/installation/)
for all options):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

or `pip install uv` into an existing environment. uv can create and manage
the environment itself, so conda is not needed:

```bash
# macOS / Linux
uv venv
source .venv/bin/activate
```

```bash
# Windows (PowerShell)
uv venv
.venv\Scripts\activate
```

uv also works inside a conda or venv environment; just activate that
environment instead. Then, with the environment active:

```bash
uv pip install "fibsem[ui]"
```

or `uv pip install -e ".[ui]"` from a checkout. It resolves the same
dependencies from `pyproject.toml` as `pip`, faster; no lockfile or extra
configuration is needed.

### Manual shortcut

If the Tools menu is not an option, make a script that runs the entry point
and put a shortcut to it on the desktop. With the environment active:

```cmd
:: Windows: creates AutoLamella.bat
echo @echo off > AutoLamella.bat & where fibsem-autolamella-ui >> AutoLamella.bat
```

```bash
# Linux / macOS: creates AutoLamella.sh
printf '#!/bin/bash\n%s' $(which fibsem-autolamella-ui) > AutoLamella.sh
chmod +x AutoLamella.sh
```

Then create a shortcut to the script and place it on your desktop.

## Next

Start the application and follow Guided Setup, or read
https://www.fibsemos.org/docs/getting-started/ to tell fibsemOS which
microscope it is talking to and connect.
