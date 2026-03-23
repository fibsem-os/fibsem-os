[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fibsem-os/fibsem-os) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

# fibsemOS

A universal API for FIBSEM Control, Development and Automation

## Overview

fibsemOS is a Python package for controlling and automating FIB/SEM microscopes. It provides a universal API for FIBSEM control, development and automation, abstracting away the details of the microscope hardware to provide a simple, intuitive interface. The package includes reusable modules for common workflows and operations, and is extensible to support new microscopes.

We currently aim to support  [ThermoFisher AutoScript](https://www.thermofisher.com/at/en/home/electron-microscopy/products/software-em-3d-vis/autoscript-4-software.html), [TESCAN Automation SDK](https://tescan.com/product-portfolio/fib-sem), and [Zeiss Python API](https://www.zeiss.com/metrology/en/software/zeiss-inspect/features/python-interface.html). Support for other FIBSEM systems is planned.

For more information see the [website](https://www.fibsemos.org).

## Installation

There are several ways to install fibsemOS depending on your application and needs. Requires Python 3.9+.

#### PyPI (For Users)

```bash
pip install fibsem 
```

#### Github (For Development)

Clone this repository:

```bash
git clone https://github.com/fibsem-os/fibsem-os.git
cd fibsem-os
```

Install dependencies and package:

```bash
conda create -n fibsem python=3.11 pip
conda activate fibsem
pip install -e '.[ui]'
```

To run:
```bash
fibsem-autolamella-ui
```

### Offline Installation
For computers with no internet connection, you can download the dependencies on a separate internet connected computer and transfer it to the Support PC (e.g. via USB).

On internet connected PC (Environment should match python version): 
```bash
mkdir pkg
cd pkg
pip download fibsem[ui]
```
On Support PC:
Transfer the pkg directory to the support pc, and then change to the pkg directory
```bash
cd pkg
pip install --no-index --find-links . fibsem[ui]
```

#### Additional Installation Information

For detailed instructions on installation, and installing the commercial microscope APIs, see [Installation Guide](INSTALLATION.md).

## Getting Started

For a complete walkthrough of the AutoLamella workflow, see the [Getting Started Guide](GETTING_STARTED.md).

### Getting Started with the API

To get started with the fibsemOS API, see the example/example.py:

You can start an offline demo microscope by specifying manufacturer: "Demo" in the configuration yaml file (fibsem/config/microscope-configuration.yaml). This will start a demo microscope that you can use to test the API without connecting to a real microscope. To connect to a real microscope, set the ip_address and manufacturer of your microscope in the configuration file or alternatively, you can pass these arguments to utils.setup_session() directly.

This example shows you how to connect to the microscope, take an image with both beams, and plot the results.

```python
from fibsem import utils, acquire
import matplotlib.pyplot as plt

def main():

    # connect to microscope
    microscope, settings = utils.setup_session(ip_address="localhost", manufacturer="Demo")

    # take image with both beams
    sem_image, fib_image = acquire.take_reference_images(microscope, settings.image)

    # show images
    fig, ax = plt.subplots(1, 2, figsize=(7, 5))
    ax[0].imshow(sem_image.data, cmap="gray")
    ax[1].imshow(fib_image.data, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

```

This example is available as a script in example/example.py.
For more detailed examples, see the Examples section below.

## Examples

### Core Functionality

For examples of core functionality please see:

- example/example_imaging.py: image acquisition
- example/example_movement.py: stage movement
- example/example_milling.py: drawing patterns and beam milling
- example/autolamella.py: recreation of [AutoLamella V1](https://github.com/DeMarcoLab/autolamella) (automated cryo-lamella preparation) in ~150 lines of code

Additional example scripts and notebooks are available.

## Contributing

Contributions are welcome! Please open a pull request or issue.

## Docs

fibsemOS is a large package with many features. For more detailed documentation, please see the [Documentation Website](https://www.fibsemos.org).

## Related Projects and Publications

| Name | Full Title | Date |
|------|------------|------|
| 3DCT | [Site-Specific Cryo-focused Ion Beam Sample Preparation Guided by 3D Correlative Microscopy](https://doi.org/10.1016/j.bpj.2015.10.053) | 2016 |
| AutoLamella | [Automated cryo-lamella preparation for high-throughput in-situ structural biology](https://doi.org/10.1016/j.jsb.2020.107488) | 2020 |
| SerialFIB | [A modular platform for automated cryo-FIB workflows](https://elifesciences.org/articles/70506) | 2021 |
| PFIB-SEM | [Cryo-plasma FIB/SEM volume imaging of biological specimens](https://doi.org/10.7554/eLife.83623) | 2023 |
| OpenFIBSEM | [OpenFIBSEM: A universal API for FIBSEM control](https://doi.org/10.1016/j.jsb.2023.107967) | 2023 |
| SEM Charging | [Reduction of SEM charging artefacts in native cryogenic biological samples](https://www.nature.com/articles/s41467-025-60545-3) | 2025 |
| Fillets | [Mind the corner: Fillets in cryo-FIB lamella preparation to minimise sample loss](https://doi.org/10.1016/j.jsb.2025.108249) | 2025 |
