# VAST

<!---
[![Python](https://img.shields.io/pypi/pyversions/lsdo_project_template)](https://img.shields.io/pypi/pyversions/lsdo_project_template)
[![Pypi](https://img.shields.io/pypi/v/lsdo_project_template)](https://pypi.org/project/lsdo_project_template/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

[![GitHub Actions Test Badge](https://github.com/LSDOlab/lsdo_project_template/actions/workflows/actions.yml/badge.svg)](https://github.com/jiy352/VAST/actions)
[![Forks](https://img.shields.io/github/forks/LSDOlab/lsdo_project_template.svg)](https://github.com/jiy352/VAST/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/lsdo_project_template.svg)](https://github.com/jiy352/VAST/issues)

Repository for **V**ortex-based **A**erodynamic **S**olver **T**oolkit **(VAST)**.
We aim to have vortex-lattice method, (lifting line method), panel method 


This repository serves as a template for all LSDOlab projects with regard to documentation, testing and hosting of open-source code.

*README.md file contains high-level information about your package: it's purpose, high-level instructions for installation and usage.*

# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/lsdo_project_template.git
```
If you want users to install a specific branch, run
```sh
pip install git+https://github.com/LSDOlab/lsdo_project_template.git@branch
```

**Enabled by**: `packages=find_packages()` in the `setup.py` file.

## Installation instructions for developers
To install `lsdo_project_template`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/lsdo_project_template.git
pip install -e ./lsdo_project_template
```

# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.


[1]: https://github.com/OpenMDAO/OpenMDAO/actions/workflows/openmdao_test_workflow.yml/badge.svg "Github Actions Badge"
[2]: https://github.com/jiy352/VAST/actions "Github Actions"


[1]: https://img.shields.io/github/issues/LSDOlab/lsdo_project_template.svg
[2]: https://github.com/jiy352/VAST/actions "Github Actions"

