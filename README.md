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

VAST

**V**ortex-based **A**erodynamic **S**olver **T**oolkit

*README.md file contains high-level information about your package: it's purpose, high-level instructions for installation and usage.*



| Items         |          | Subtasks                                   | Timeline                     | Notes                                    |
|---------------|----------|--------------------------------------------|------------------------------|------------------------------------------|
| VAST examples | steady   | **ex_1vlm_simulation_rec_wing**            | done                         | Rec wing -> OAS                          |
|               |          | **ex_2vlm_simulation_PAV_wing**            | done                         | PAV wing -> AVL                          |
|               |          | **ex_3vlm_simulation_PAV_wing_tail**       | to be confirmed w/ v&v group | PAV wing tail -> AVL                     |
|               |          | **ex_4vlm_simulation_CRM_wing**            | done                         | CRM PG transform for compressible -> OAS |
|               | unsteady | **ex_5vlm_simulation_sudden_acc**          | done could further cleanup   | Rec differnet AR -> Katz & Plotkin       |
|               |          | **ex_6vlm_simulation_pitching_theodorsen** | done could further cleanup   | Rec inf AR -> Theodorsen                 |
|               |          | **ex_7vlm_simulation_plunging**            | do be done                   | Rec AR=4 -> Katz & Plotkin               |



# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
git clone https://github.com/jiy352/VAST
```
```sh
pip install -e .
```


# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.


[1]: https://github.com/OpenMDAO/OpenMDAO/actions/workflows/openmdao_test_workflow.yml/badge.svg "Github Actions Badge"
[2]: https://github.com/jiy352/VAST/actions "Github Actions"


[1]: https://img.shields.io/github/issues/LSDOlab/lsdo_project_template.svg
[2]: https://github.com/jiy352/VAST/actions "Github Actions"

