# Analog I/O Tools for Python

![GitHub License](https://img.shields.io/github/license/francescopapaleo/analog-io)
[![codecov](https://codecov.io/gh/francescopapaleo/analog-io/graph/badge.svg?token=7BT4XETDXS)](https://codecov.io/gh/francescopapaleo/analog-io)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/francescopapaleo/analog-io/main.yaml)


## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)


## Description

Analog I/O Tools for Python is a library that provides tools to playback and record analog signals using hardware.

## Installation

To install the package, run the following command:

```bash
pip install git+https://github.com/francescopapaleo/analog-io.git
```

In case you need an editable installation, clone the repository:

```bash
git clone https://github.com/francescopapaleo/analog-io.git
```
Then, navigate to the root directory of the repository and install the package:

```bash
pip install -e .
```


## Usage

Compute the latency of the hardware:

```python
from analog_io import HardwareLatencyMeasure

tester = HardwareLatencyMeasure()
tester.select_device_and_channels()
latency_time, latency_index = tester.measure_latency()
```

More examples can be found in the [examples](examples) folder.


## Folder Structure

```bash
.
├── LICENSE
├── README.md
├── docs
│   ├── conf.py
│   ├── index.md
│   └── tutorials
│       ├── first-steps.md
│       ├── installation.md
│       └── real-application.md
├── examples
│   └── dataset_pipeline.py
├── pyproject.toml
├── setup.cfg
├── setup.py
├── src
│   └── analog_io
│       ├── __init__.py
│       └── core.py
└── tests
    └── test_core.py
```


## Acknowledgements

[Scientific Python Library Development Guide](https://learn.scientific-python.org/development/)


## License

This package is distributed under the GNU General Public License.
For more information, see the [LICENSE](LICENSE) file.


## Citation

If you plan to use this project in your work please consider citing it:

```bibtex
@misc{papaleo2024analog-io,
  title = {Analog I/O Package},
  author = {Francesco Papaleo},
  year = {2024},
  howpublished = {GitHub},
  url = {https://github.com/francescopapaleo/analog-io}
}
```
