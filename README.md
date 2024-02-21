# Signal Generator Package

This package is designed to simplify the creation and manipulation of various audio signals for testing, analysis, and synthesis in scientific and engineering applications.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)


## Description

The signal_generator package provides a comprehensive toolkit for generating audio signals including sine waves, square waves, sawtooth waves, chirps, and logarithmic sweeps. It allows for precise control over signal characteristics such as frequency, amplitude, duration, and placement within a duration of silence, facilitating a wide range of applications from acoustics research to audio processing testing.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/francescopapaleo/signal-generator.git
```

Install the package typing:

```bash
pip install -e .
```


## Usage

Here's a quick example to generate a 440 Hz sine wave, place it 1 second into a 5-second duration, and save it as a WAV file:

```python
from signal_generator import SignalGenerator

# Initialize the signal generator
gen = SignalGenerator()

# Generate a 440 Hz sine wave
signal = gen.generate_tone(pitch=440, duration=5, waveform='sin', start_time=1, max_amplitude_dbfs=-3)

# Save the signal to a file
gen.save_signal(signal, './', 'sine_wave_440Hz.wav')
```


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
├── pyproject.toml
├── setup.cfg
├── setup.py
├── src
│   └── signal_generator
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   └── core.cpython-310.pyc
│       └── core.py
└── tests
    └── test_core.py
```


## Acknowledgements

Thanks to the Community behind: [Scientific Python Library Development Guide](https://learn.scientific-python.org/development/)
Thanks to @xaviliz for his [signalGenerator](https://github.com/xaviliz/signalGenerator/tree/master) from which this package is inspired 

## License

This package is distributed under the GNU Affero GPL License. For more information, see the [LICENSE](LICENSE) file.


## Citation

If you plan to use this project in your work please consider citing it:

```bibtex
@misc{signal_generator,
  title = {Signal Generator Package},
  author = {Francesco Papaleo},
  year = {2024},
  howpublished = {GitHub},
  url = {https://github.com/francescopapaleo/signal_generator}
}
```