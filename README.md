# NuMI at ICARUS Flux Analysis

This package reads neutrino flux universes produced by Package to Predict the Flux (PPFX), and interprets them into a flux prediction with expected
statistical and systematic uncertainties.

## Installation

```bash
$ git clone https://gitlab.com/apwood-physics/flux.git
$ cd flux


### It is recommended to create a virtual environment

$ python -m venv pyenv 
$ source pyenv/bin/activate

###

$ python -m pip install .
```

## Running

Example `config.yaml`

```yaml
sources: path/to/directory/containing/input/files

results: path/to/directory/where/output/should/go


# Desired output bin structure in units of true neutrino energy (GeV).

bin_edges: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 12.0, 20.0]

```

Execute the analysis

```bash
$ python -m flux_uncertainties -c path/to/config.yaml
```
