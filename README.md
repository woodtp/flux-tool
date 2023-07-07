# Flux-Tool

This package reads neutrino flux universes produced by Package to Predict the Flux (PPFX), and analyzes them to extract a flux prediction with uncertainties.
All analysis products are output to a `.root` file specified in the `config.toml`

## Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- Python 3.11 or later: Visit the official Python website at https://www.python.org/downloads/ to download and install the latest version of Python.
- ROOT 6.28 or later: **Flux-Tool** requires ROOT/PyROOT version 6.28 or later. You can obtain ROOT from the official ROOT website at https://root.cern/install/.

## Installation

**Flux-Tool** is available for installation from PyPI, the Python Package Index. Follow the steps below to install the project:

1. Open your terminal or command prompt.

2. Create a virtual environment (optional but recommended):
    ```bash
    $ python -m venv myenv
    ```
3. Activate the virtual environment:
    ```bash
    $ source myenv/bin/activate
    ```
4. Install **Flux-Tool** using pip:
    ```bash
    $ pip install flux-tool
    ```
## Usage
To use **Flux-Tool**, you should specify a `config.toml`.
An example can be found below.

Execute:
```bash
$ python -m flux-tool -c <path/to/config.toml>`
```

### Example `config.toml`
```toml
output_file_name = "out.root"

sources = "~/path/to/directory/containing/input/histograms"

# Specify bin edges for the output histograms.
# If no edges are specified, the output histograms will use the same binning
# as the inputs.
bin_edges = [
  0.0,
  0.2,
  0.4,
  0.6,
  0.8,
  1.0,
  1.5,
  2.0,
  2.5,
  3.0,
  3.5,
  4.0,
  6.0,
  8.0,
  12.0,
  20.0
]

[PPFX]
# enable/disable specific PPFX reweight categories from 
# appearing in the fractional uncertainty directory
# true = included, false = excluded
[PPFX.enabled]
attenuation = true
mesinc = true
mesinc_parent_K0 = true
mesinc_parent_Km = true
mesinc_parent_Kp = true
mesinc_parent_pim = true
mesinc_parent_pip = true
mesinc_daughter_K0 = true
mesinc_daughter_Km = true
mesinc_daughter_Kp = true
mesinc_daughter_pim = true
mesinc_daughter_pip = true
mippnumi = false
nua = true
pCfwd = false
pCk = true
pCpi = true
pCnu = true
pCQEL = false
others = true
thintarget = false
