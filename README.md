# Flux-Tool
[![PyPI](https://img.shields.io/pypi/v/flux-tool)](https://pypi.org/project/flux-tool/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flux-tool)](https://www.python.org/)
[![PyPI - License](https://img.shields.io/pypi/l/flux-tool)](https://github.com/apwood-physics/flux-tool/blob/main/LICENSE)

This package reads neutrino flux universes produced by Package to Predict the Flux (PPFX), and extracts a neutrino flux prediction with corresponding uncertainties.
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
$ python -m flux-tool -c <path/to/config.toml>
```

Alternatively, this package can be imported directly into an existing python script:

```python
import flux_tool
```

### Example `config.toml`
```toml
# flux_tool configuration file

output_file_name = "out.root"
sources = "/path/to/directory/containing/input/histograms"

[Binning]
# Histogram bin edges for each neutrino flavor.
# Accepts:
#    1. an integer number of bins (between 0 and 20 GeV)
#    2. An array of bin edges (NOTE: they can be variable bin widths, but must be monotonically increasing)
#    3. If unspecified, then fixed bin widths of 100 MeV is applied along the [0, 20] GeV interval.
nue = 200
nuebar = [
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
]
numu = []
numubar = [
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

[Plotting]
draw_label = true # whether or not to draw the experiment label, e.g., ICARUS Preliminary
experiment = "ICARUS"
stage = "Preliminary"
neutrino_energy_range = [0.0, 6.0] # horizontal axis limits in [GeV]

[Plotting.enabled]
# Enable/disable specific plots from the visualization output
uncorrected_flux = true
flux_prediction = true
flux_prediction_parent_spectra = true
flux_prediction_parent_spectra_stacked = true
ppfx_universes = true
hadron_uncertainties = true
hadron_uncertainties_meson = true
hadron_uncertainties_meson_only = true
pca_scree_plot = true
pca_mesinc_overlay = true
pca_top_components = true
pca_variances = true
pca_components = true
hadron_covariance_matrices = true
hadron_correlation_matrices = true
beam_uncertainties = true
beam_covariance_matrices = true
beam_correlation_matrices = true
beam_systematic_shifts = true
```
