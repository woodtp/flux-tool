# Flux-Tool
[![PyPI](https://img.shields.io/pypi/v/flux-tool)](https://pypi.org/project/flux-tool/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flux-tool)](https://www.python.org/)
[![PyPI - License](https://img.shields.io/pypi/l/flux-tool)](https://github.com/apwood-physics/flux-tool/blob/main/LICENSE)

This package reads neutrino flux universes produced by Package to Predict the Flux (PPFX), and extracts a neutrino flux prediction with corresponding uncertainties.
All analysis products are output to a `.root` file specified in a `config.toml`. The package will also produce figures as `pdf`, `png`, and a `.tex`, for the majority of the products stored in the ROOT file.
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
```shell
$ flux_tool -h
usage: flux_uncertainties [-h] [-c CONFIG] [-p PRODUCTS_FILE] [-v] [-z]

This package coerces PPFX output into a neutrino flux prediction with uncertainties, and stores various spectra related to the flux, e.g., fractional uncertainties, covariance matrices, etc.

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        specify the path to a toml configuration file
  -p PRODUCTS_FILE, --plots-only PRODUCTS_FILE
                        Specify path to an existing ROOT file for which to produce plots
  -v, --verbose
  -z, --enable-compression
                        Enable compression of the output plots directory
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

## Contents of the Output ROOT File

## `beam_samples`
If provided to `flux_tool`, copies of the systematically altered neutrino flux samples, including the nominal, are stored here.
## `beam_systematic_shifts`
Fractional shifts from the nominal, calculated for each flux sample in `beam_samples`.
## `covariance_matrices`
Contains all covariance and correlation matrices, organized into two subdirectories: one for `hadron` effects and another for `beam` effects (if applicable). Each covariance matrix is stored in 2 forms:
1. `TH2D` (prefixed `hcov_` or `hcorr_`)
2. `TMatrixD`(prefixed `cov_` or `corr_`)
Covariance matrices with the `_abs` suffix are in absolute units of the flux, whereas those without the suffix are normalized the PPFX universe mean, in the case of hadron systematics, or to the nominal beam run, in the case of the beam line systematics.
Each bin is labeled according to the combination of horn polarity, neutrino flavor, and energy bin number, e.g., `fhc-nue-1`.
## `flux_prediction`
This directory holds a set of `TH1D` for each neutrino mode. The flux value is
extracted as the PPFX mean, while the uncertainties incorporate statistical,
hadron systematic, and beam line systematic (if applicable) uncertainties.
## `fractional_uncertainties`
This directory contains two subdirectories, `beam` and `hadron`, containing the fractional contributions to the flux uncertainty for each effect.
## `pca`/
This directory houses the outputs of the Principal Component Analysis of the hadron covariance matrix.
- `eigenvectors/hevec_*` Unit eigenvectors
- `principal_components/hpc_*` principal components scaled by the square root of the corresponding eigenvalue and transposed into bins of neutrino energy
- `hcov_pca` reconstructed hadron covariance matrix used for validation purpose.
- `heigenvals` Each bin of this histogram (`TH2D`) holds the eigenvalues extracted
from the PCA
- `heigenvals_frac` same as the previous, but each eigenvalue is divided by
the sum of all eigenvalues such that each eigenvalue is represented as its contribution to the total variance.
## `ppfx_corrected_flux`
Directory containing the PPFX-corrected neutrino spectra. These histograms
are produced by calculating the means and sigmas of the flux distributions across
the 100 universes contained in `ppfx_output`.
## `ppfx_flux_weights`
Directory containing `TH1D` for each horn-neutrino flavor combination, the bins of which contain weights that can be used to apply the PPFX  flux correction.
## `ppfx_output`
Contains the original output received from PPFX, organized into two subdirectories corresponding to Forward Horn Current (FHC) and Reverse Horn Current (RHC). Each contains a `nom` subdirectory which holds the nominal (uncorrected) neutrino flux vs. energy spectrum, `hnom_nu*`, in addition to the PPFX central value, `hcv_nu`. Spectra broken down by parent hadron can be found under the `parent` subdirectory. The remaining subdirectories hold the universes for each hadron production
systematic:
## `statistical_uncertainties`
Directory containing statistical uncertainties for every horn-neutrino flavor combination. Histograms with the suffix `_abs` are in absolute units of the flux, and those without the suffix are in the fractional scale. The two matrices, `hstatistical_uncertainty_matrix` and `statistical_uncertainty_matrix`, are diagonal `TH2D` and `TMatrixD`, respectively, organizing the statistical uncertainties into a useful form to be added with covariance matrices.
## `corr_total`
`TMatrixD` correlation matrix incorporating all sources of uncertainty
## `cov_total_abs`
`TMatrixD` covariance matrix in units of the flux, incorporating all sources of uncertainty
## `hcorr_total`
`TH2D` correlation matrix incorporating all sources of uncertainty
## `hcov_total_abs`
`TH2D` covariance matrix in units of the flux, incorporating all sources of uncertainty
## `matrix_axis`
`TAxis` with the binning and labels of all matrix axes
## `xaxis_variable_bins`
`TAxis` containing the binning applied to all spectra w.r.t. $E_\nu$ in GeV.
