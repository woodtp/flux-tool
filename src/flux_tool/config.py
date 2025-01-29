import logging
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Generator, Self

import numpy as np


class AnalysisConfig:
    """
    Configuration class for analysis parameters and paths.

    This class encapsulates the configuration settings and paths required
    for performing analysis. It allows setting up binning, paths for sources,
    results, plots, and handling ignored histogram names. Additionally, it
    provides methods for parsing filenames, iterating through samples, and
    loading configuration from strings or files.

    Parameters:
    project_config (dict): A dictionary containing the project configuration
        settings, including plotting options, binning, paths, and more.

    Attributes:
    bin_edges (dict): A dictionary containing the bin edges for different
        neutrino types.
    neutrinos (list): A list of neutrino types: ["nue", "nuebar", "numu", "numubar"].
    enabled_histogram_names: List of ignored histogram names based on specification under [PPFX] in the config.toml.
    samples (dict): A dictionary containing paths to samples for "fhc" and "rhc" horn operating modes.
    output_file_name (str): The name of the output file.
    plot_opts (dict): Plotting options, such as x-axis limits.
    plots_path (Path): Path to the directory where plots will be saved.
    products_file (str): Path to the products file.
    results_path (Path): Path to the directory where analysis results will be saved.
    inputs_path (Path): Path to the directory containing analysis sources.

    Methods:
    verify_paths(): Verifies the existence of necessary paths and creates them if missing.
    enabled_hist_filter(hist_name: str) -> bool: Checks if a histogram name should be ignored.
    itersamples(): Iterates through source files, yielding file, horn current, and run ID.
    from_str(config_str: str) -> AnalysisConfig: Creates an instance from a configuration string.
    from_file(config_file: str) -> AnalysisConfig: Creates an instance from a configuration file.
    """

    __slots__ = (
        "bin_edges",
        "enabled_histogram_names",
        "neutrinos",
        "output_file_name",
        "plot_opts",
        "plots_path",
        "products_file",
        "results_path",
        "samples",
        "inputs_path",
    )

    def __init__(self, project_config: dict) -> None:
        plotting = project_config["Plotting"]

        self.plot_opts = {
            "draw_label": plotting["draw_label"],
            "experiment": plotting["experiment"],
            "stage": plotting["stage"],
            "xlim": plotting.get("neutrino_energy_range", (0.0, 20.0)),
            "flux_prediction_bullets": plotting["flux_prediction_bullets"],
            "enabled": plotting["enabled"],
        }

        self.neutrinos: list[str] = ["nue", "nuebar", "numu", "numubar"]

        def_binning = {nu: np.linspace(0, 20, num=201) for nu in self.neutrinos}

        binning = project_config.get("Binning", def_binning)

        self.bin_edges = {}

        for nu, bins in binning.items():
            if isinstance(bins, int):
                self.bin_edges[nu] = np.linspace(0, 20, num=bins + 1)
            elif isinstance(bins, list) and isinstance(bins[0], float):
                self.bin_edges[nu] = np.asarray(bins)
            elif (
                isinstance(bins, list)
                and isinstance(bins[0], list)
                and len(bins[0]) == 3
            ):
                tmp = np.array([])
                for start, stop, step in bins:
                    tmp = np.append(tmp, np.arange(start, stop, step))
                self.bin_edges[nu] = tmp
            else:
                logging.error(
                    f"Invalid binning for {nu}. Falling back to default binning."
                )
                self.bin_edges[nu] = def_binning[nu]

        logging.info(f"Using bin edges: {self.bin_edges}")

        self.inputs_path = (
            Path(project_config["Inputs"]["directory"]).expanduser().resolve()
        )

        self.results_path = Path(
            project_config.get(
                "results",
                self.inputs_path.parent,
            )
        )

        self.plots_path = Path(
            project_config.get(
                "plots",
                self.inputs_path.parent / "plots/",
            )
        )

        output_file = project_config["output_file_name"]

        self.products_file = f"{self.results_path}/{date.today()}_{output_file}"

        self.enabled_histogram_names = []
        for k, v in project_config["PPFX"]["enabled"].items():
            if v:
                continue
            if k == "thintarget":
                self.enabled_histogram_names += ["hthin_nue", "hthin_numu"]
                continue
            if k == "mippnumi":
                self.enabled_histogram_names.append("mipp")
                continue
            self.enabled_histogram_names.append(k)

        inputs = {k: v for k, v in project_config["Inputs"].items() if k != "directory"}

        self.samples = {}
        for horn, input in inputs.items():
            self.samples[horn] = {}
            for name, sample in input.items():
                self.samples[horn][name] = self.inputs_path / Path(sample)

    def verify_paths(self) -> None:
        for path in [self.inputs_path, self.results_path, self.plots_path]:
            if not path.exists():
                opt = input(f"{path} does not exist. Create it? (y/n) ").lower()
                if opt == "y":
                    path.mkdir()
                else:
                    print("Directory not created. Exiting...")
                    sys.exit()

        if not any(self.inputs_path.iterdir()):
            msg = (
                f'No files found in input directory: "{self.inputs_path}"'
                "\nExiting..."
            )
            raise FileNotFoundError(msg)

    def enabled_hist_filter(self, hist_name: str) -> bool:
        return not any(
            x.lower() in hist_name.lower() for x in self.enabled_histogram_names
        )

    def itersamples(self) -> Generator[tuple[str, str, int], None, None]:
        for horn, samples in self.samples.items():
            for name, sample in samples.items():
                yield str(sample), horn, int(name)

    @classmethod
    def from_str(cls, config_str: str) -> Self:
        config = tomllib.loads(config_str)
        return cls(config)

    @classmethod
    def from_file(cls, config_file: str) -> Self:
        with open(config_file, "rb") as file:
            config = tomllib.load(file)
        logging.info(f"Read configuration from {config_file}")
        return cls(config)

    @staticmethod
    def dump_default_config() -> str:
        return """\
# flux_tool configuration file

output_file_name = "out.root"

[Inputs]
directory = "/path/to/directory/containing/input/histograms"
fhc.nominal = "input_fhc_nominal.root"
rhc.nominal = "input_rhc_nominal.root"
fhc.horn_current_up = "input_fhc_horn_current_up.root"
fhc.horn_current_down = "input_fhc_horn_current_down.root"

[Binning]
# Histogram bin edges for each neutrino flavor.
# Accepts:
#    1. an integer number of bins (between 0 and 20 GeV)
#    2. An array of bin edges (NOTE: they can be variable bin widths, but must be monotonically increasing)
#    3. An array of arrays where the inner arrays are [start, stop, step] for fixed bin widths.
#    4. If unspecified, then fixed bin widths of 100 MeV is applied along the [0, 20] GeV interval.
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

numu = [[0.0, 6.0, 0.1], [6.0, 20.0, 0.5]]

numubar = [[0.0, 6.0, 0.1], [6.0, 20.0, 0.5]]

  [PPFX]
# enable/disable specific PPFX reweight categories from
# appearing in the fractional uncertainty directory
# true = included, false = excluded
[PPFX.enabled]
total = true
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
draw_label = true                   # whether or not to draw the experiment label, e.g., ICARUS Preliminary
experiment = "ICARUS"
stage = "Preliminary"
neutrino_energy_range = [0.0, 6.0]  # horizontal axis limits in [GeV]
flux_prediction_bullets = false     # whether or not to draw bullets or lines with error band for flux prediction

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
hadron_covariance_matrices = "total"
hadron_correlation_matrices = true
beam_uncertainties = true
beam_covariance_matrices = true
beam_correlation_matrices = true
beam_systematic_shifts = true
"""
