import logging
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Self

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
    nominal_samples (dict): A dictionary containing paths to nominal samples for
        "fhc" and "rhc" horn operating modes.
    output_file_name (str): The name of the output file.
    plot_opts (dict): Plotting options, such as x-axis limits.
    plots_path (Path): Path to the directory where plots will be saved.
    ppfx (dict): Configuration for the PPFX.
    products_file (str): Path to the products file.
    results_path (Path): Path to the directory where analysis results will be saved.
    sources_path (Path): Path to the directory containing analysis sources.

    Methods:
    verify_paths(): Verifies the existence of necessary paths and creates them if missing.
    ignored_histogram_names: Generator yielding ignored histogram names based on PPFX settings.
    ignored_hist_filter(hist_name: str) -> bool: Checks if a histogram name should be ignored.
    parse_filename(name: str) -> tuple[str, int]: Parses a filename to extract horn current and run ID.
    itersamples(): Iterates through source files, yielding file, horn current, and run ID.
    from_str(config_str: str) -> AnalysisConfig: Creates an instance from a configuration string.
    from_file(config_file: str) -> AnalysisConfig: Creates an instance from a configuration file.
    """

    __slots__ = (
        "bin_edges",
        "neutrinos",
        "nominal_samples",
        "nominal_run_id",
        "output_file_name",
        "plot_opts",
        "plots_path",
        "ppfx",
        "products_file",
        "results_path",
        "sources_path",
    )

    def __init__(self, project_config: dict) -> None:
        self.neutrinos: list[str] = ["nue", "nuebar", "numu", "numubar"]

        def_binning = {nu: np.linspace(0, 20, num=201) for nu in self.neutrinos}

        plotting = project_config["Plotting"]

        self.plot_opts = {
            "draw_label": plotting["draw_label"],
            "experiment": plotting["experiment"],
            "stage": plotting["stage"],
            "xlim": plotting.get("neutrino_energy_range", (0.0, 20.0)),
            "flux_prediction_bullets": plotting["flux_prediction_bullets"],
            "enabled": plotting["enabled"],
        }

        binning = project_config.get("Binning", def_binning)

        self.bin_edges = {}

        for nu, bins in binning.items():
            if isinstance(bins, int):
                self.bin_edges[nu] = np.linspace(0, 20, num=bins + 1)
            elif isinstance(bins, list) and len(bins) > 0:
                self.bin_edges[nu] = np.asarray(bins)
            else:
                # Falling back to default binning
                self.bin_edges[nu] = def_binning[nu]

        self.sources_path = Path(project_config["sources"]).expanduser().resolve()

        self.results_path = Path(
            project_config.get(
                "results",
                self.sources_path.parent,
            )
        )

        self.plots_path = Path(
            project_config.get(
                "plots",
                self.sources_path.parent / "plots/",
            )
        )

        output_file = project_config["output_file_name"]

        self.products_file = f"{self.results_path}/{date.today()}_{output_file}"

        self.ppfx = project_config["PPFX"]

        self.nominal_samples = {"fhc": None, "rhc": None}

        sample = project_config.get("sample")
        if sample is not None:
            self.nominal_samples["fhc"] = self.sources_path / sample
            _, self.nominal_run_id = self.parse_filename(self.nominal_samples["fhc"].name)
        else:
            nominal_samples = self.sources_path.glob("*0015*")

            for s in nominal_samples:
                horn, run_id = self.parse_filename(s.name)
                self.nominal_samples |= {horn: s}
                self.nominal_run_id = run_id
                # if "-" in s.name:
                #     self.nominal_samples |= {"rhc": s}
                #     continue
                # self.nominal_samples |= {"fhc": s}

    def verify_paths(self) -> None:
        for path in [self.sources_path, self.results_path, self.plots_path]:
            if not path.exists():
                opt = input(f"{path} does not exist. Create it? (y/n) ").lower()
                if opt == "y":
                    path.mkdir()
                else:
                    print("Directory not created. Exiting...")
                    sys.exit()

        if not any(self.sources_path.iterdir()):
            msg = (
                f'No files found in input directory: "{self.sources_path}"'
                "\nExiting..."
            )
            raise FileNotFoundError(msg)

    @property
    def ignored_histogram_names(self):
        keys = ["hpot"]  # always ignore the POT count
        for k, v in self.ppfx["enabled"].items():
            if v:
                continue
            if k == "thintarget":
                keys += ["hthin_nue", "hthin_numu"]
                continue
            if k == "mippnumi":
                keys.append("mipp")
                continue
            keys.append(k)

        yield from keys

    def ignored_hist_filter(self, hist_name: str) -> bool:
        return not any(
            x.lower() in hist_name.lower() for x in self.ignored_histogram_names
        )

    @staticmethod
    def parse_filename(name: str) -> tuple[str, int]:
        split = name.rsplit("_")
        horn = "rhc" if "-" in split[-1] else "fhc"
        run_id = int(split[-2])
        return horn, run_id

    def itersamples(self):
        for f in self.sources_path.glob("*.root"):
            horn, run_id = self.parse_filename(f.name)
            yield f, horn, run_id

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
