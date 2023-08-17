import logging
import sys
import tomllib
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Self

import numpy as np


class AnalysisConfig:
    __slots__ = (
        "bin_edges",
        "neutrinos",
        "nominal_samples",
        "output_file_name",
        "plots_path",
        "ppfx",
        "products_file",
        "results_path",
        "sources_path",
    )

    def __init__(self, project_config: dict) -> None:
        self.neutrinos: list[str] = ["nue", "nuebar", "numu", "numubar"]

        def_binning = {nu: np.linspace(0, 20, num=201) for nu in self.neutrinos}

        binning = project_config.get("Binning", def_binning)

        self.bin_edges = {}

        for nu, bins in binning.items():
            if isinstance(bins, int):
                self.bin_edges[nu] = np.linspace(0, 20, num=bins + 1)
            elif isinstance(bins, Iterable):
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

        self.verify_paths()

        output_file = project_config["output_file_name"]

        self.products_file = f"{self.results_path}/{date.today()}_{output_file}"

        self.ppfx = project_config["PPFX"]

        nominal_samples = self.sources_path.glob("*0015*")

        self.nominal_samples = {"fhc": None, "rhc": None}

        for s in nominal_samples:
            if "-" in s.name:
                self.nominal_samples |= {"rhc": s}
                continue
            self.nominal_samples |= {"fhc": s}

    def verify_paths(self) -> None:
        for path in [self.sources_path, self.results_path, self.plots_path]:
            if not path.exists():
                opt = input(
                    f"{self.results_path} does not exist. Create it? (y/n) "
                ).lower()
                if opt == "y":
                    self.results_path.mkdir()
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
