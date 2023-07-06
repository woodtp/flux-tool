import tomllib
from datetime import date
from pathlib import Path

import numpy as np


class AnalysisConfig:
    __slots__ = (
        "bin_edges",
        "nominal_samples",
        "output_file_name",
        "plots_path",
        "ppfx",
        "products_file",
        "results_path",
        "sources_path",
    )

    def __init__(self, project_config: dict) -> None:
        bin_edges = project_config.get("bin_edges")
        if bin_edges is not None:
            if isinstance(bin_edges, float):
                self.bin_edges = np.arange(
                    0.0, 20.0 + bin_edges, bin_edges, dtype=np.float64
                )
            else:
                self.bin_edges = np.asarray(bin_edges)
        else:
            self.bin_edges = np.linspace(0.0, 20.0, num=201)
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

        nominal_samples = self.sources_path.glob("*0015*")

        self.nominal_samples = {"fhc": None, "rhc": None}

        for s in nominal_samples:
            if "-" in s.name:
                self.nominal_samples |= {"rhc": s}
                continue
            self.nominal_samples |= {"fhc": s}

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
        return not any(x.lower() in hist_name.lower() for x in self.ignored_histogram_names)

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
    def from_str(cls, config_str: str):
        config = tomllib.loads(config_str)
        return cls(config)

    @classmethod
    def from_file(cls, config_file: str):
        with open(config_file, "rb") as file:
            config = tomllib.load(file)
        return cls(config)
