import logging
from concurrent.futures import ProcessPoolExecutor as Executor
from functools import partial

import pandas as pd
from rich.progress import Progress

from flux_tool.config import AnalysisConfig
from flux_tool.normalize_and_rebin_data import normalize_flux_to_pot


class Preprocessor:
    __slots__ = ("nominal_flux_df", "ppfx_correction_df")

    def __init__(self, cfg: AnalysisConfig) -> None:
        logging.info("\n=============== BEGINNING PREPROCESSING ===============")
        logging.info(
            "Reading input files, normalized to POT, and rebinning, if necessary"
        )
        jobs = list(cfg.itersamples())
        fn = partial(
            normalize_flux_to_pot,
            bin_edges=cfg.bin_edges,
            hist_name_filter=cfg.enabled_hist_filter,
        )
        results = []
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Working...", total=len(jobs))
            with Executor() as executor:
                for job in jobs:
                    future = executor.submit(fn, *job)
                    future.add_done_callback(lambda _: progress.advance(task_id))
                    results.append(future.result())

        df = pd.concat(results)
        self.nominal_flux_df = df.loc[df["universe"].isna()].drop("universe", axis=1)
        self.ppfx_correction_df = df.loc[(df["run_id"] == cfg.nominal_id) & ~df["universe"].isna()]
