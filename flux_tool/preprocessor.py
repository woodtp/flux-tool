import logging
from concurrent.futures import ProcessPoolExecutor as Executor
from functools import partial

import pandas as pd

from flux_tool.config import AnalysisConfig
from flux_tool.normalize_and_rebin_data import normalize_flux_to_pot


class Preprocessor:
    __slots__ = ("nominal_flux_df", "ppfx_correction_df")

    def __init__(self, cfg: AnalysisConfig) -> None:
        logging.info("Beginning preprocessing...")
        with Executor() as executor:
            res = executor.map(
                partial(
                    normalize_flux_to_pot,
                    bin_edges=cfg.bin_edges,
                    hist_name_filter=cfg.ignored_hist_filter,
                ),
                *zip(*list(cfg.itersamples())),
            )

        df = pd.concat(res)
        self.nominal_flux_df = df.loc[df["universe"].isna()].drop("universe", axis=1)
        self.ppfx_correction_df = df.loc[
            (df["run_id"] == 15) & (df["universe"].notna())
        ]
