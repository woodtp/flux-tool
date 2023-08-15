from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from flux_tool.helpers import (calculate_correlation_matrix,
                               convert_pandas_to_th1)


def smooth_stat_fluctuations(
    df: pd.DataFrame, bin_edges: dict[str, np.ndarray]
) -> pd.DataFrame:
    groups = df.stack("run_id").groupby(by=["run_id", "horn_polarity", "neutrino_mode"])

    smoothed_flux_list = []

    for (_, _, nu), g in groups:  # type: ignore
        th1 = convert_pandas_to_th1(series=g, bin_edges=bin_edges[nu])  # type: ignore

        th1.Smooth()

        smoothed_fluxes = [th1[i[-2]] for i in g.index]  # type: ignore

        series = pd.Series(smoothed_fluxes, index=g.index)

        smoothed_flux_list.append(series)

    return pd.concat(smoothed_flux_list).unstack("run_id")  # type: ignore


@dataclass(repr=False)
class BeamFocusingSystematics:
    beam_flux_df: pd.DataFrame
    bin_edges: dict[str, np.ndarray]
    smoothing: Optional[bool] = False
    nominal_run: pd.DataFrame = field(init=False)
    _beam_pt: pd.DataFrame = field(init=False)
    _run_id_map: dict[str, int | tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self._beam_pt = pd.pivot_table(
            data=self.beam_flux_df.query("category == 'nominal'"),
            values=["flux"],
            index=["horn_polarity", "neutrino_mode", "bin"],
            columns=["run_id"],
        )["flux"]

        self._run_id_map = {
            "beam_power": 1,
            "horn_current_plus": 8,
            "horn1_x": (10, 11),
            "horn1_y": (12, 13),
            "beam_spot": (14, 16),
            "water_layer": (21, 22),
            "beam_shift_x": (24, 25),
            "beam_shift_y_plus": 26,
            "beam_shift_y_minus": 27,
            "beam_div": 32,
        }

        self.nominal_run = self._beam_pt[[15]]

        self.apply_systematic_selection()

    def is_excluded(self, run_id: int) -> bool:
        """Checks if run_id appears in the list of run_ids in the _run_id_map member variable.
        Returns True if it doesn't.
        """
        return not any(
            isinstance(item, int)
            and item == run_id
            or (isinstance(item, tuple) and len(item) == 2 and run_id in item)
            for item in self._run_id_map.values()
        )

    @cached_property
    def flux_shifts(self) -> pd.DataFrame:
        # All runs have a +/- 1sigma flux variant except for runs 30 and 32

        nom_vals = self.nominal_run.values

        beam_shifts = (self._beam_pt - nom_vals).drop(labels=[15], axis=1)

        beam_fractional_shifts = beam_shifts / nom_vals

        beam_fractional_shifts[beam_fractional_shifts.isna()] = 0

        if self.smoothing:
            beam_fractional_shifts = smooth_stat_fluctuations(
                beam_fractional_shifts, self.bin_edges
            )
            beam_shifts = beam_fractional_shifts * nom_vals

        flux_shifts_df = pd.concat(
            [beam_shifts, beam_fractional_shifts], keys=["absolute", "fractional"]
        )

        return flux_shifts_df

    def energy_to_bin_slice(self, elow, ehigh):
        index1 = np.argmax(self.bin_edges["numu"] >= elow)
        index2 = np.argmax(self.bin_edges["numu"] >= ehigh)
        return slice(index1, index2)

    def apply_systematic_selection(self):
        ids_to_drop = filter(self.is_excluded, self.flux_shifts.columns)
        self.flux_shifts.drop(ids_to_drop, axis=1, inplace=True)

        water_layer_indexer = (
            slice(None),
            slice(None),
            slice(None),
            self.energy_to_bin_slice(1.0, 20.0),
        ), (21, 22)
        self.flux_shifts.loc[water_layer_indexer] *= 0.0

        div_indexer = (
            slice(None),
            slice(None),
            slice(None),
            self.energy_to_bin_slice(0, 1.0),
        ), 32
        self.flux_shifts.loc[div_indexer] *= 0.0

    @cached_property
    def beam_systematic_shifts(self):
        flux_systs = {}
        shifts = self.flux_shifts.loc["absolute"]
        for key, run_id in self._run_id_map.items():
            if isinstance(run_id, tuple):
                id1, id2 = run_id
                flux_systs[key] = 0.5 * (shifts[id1] + shifts[id2])
                continue
            flux_systs[key] = shifts[run_id]

        df = pd.DataFrame(flux_systs, index=shifts.index)

        df_frac = df / self.nominal_run.values
        df_frac[df_frac.isna()] = 0

        beam_shifts = pd.concat(
            [df, df_frac],
            keys=["absolute", "fractional"],
            names=["scale"] + df.index.names,
        )

        beam_shifts.columns.name = "category"

        beam_shifts = (
            beam_shifts.stack("category")
            .reorder_levels(
                ["scale", "category", "horn_polarity", "neutrino_mode", "bin"]
            )
            .sort_index()
            .replace(-0.0, 0)
        )

        return beam_shifts

    @cached_property
    def covariance_matrices(self) -> pd.DataFrame:
        flux_shifts = self.beam_systematic_shifts.loc["absolute"]

        def _calc_cov(x):
            x = x.droplevel("category")
            outer = np.outer(x, x)
            return pd.DataFrame(outer, index=x.index, columns=x.index)

        covs = flux_shifts.groupby("category").apply(_calc_cov).replace(-0.0, 0.0)

        nom_mat = np.outer(self.nominal_run, self.nominal_run)

        grp_divide = lambda grp: np.divide(
            grp.droplevel(0), nom_mat, out=np.zeros_like(grp), where=nom_mat != 0
        )

        covs_frac = covs.groupby(level=0).apply(grp_divide)

        beam_covariance_matrices = pd.concat(
            [covs, covs_frac],
            keys=["absolute", "fractional"],
            names=["scale", "category", "horn_polarity", "neutrino_mode", "bin"],
        )

        # beam_covariance_matrices = beam_covariance_matrices.replace(
        #     -0.0, 0.0
        # ).sort_index()

        return beam_covariance_matrices

    @cached_property
    def correlation_matrices(self) -> pd.DataFrame:
        mat_groups = self.covariance_matrices.loc["absolute"].groupby(
            "category", group_keys=True
        )

        return mat_groups.apply(calculate_correlation_matrix)

    @cached_property
    def total_covariance_matrix(self) -> pd.DataFrame:
        covs_abs = self.covariance_matrices.loc["absolute"]
        beam_total_covariance_matrix_abs = (
            covs_abs.iloc[covs_abs.index.get_level_values("category") != "beam_power"]
            .groupby(level=["horn_polarity", "neutrino_mode", "bin"])
            .sum()
        )

        beam_total_covariance_matrix_frac = beam_total_covariance_matrix_abs / np.outer(
            self.nominal_run, self.nominal_run
        )

        total_covariance_matrix = pd.concat(
            [beam_total_covariance_matrix_frac, beam_total_covariance_matrix_abs],
            keys=["fractional", "absolute"],
            names=["scale", "horn_polarity", "neutrino_mode", "bin"],
        )

        return total_covariance_matrix

    @cached_property
    def total_correlation_matrix(self) -> pd.DataFrame | NDArray[Any]:
        corr = calculate_correlation_matrix(
            self.total_covariance_matrix.loc["absolute"]
        )
        return corr

    @cached_property
    def fractional_uncertainties(self) -> pd.DataFrame | pd.Series:
        levels = ["category", "horn_polarity", "neutrino_mode", "bin"]

        total_cov = self.total_covariance_matrix.loc["fractional"]
        total_cov["category"] = "total"
        total_cov = total_cov.set_index("category", append=True).reorder_levels(levels)

        covs = pd.concat([self.covariance_matrices.loc["fractional"], total_cov])

        cov_groups = covs.groupby(level=levels[0])

        # for some reason I get -0.0 for some values on the diagonal, so I'm wrapping in np.abs
        diag_sqrt = lambda x: pd.Series(np.sqrt(np.abs(np.diag(x))), index=covs.columns)

        return cov_groups.apply(diag_sqrt).stack(levels[1:]).sort_index()  # type: ignore
