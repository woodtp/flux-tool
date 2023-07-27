from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd

from flux_tool.helpers import (calculate_correlation_matrix,
                               convert_pandas_to_th1)


def smooth_stat_fluctuations(df: pd.DataFrame, bin_edges: np.ndarray) -> pd.DataFrame:
    groups = df.stack("run_id").groupby(by=["run_id", "horn_polarity", "neutrino_mode"])

    smoothed_flux_list = []

    for _, g in groups:
        th1 = convert_pandas_to_th1(series=g, bin_edges=bin_edges)

        th1.Smooth()

        smoothed_fluxes = [th1[i[-2]] for i in g.index]

        series = pd.Series(smoothed_fluxes, index=g.index)

        smoothed_flux_list.append(series)

    return pd.concat(smoothed_flux_list).unstack("run_id")


@dataclass(repr=False)
class BeamFocusingSystematics:
    beam_flux_df: pd.DataFrame
    bin_edges: np.ndarray
    smoothing: Optional[bool] = False
    nominal_run: pd.DataFrame = field(init=False)
    _beam_pt: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self._beam_pt = pd.pivot_table(
            data=self.beam_flux_df.query("category == 'nominal'"),
            values=["flux"],
            index=["horn_polarity", "neutrino_mode", "bin"],
            columns=["run_id"],
        )["flux"]

        self.nominal_run = self._beam_pt[[15]]

        self.apply_systematic_selection()

    @cached_property
    def flux_shifts(self) -> pd.DataFrame:
        # ncol = len(self._beam_pt.columns)

        # All runs have a +/- 1sigma flux variant except for runs 30 and 32
        # scale_factors = (ncol - 3) * [0.5] + 2 * [1.0]

        nom_vals = self.nominal_run.values

        beam_shifts = (self._beam_pt - nom_vals).drop(
            labels=[15], axis=1
        )  # * scale_factors

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
        index1 = np.argmax(self.bin_edges >= elow)
        index2 = np.argmax(self.bin_edges >= ehigh)
        return slice(index1, index2)

    def apply_systematic_selection(self):
        self.flux_shifts.drop([9, 17, 18, 19, 20, 28, 29, 30], axis=1, inplace=True)

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
    def covariance_matrices(self) -> pd.DataFrame:
        flux_shifts = self.flux_shifts.loc["absolute"]

        beam_covs_abs = pd.concat(
            [
                pd.DataFrame(np.outer(df, df), index=df.index, columns=df.index)
                for _, df in flux_shifts.items()
            ],
            keys=flux_shifts.columns,
        )

        covs = {
            "beam_power": beam_covs_abs.loc[1],
            # "horn_current": 0.5 * (beam_covs_abs.loc[8] + beam_covs_abs.loc[9]),
            "horn_current_plus": beam_covs_abs.loc[8],
            "horn1_x": 0.5 * (beam_covs_abs.loc[10] + beam_covs_abs.loc[11]),
            "horn1_y": 0.5 * (beam_covs_abs.loc[12] + beam_covs_abs.loc[13]),
            "beam_spot": 0.5 * (beam_covs_abs.loc[14] + beam_covs_abs.loc[16]),
            # "horn2_x": beam_covs_abs.loc[17] + beam_covs_abs.loc[18],
            # "horn2_y": beam_covs_abs.loc[19] + beam_covs_abs.loc[20],
            "water_layer": 0.5 * (beam_covs_abs.loc[21] + beam_covs_abs.loc[22]),
            "beam_shift_x": 0.5 * (beam_covs_abs.loc[24] + beam_covs_abs.loc[25]),
            # "beam_shift_y": beam_covs_abs.loc[26] + beam_covs_abs.loc[27],
            "beam_shift_plus_y": beam_covs_abs.loc[26],
            "beam_shift_minus_y": beam_covs_abs.loc[27],
            # "target_z": 0.5 * (beam_covs_abs.loc[28] + beam_covs_abs.loc[29]),
            # "B_field": beam_covs_abs.loc[30],
            "beam_div": beam_covs_abs.loc[32],
        }

        nom_mat = np.outer(self.nominal_run, self.nominal_run)

        covs_frac = {k: v / nom_mat for k, v in covs.items()}

        beam_covs_combined = pd.concat(
            list(covs.values()), keys=covs.keys(), names=beam_covs_abs.index.names
        )
        beam_covs_combined_frac = pd.concat(
            list(covs_frac.values()),
            keys=covs_frac.keys(),
            names=beam_covs_abs.index.names,
        )

        beam_covariance_matrices = pd.concat(
            [beam_covs_combined_frac, beam_covs_combined],
            keys=["fractional", "absolute"],
            names=["scale", "category", "horn_polarity", "neutrino_mode", "bin"],
        )

        return beam_covariance_matrices

    @cached_property
    def correlation_matrices(self) -> pd.DataFrame:
        mat_groups = self.covariance_matrices.loc["absolute"].groupby(
            "category", group_keys=True
        )

        return mat_groups.apply(calculate_correlation_matrix)

    @cached_property
    def total_covariance_matrix(self) -> pd.DataFrame:
        beam_total_covariance_matrix_abs = (
            self.covariance_matrices.loc["absolute"]
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
    def total_correlation_matrix(self) -> pd.DataFrame:
        corr = calculate_correlation_matrix(
            self.total_covariance_matrix.loc["absolute"]
        )

        return corr

    @cached_property
    def fractional_uncertainties(self) -> pd.DataFrame:
        frac_uncerts = []

        cov_groups = self.covariance_matrices.loc["fractional"].groupby(
            level=("category", "horn_polarity", "neutrino_mode")
        )
        for idx, mat in cov_groups:
            diag = np.clip(
                a=np.diag(mat[idx[1:]]),
                a_min=0,
                a_max=None,
            )
            sigmas = np.sqrt(diag)
            uncerts = pd.Series(sigmas, index=mat.index, name=idx[0])
            frac_uncerts.append(uncerts)

        total_uncerts = []

        total_cov_groups = self.total_covariance_matrix.loc["fractional"].groupby(
            level=["horn_polarity", "neutrino_mode"]
        )

        for idx, mat in total_cov_groups:
            sigmas = np.sqrt(np.diag(mat[idx]))
            uncerts = pd.Series(sigmas, index=mat.index, name="total")
            total_uncerts.append(uncerts)

        frac_uncerts_df = pd.concat(frac_uncerts).unstack("category")
        total_uncerts_df = pd.concat(total_uncerts)

        fractional_uncertainties = pd.concat(
            [frac_uncerts_df, total_uncerts_df], axis=1
        )

        return fractional_uncertainties
