from dataclasses import dataclass, field
from functools import cached_property
from uuid import uuid4

import numpy as np
import pandas as pd
from ROOT import TF1, TH1D

from flux_tool.config import AnalysisConfig  # type: ignore


@dataclass(repr=False)
class FluxUniverseFit:
    flux_universes: pd.Series
    _flux_th1: TH1D = field(init=False)
    _fit_function: TF1 = field(init=False)

    def __post_init__(self) -> None:
        self._fit_function, self._flux_th1 = self.fit()

    def fit(self) -> tuple[TF1, TH1D]:
        hist_title = ";#phi_{#nu} (m^{-2} POT^{-1});PPFX Universes"
        n_universes = self.flux_universes.shape[0]
        nbins = int(np.ceil(np.sqrt(n_universes))) + 2
        xmin = np.amin(self.flux_universes)
        xmax = np.amax(self.flux_universes)

        flux_th1 = TH1D(str(uuid4()), hist_title, nbins, xmin, xmax)

        for uni in self.flux_universes:
            flux_th1.Fill(uni)

        fit_function = TF1(str(uuid4()), "gaus", xmin, xmax)

        flux_th1.Fit(fit_function, "0Q")

        return fit_function, flux_th1

    def eval_fit_function(self, x) -> float:
        return self._fit_function.Eval(x)

    @cached_property
    def flux_histogram(self) -> TH1D:
        return self._flux_th1

    @cached_property
    def universe_mean(self) -> float:
        # return np.mean(self.flux_universes)
        return self.flux_universes.mean()  # type: ignore

    @cached_property
    def universe_sigma(self) -> float:
        # return np.std(self.flux_universes)
        return self.flux_universes.std()  # type: ignore

    @cached_property
    def fit_mean(self) -> float:
        return self._fit_function.GetParameter(1)

    @cached_property
    def fit_sigma(self) -> float:
        return self._fit_function.GetParameter(2)

    @cached_property
    def mean_fractional_error(self) -> float:
        frac_err = np.abs(self.fit_mean - self.universe_mean) / self.universe_mean
        return frac_err

    @cached_property
    def sigma_fractional_error(self) -> float:
        frac_err = np.abs(self.fit_sigma - self.universe_sigma) / self.universe_sigma
        return frac_err

    @cached_property
    def chi2ndf(self) -> float:
        return self._fit_function.GetChisquare() / self._fit_function.GetNDF()


@dataclass(repr=False)
class HadronProductionSystematics:
    ppfx_dataframe: pd.DataFrame
    nominal_dataframe: pd.DataFrame
    cfg: AnalysisConfig
    _flux_pt: pd.DataFrame = field(init=False)
    _nom_pt: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        index = ("category", "horn_polarity", "neutrino_mode", "bin", "universe")
        flux_pt = pd.pivot_table(self.ppfx_dataframe, index=index, values="flux")
        nom_pt = pd.pivot_table(
            self.nominal_dataframe.loc[
                (self.nominal_dataframe["run_id"] == self.cfg.nominal_id )
                & (self.nominal_dataframe["category"] == "nominal")
            ],
            values="flux",
            index=("horn_polarity", "neutrino_mode", "bin"),
        )
        assert not flux_pt.empty, "No flux data found in the PPFX dataframe."
        assert not nom_pt.empty, "No flux data found in the beam samples dataframe."
        self._flux_pt = flux_pt["flux"]  # type: ignore
        self._nom_pt = nom_pt["flux"]    # type: ignore

    @cached_property
    def ppfx_corrected_flux(self) -> pd.DataFrame:
        mean = self._flux_pt.groupby(level=self._flux_pt.index.names[:4]).mean()
        sigma = self._flux_pt.groupby(level=self._flux_pt.index.names[:4]).std()

        return pd.concat([mean, sigma], axis=1, keys=["mean", "sigma"])  # type: ignore

    @cached_property
    def ppfx_flux_weights(self) -> pd.DataFrame:
        total_correction = self.ppfx_corrected_flux.loc["total", "mean"]
        weights = total_correction / self._nom_pt
        return weights.fillna(0)

    @cached_property
    def covariance_matrices(self) -> pd.DataFrame:
        mean = self.ppfx_corrected_flux["mean"]

        pt_frac = self._flux_pt.div(mean)

        group_abs = self._flux_pt.unstack(
            ("horn_polarity", "neutrino_mode", "bin")
        ).groupby(level="category")

        group_frac = pt_frac.unstack(("horn_polarity", "neutrino_mode", "bin")).groupby(
            level="category"
        )

        cov_abs = group_abs.cov()  # type: ignore
        cov_frac = group_frac.cov()  # type: ignore

        cov = pd.concat(
            [cov_frac, cov_abs],
            keys=["fractional", "absolute"],
            names=["scale"] + cov_abs.index.names,
        ).fillna(0)

        return cov

    @cached_property
    def correlation_matrices(self) -> pd.DataFrame:
        group = self._flux_pt.unstack(
            ("horn_polarity", "neutrino_mode", "bin")
        ).groupby(level="category")

        corr = group.corr().fillna(0)  # type: ignore

        return corr

    @cached_property
    def fractional_uncertainties(self) -> pd.DataFrame:
        cov = self.covariance_matrices.loc["fractional"]

        uncerts = (
            cov.groupby("category")
            .apply(lambda x:  pd.Series(np.sqrt(np.diag(x)), index=cov.columns))
            # TODO remove future_stack=True in pandas ^3.0.0
            .stack(["horn_polarity", "neutrino_mode", "bin"], future_stack=True)
            .sort_index()
        )

        return uncerts

    @cached_property
    def flux_fit_results(self) -> dict[tuple[str, str, int], FluxUniverseFit]:
        bins_df = pd.pivot_table(
            data=self.ppfx_dataframe.query("category == 'total'"),
            index=["horn_polarity", "neutrino_mode", "bin", "universe"],
            values="flux",
        )["flux"]

        flux_fits = {}
        for idx, b in bins_df.groupby(level=["horn_polarity", "neutrino_mode", "bin"]):
            flux_fits |= {idx: FluxUniverseFit(b.droplevel(0))}  # type: ignore

        return flux_fits
