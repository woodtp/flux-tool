import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ROOT import TH2D, TAxis  # type: ignore

from flux_tool import uncertainty
from flux_tool.beam_focusing_systematics import BeamFocusingSystematics
from flux_tool.config import AnalysisConfig
from flux_tool.hadron_production_systematics import HadronProductionSystematics
from flux_tool.helpers import (calculate_correlation_matrix,
                               convert_groups_to_dict, convert_pandas_to_th1,
                               convert_pandas_to_th2,
                               convert_symmetric_ndarray_to_tmatrix)
from flux_tool.principal_component_analysis import PCA


class FluxSystematicsAnalysis:
    __slots__ = (
        "beam_systematics",
        "beam_systematics_is_initialized",
        "bin_edges",
        "cfg",
        "flux_prediction",
        "hadron_systematics",
        "horn_modes",
        "nominal_flux_df",
        "pca_eigen_values",
        "pca_components",
        "pca_covariance_matrix",
        "ppfx_correction_df",
        "statistical_uncertainties",
        "stat_uncert_matrix",
        "th2_bins",
        "total_covariance_matrix",
    )

    def __init__(
        self,
        nominal_flux_df: pd.DataFrame,
        ppfx_correction_df: pd.DataFrame,
        bin_edges: dict[str, np.ndarray],
        cfg: AnalysisConfig,
    ) -> None:
        self.nominal_flux_df = nominal_flux_df
        self.ppfx_correction_df = ppfx_correction_df
        self.bin_edges = bin_edges
        self.cfg = cfg
        self.beam_systematics_is_initialized = (
            len(self.nominal_flux_df["run_id"].unique()) > 1
        )
        self.horn_modes = list(self.nominal_flux_df["horn_polarity"].unique())

    def run(self, pca_threshold: float = 1, apply_smoothing: bool = False, enable_beam_selection: bool = False) -> None:
        """Run the analysis.
        Parameters
        ----------
            pca_threshold : float, optional
                The fractional variance threshold for the PCA above which components are discarded.
                Takes a value between 0 and 1, where '1' keeps all components (default is 1)
            apply_smoothing : bool, optional
                Whether to apply smoothing to the beam fluxes (default is false)
            enable_beam_selection : bool, optional
                Enables the predefined beam systematics handling outlined in SBN-doc-27384-v6 (default is false)
        Returns
        -------
            None
        """
        logging.info("\n=============== BEGINNING ANALYSIS ===============")

        logging.info(
            "Calculating flux correction and hadron production systematic uncertainties..."
        )
        self.hadron_systematics = HadronProductionSystematics(
            self.ppfx_correction_df, self.nominal_flux_df, self.cfg
        )

        logging.info("Reading statistical uncertainties...")
        statistical_uncertainties = uncertainty.extract_statistical_uncertainties(
            self.nominal_flux_df, self.hadron_systematics.ppfx_flux_weights
        )

        statistical_uncertainties_fraction = (
            uncertainty.extract_statistical_uncertainties(
                self.nominal_flux_df,
                self.hadron_systematics.ppfx_flux_weights,
                normalized=True,
            )
        )

        diag = np.power(np.diag(statistical_uncertainties), 2)

        self.stat_uncert_matrix = pd.DataFrame(
            diag,
            index=statistical_uncertainties.index,
            columns=statistical_uncertainties.index,
        )

        self.statistical_uncertainties = pd.concat(
            [statistical_uncertainties, statistical_uncertainties_fraction],
            keys=["absolute", "fractional"],
            names=["scale"] + statistical_uncertainties.index.names,
        )

        self.th2_bins = np.arange(
            self.hadron_systematics.correlation_matrices.loc["total"].shape[0] + 1
        )

        if self.beam_systematics_is_initialized:
            logging.info("Calculating beamline focusing systematic uncertainties...")
            self.beam_systematics = BeamFocusingSystematics(
                beam_flux_df=self.nominal_flux_df,
                bin_edges=self.bin_edges,
                cfg=self.cfg,
                smoothing=apply_smoothing,
                enable_selection=enable_beam_selection,
            )

        total_cov = self.hadron_systematics.covariance_matrices.loc[
            ("fractional", "total")
        ]

        pca = PCA(total_cov, threshold=pca_threshold)  # type: ignore

        logging.info("Performing PCA on hadron production covariance matrix...")
        pca.fit()

        self.pca_covariance_matrix = pd.DataFrame(
            pca.new_covariance_matrix, index=total_cov.index, columns=total_cov.columns
        )

        self.pca_eigen_values = pca.eigenvalues_df
        self.pca_components = pca.principal_component_df

        self.total_covariance_matrix = self._total_covariance_matrix

        self.flux_prediction = self._flux_prediction

    def rescale_matrix(self, matrix, fractional=True):
        """Helper function to convert between absolute and fractional scales of the covariance matrices.
        If fractional=True (default), then the input matrix is presumed to be in the fractional scale and will be multipled by flux to return to absolute scale.
        """
        flux = self.hadron_systematics.ppfx_corrected_flux.loc["total", "mean"]
        scale_factor = np.outer(flux, flux)
        if not fractional:
            scale_factor = np.reciprocal(
                scale_factor,
                out=np.zeros(shape=scale_factor.shape),
                where=scale_factor != 0,
            )
        return matrix * scale_factor

    @property
    def xaxis_variable_bins(self) -> TAxis:
        xbins = self.bin_edges["numu"]
        nbins = xbins.shape[0] - 1

        axis = TAxis(nbins, xbins)
        axis.SetTitle("E_{#nu} [GeV]")

        return axis

    @property
    def matrix_taxis(self) -> TAxis:
        matrix = self.total_covariance_matrix
        rows = matrix.index
        columns = matrix.columns

        xbins = np.arange(0, len(rows) + 1, dtype=float)
        nbinsx = xbins.shape[0] - 1

        taxis = TAxis(nbinsx, xbins)

        it = zip(enumerate(rows), enumerate(columns))

        for (ii, row), (jj, col) in it:
            labelx = f"{row[0]}-{row[1]}-{row[2]}"
            labely = f"{col[0]}-{col[1]}-{col[2]}"
            taxis.SetBinLabel(ii + 1, labelx)
            taxis.SetBinLabel(jj + 1, labely)

        return taxis

    @property
    def matrix_binning_str(self) -> str:
        index = self.total_covariance_matrix.index
        bins = self.bin_edges["numu"]

        nu_pdg = {"nue": 12, "nuebar": -12, "numu": 14, "numubar": -14}

        lines = ["variables: isRHC NeutrinoCode Enu Enu"]
        for horn, nu, b in index:
            is_RHC = 1 if horn == "rhc" else 0
            lines.append(f"{is_RHC} {nu_pdg[nu]} {bins[b-1]} {bins[b]}")

        return "\n".join(lines)

    @property
    def _total_covariance_matrix(self) -> pd.DataFrame:
        logging.info("Computing total covariance matrix...")
        hp_mat = self.hadron_systematics.covariance_matrices.loc["absolute", "total"]
        total_mat = hp_mat + self.stat_uncert_matrix
        if self.beam_systematics_is_initialized:
            beam_tot_cov = self.beam_systematics.total_covariance_matrix.loc["absolute"]
            # beam_power_cov = self.beam_systematics.covariance_matrices.loc[
            #     "absolute", "beam_power"
            # ]
            return total_mat + beam_tot_cov # + beam_power_cov
        return total_mat

    @property
    def total_correlation_matrix(self) -> pd.DataFrame | NDArray[Any]:
        corr_mat = calculate_correlation_matrix(self.total_covariance_matrix)
        return corr_mat

    @property
    def _flux_prediction(self) -> pd.DataFrame:
        total_sigma = pd.Series(
            np.sqrt(np.diag(self.total_covariance_matrix)),
            index=self.total_covariance_matrix.index,
            name="sigma",
        )

        ppfx_mean = self.hadron_systematics.ppfx_corrected_flux.loc["total", "mean"]

        return pd.concat([ppfx_mean, total_sigma], axis=1)

    def total_uncertainties_in_range(
        self, elow, ehigh, mat: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if mat is None:
            mat = self.total_covariance_matrix

        bin_slices = {}
        for nu, bins in self.bin_edges.items():
            bins_selected = np.where((bins >= elow) & (bins <= ehigh))[0]
            if bins_selected.shape == bins.shape:
                bin_slice = slice(None)
            else:
                bin_slice = slice(bins_selected[0], bins_selected[-1])
            bin_slices[nu] = bin_slice

        df_list = []

        for horn in self.horn_modes:
            index_slice = pd.IndexSlice[horn, :, bin_slices["numu"]]

            cov_mat = (
                mat.loc[index_slice, index_slice]
                .droplevel(level="horn_polarity", axis=0)
                .droplevel(level="horn_polarity", axis=1)
            )

            nue_flux = self.flux_prediction.loc[
                (horn, "nue", bin_slices["nue"]), "mean"
            ].sum()

            nuebar_flux = self.flux_prediction.loc[
                (horn, "nuebar", bin_slices["nuebar"]), "mean"
            ].sum()

            numu_flux = self.flux_prediction.loc[
                (horn, "numu", bin_slices["numu"]), "mean"
            ].sum()

            numubar_flux = self.flux_prediction.loc[
                (horn, "numubar", bin_slices["numubar"]), "mean"
            ].sum()

            total_nue_flux = nue_flux + nuebar_flux
            total_numu_flux = numu_flux + numubar_flux

            ratio_uncert = uncertainty.ratio_uncertainty(
                cov=cov_mat,
                total_nue_flux=total_nue_flux,
                total_numu_flux=total_numu_flux,
            )

            nue_uncert = uncertainty.flux_uncertainty(
                cov=cov_mat.loc["nue", "nue"], total_flux=nue_flux
            )

            nuebar_uncert = uncertainty.flux_uncertainty(
                cov=cov_mat.loc["nuebar", "nuebar"], total_flux=nuebar_flux
            )

            nue_nuebar_uncert = uncertainty.flux_uncertainty(
                cov=cov_mat.loc[slice("nue", "nuebar"), slice("nue", "nuebar")],
                total_flux=total_nue_flux,
            )

            numu_uncert = uncertainty.flux_uncertainty(
                cov=cov_mat.loc["numu", "numu"], total_flux=numu_flux
            )

            numubar_uncert = uncertainty.flux_uncertainty(
                cov=cov_mat.loc["numubar", "numubar"], total_flux=numubar_flux
            )

            numu_numubar_uncert = uncertainty.flux_uncertainty(
                cov=cov_mat.loc[slice("numu", "numubar"), slice("numu", "numubar")],
                total_flux=total_numu_flux,
            )

            res_dict = {
                "nue": [nue_uncert],
                "nuebar": [nuebar_uncert],
                "nue+nuebar": [nue_nuebar_uncert],
                "numu": [numu_uncert],
                "numubar": [numubar_uncert],
                "numu+numubar": [numu_numubar_uncert],
                "nue+nuebar/numu+numubar": [ratio_uncert],
            }

            df = pd.DataFrame(res_dict)
            df_list.append(df)

        return pd.concat(df_list, keys=["fhc", "rhc"]).droplevel(level=-1)

    @property
    def total_uncertainty_table(self) -> pd.DataFrame:
        enu_min, enu_max = 0, 20

        uncertainties = [
            self.total_uncertainties_in_range(
                enu_min,
                enu_max,
                self.hadron_systematics.covariance_matrices.loc["absolute", "total"],
            )
        ]

        if self.beam_systematics_is_initialized:
            uncertainties += [
                self.total_uncertainties_in_range(
                    enu_min,
                    enu_max,
                    self.beam_systematics.total_covariance_matrix.loc["absolute"],
                ),
                # self.total_uncertainties_in_range(
                #     enu_min,
                #     enu_max,
                #     self.beam_systematics.covariance_matrices.loc[
                #         "absolute", "beam_power"
                #     ],
                # ),
            ]

        uncertainties += [
            self.total_uncertainties_in_range(
                enu_min, enu_max, self.stat_uncert_matrix
            ),
            self.total_uncertainties_in_range(enu_min, enu_max),
        ]

        table_keys = [
            "Hadron",
            "Beamline",
            # "Beam Power Upgrade",
            "Statistical",
            "Total",
        ]

        uncert_table = pd.concat(uncertainties, keys=table_keys)

        uncert_table.columns = [
            r"$\nu_e$",
            r"$\bar{\nu}_e$",
            r"$\nu_e + \bar{\nu}_e$",
            r"$\nu_\mu$",
            r"$\bar{\nu}_\mu$",
            r"$\nu_\mu + \bar{\nu}_\mu$",
            r"$\frac{\nu_e + \bar{\nu}_e}{\nu_\mu + \bar{\nu}_\mu}$",
        ]

        return uncert_table

    @property
    def total_uncertainty_table_latex(self) -> str:
        table = self.total_uncertainty_table

        table *= 100  # convert to percentage
        table_str = table.to_latex(float_format="%.2f")
        table_str = table_str.replace("fhc", "FHC")
        table_str = table_str.replace("rhc", "RHC")

        return table_str

    @staticmethod
    def _export_matrices(
        matrix: pd.DataFrame,
        title_gen: str | Callable[[str], str],
    ) -> dict[str, TH2D]:
        if isinstance(title_gen, str):
            return {
                title_gen: convert_pandas_to_th2(matrix, hist_name=title_gen),
                title_gen.replace("hcov", "cov").replace(
                    "hcorr", "corr"
                ): convert_symmetric_ndarray_to_tmatrix(matrix.values),
            }

        export_dict = {}

        for category, hist in matrix.groupby(level=0):
            hist_title = title_gen(str(category))
            h = hist.droplevel(0)
            export_dict[hist_title] = convert_pandas_to_th2(h, hist_name=hist_title)
            mat_title = hist_title.replace("hcov", "cov").replace("hcorr", "corr")
            export_dict[mat_title] = convert_symmetric_ndarray_to_tmatrix(hist.values)

        return export_dict

    def _export_nominal_flux_df(self):
        pt = pd.pivot_table(
            self.nominal_flux_df,
            index=("run_id", "category", "horn_polarity", "neutrino_mode", "bin"),
            values=["flux", "stat_uncert"],
        )

        export_dict = {}

        for idx, spectra in pt.groupby(
            level=("run_id", "category", "horn_polarity", "neutrino_mode")
        ):
            run, cat, horn, nu = idx  # type: ignore

            directory = f"beam_samples/run_{run}/"

            hist_name = "h"

            if cat == "central_value":
                hist_name += "cv"
            elif cat == "nominal":
                hist_name += "nom"
            else:
                hist_name += f"nom_{cat}"
                directory += "parent/"

            hist_name += f"_{horn}_{nu}"

            th1 = convert_pandas_to_th1(
                series=spectra["flux"],
                bin_edges=self.bin_edges[nu],
                hist_name=directory + hist_name,
                uncerts=spectra["stat_uncert"],
            )
            th1.SetTitle(";E_{#nu} [GeV]; #Phi_{#nu} [m^{-2} POT^{-1}]")

            export_dict[directory + hist_name] = th1

        return export_dict

    def _export_flux_universes(self):
        df = self.ppfx_correction_df.drop("run_id", axis=1)
        df["bin"] = df["bin"].astype(str)
        df = pd.pivot_table(
            df,
            index="universe",
            columns=("category", "horn_polarity", "neutrino_mode", "bin"),
            values=["flux"],
        )["flux"]
        df.columns = df.columns.map("_".join)

        return {"flux_universes": df}

    def get_products(self) -> dict[str, Any]:
        product_dict = {}

        product_dict["matrix_axis"] = self.matrix_taxis

        product_dict["xaxis_variable_bins"] = self.xaxis_variable_bins

        stat_mat_title = "hstatistical_uncertainty_matrix"

        stat_uncert_mat = convert_pandas_to_th2(
            self.stat_uncert_matrix, hist_name=stat_mat_title
        )

        stat_uncert_tmatrix = convert_symmetric_ndarray_to_tmatrix(
            self.stat_uncert_matrix.values
        )

        product_dict[f"statistical_uncertainties/{stat_mat_title}"] = stat_uncert_mat

        product_dict[
            "statistical_uncertainties/statistical_uncertainty_matrix"
        ] = stat_uncert_tmatrix

        product_dict |= self._export_nominal_flux_df()

        product_dict |= self._export_flux_universes()

        levels = ["horn_polarity", "neutrino_mode"]
        three_levels = ["category"] + levels

        stat_uncrt_abs_groups = self.statistical_uncertainties.loc["absolute"].groupby(
            level=levels
        )

        stat_uncrt_frac_groups = self.statistical_uncertainties.loc[
            "fractional"
        ].groupby(level=levels)

        had_uncrt_groups = self.hadron_systematics.fractional_uncertainties.groupby(
            level=three_levels
        )

        flux_weights_groups = self.hadron_systematics.ppfx_flux_weights.groupby(
            level=levels
        )

        ppfx_corrected_flux_groups = (
            self.hadron_systematics.ppfx_corrected_flux.groupby(level=three_levels)
        )

        flux_prediction_groups = self.flux_prediction.groupby(level=levels)

        product_dict |= convert_groups_to_dict(
            df_groups=stat_uncrt_abs_groups,
            bins=self.bin_edges,
            hist_name_builder=lambda horn, nu: f"hstat_{horn}_{nu}_abs",
            hist_title=";E_{#nu} [GeV];#sigma^{stat} [m^{-2} POT^{-1}]",
            directory_builder=lambda _: "statistical_uncertainties",
        )

        product_dict |= convert_groups_to_dict(
            df_groups=stat_uncrt_frac_groups,
            bins=self.bin_edges,
            hist_name_builder=lambda horn, nu: f"hstat_{horn}_{nu}",
            hist_title=";E_{#nu} [GeV];#sigma^{stat} / #Phi",
            directory_builder=lambda _: "statistical_uncertainties",
        )

        product_dict |= convert_groups_to_dict(
            df_groups=had_uncrt_groups,
            bins=self.bin_edges,
            hist_name_builder=lambda cat, horn, nu: f"hfrac_hadron_{cat}_{horn}_{nu}",
            hist_title=";E_{#nu} [GeV];Fractional Uncertainty",
            directory_builder=lambda cat: f"fractional_uncertainties/hadron/{cat}",
        )

        product_dict |= convert_groups_to_dict(
            df_groups=flux_weights_groups,
            bins=self.bin_edges,
            hist_name_builder=lambda horn, nu: f"hweights_{horn}_{nu}",
            hist_title=";E_{#nu} [GeV]; #Phi_{#nu}^{PPFX} / #Phi_{#nu}^{nom}",
            directory_builder=lambda _: "ppfx_flux_weights",
        )

        product_dict |= convert_groups_to_dict(
            df_groups=ppfx_corrected_flux_groups,
            bins=self.bin_edges,
            hist_name_builder=lambda cat, horn, nu: f"h{cat}_{horn}_{nu}",
            hist_title=";E_{#nu} [GeV]; #phi_{#nu} [m^{-2} POT^{-1}]",
            directory_builder=lambda cat: f"ppfx_corrected_flux/{cat}",
            has_uncerts=True,
        )

        product_dict |= convert_groups_to_dict(
            df_groups=flux_prediction_groups,
            bins=self.bin_edges,
            hist_name_builder=lambda horn, nu: f"hflux_{horn}_{nu}",
            hist_title=";E_{#nu} [GeV]; #phi_{#nu} [m^{-2} POT^{-1}]",
            directory_builder=lambda _: "flux_prediction",
            has_uncerts=True,
        )

        matrix_objects = [
            (
                self.hadron_systematics.covariance_matrices.loc["fractional"],
                lambda x: f"covariance_matrices/hadron/{x}/hcov_{x}",
            ),
            (
                self.hadron_systematics.covariance_matrices.loc["absolute"],
                lambda x: f"covariance_matrices/hadron/{x}/hcov_{x}_abs",
            ),
            (
                self.hadron_systematics.correlation_matrices,
                lambda x: f"covariance_matrices/hadron/{x}/hcorr_{x}",
            ),
            (self.total_covariance_matrix, "hcov_total_abs"),
            (self.total_correlation_matrix, "hcorr_total"),
        ]

        if self.beam_systematics_is_initialized:
            beam_uncrt_groups = self.beam_systematics.fractional_uncertainties.groupby(
                level=three_levels
            )

            beam_shift_groups = self.beam_systematics.beam_systematic_shifts.loc[
                "fractional"
            ].groupby(level=three_levels)

            product_dict |= convert_groups_to_dict(
                df_groups=beam_uncrt_groups,
                bins=self.bin_edges,
                hist_name_builder=lambda cat, horn, nu: f"hfrac_beam_{cat}_{horn}_{nu}",
                hist_title=";E_{#nu} [GeV];Fractional Uncertainty",
                directory_builder=lambda cat: f"fractional_uncertainties/beam/{cat}",
            )

            product_dict |= convert_groups_to_dict(
                df_groups=beam_shift_groups,
                bins=self.bin_edges,
                hist_name_builder=lambda cat, horn, nu: f"hsyst_beam_{cat}_{horn}_{nu}",
                hist_title=";E_{#nu} [GeV]; #phi_{x} - #phi_{nom} / #phi_{nom}",
                directory_builder=lambda _: "beam_systematic_shifts",
            )

            matrix_objects += [
                (
                    self.beam_systematics.covariance_matrices.loc["fractional"],
                    lambda x: f"covariance_matrices/beam/run_{x}/hcov_{x}",
                ),
                (
                    self.beam_systematics.covariance_matrices.loc["absolute"],
                    lambda x: f"covariance_matrices/beam/run_{x}/hcov_{x}_abs",
                ),
                (
                    self.beam_systematics.correlation_matrices,
                    lambda x: f"covariance_matrices/beam/run_{x}/hcorr_{x}",
                ),
                (
                    self.beam_systematics.total_covariance_matrix.loc["fractional"],
                    "covariance_matrices/beam/hcov_total",
                ),
                (
                    self.beam_systematics.total_covariance_matrix.loc["absolute"],
                    "covariance_matrices/beam/hcov_total_abs",
                ),
                (
                    self.beam_systematics.total_correlation_matrix,
                    "covariance_matrices/beam/hcorr_total",
                ),
            ]

        for mat, title_gen in matrix_objects:
            product_dict |= self._export_matrices(mat, title_gen)

        product_dict["pca/hcov_pca"] = convert_pandas_to_th2(
            self.pca_covariance_matrix,
            hist_name="hcov_pca",
        )
        product_dict["pca/cov_pca"] = convert_symmetric_ndarray_to_tmatrix(
            self.pca_covariance_matrix.values,
        )

        eigenvals = convert_pandas_to_th1(
            series=self.pca_eigen_values["eigenvalue"],
            bin_edges=np.arange(len(self.pca_eigen_values["eigenvalue"]), dtype=float),
            hist_name="heigenvals",
        )

        eigenvals.SetTitle(";Principal Component;#lambda_{n}")

        product_dict["pca/heigenvals"] = eigenvals

        frac_eigenvals = convert_pandas_to_th1(
            series=self.pca_eigen_values["fractional_eigenvalue"],
            bin_edges=np.arange(
                len(self.pca_eigen_values["fractional_eigenvalue"]), dtype=float
            ),
            hist_name="heigenvals_frac",
        )

        frac_eigenvals.SetTitle(
            ";Principal Component;#lambda_{n} / #sum_{k} #lambda_{k}"
        )

        product_dict["pca/heigenvals_frac"] = frac_eigenvals

        cumulative_sum = convert_pandas_to_th1(
            series=self.pca_eigen_values["cumulative_sum"],
            bin_edges=np.arange(
                len(self.pca_eigen_values["cumulative_sum"]), dtype=float
            ),
            hist_name="heigenvals_cumulative_sum",
        )

        cumulative_sum.SetTitle(
            ";Principal Component; CUSUM #left(#lambda_{n} / #sum_{k} #lambda_{k}#right)"
        )

        product_dict["pca/heigenvals_cumulative_sum"] = cumulative_sum

        pca_components = self.pca_components.groupby(
            level=("scale", "horn_polarity", "neutrino_mode")
        )

        for (scale, horn, nu), pcs in pca_components:  # type: ignore
            for npc, pc in pcs.items():
                if scale == "evec":
                    hist_title = f"hevec_{npc}_{horn}_{nu}"
                    axis_titles = ";E_{#nu} [GeV]; #hat{e}_{n}"
                    subdir = "eigenvectors"
                else:
                    hist_title = f"hpc_{npc}_{horn}_{nu}"
                    axis_titles = ";E_{#nu} [GeV]; #sqrt{#lambda_{n}} #hat{e}_{n}"
                    subdir = "principal_components"

                th1 = convert_pandas_to_th1(
                    series=pc,
                    bin_edges=self.bin_edges[nu],
                    hist_name=hist_title,
                )

                th1.SetTitle(axis_titles)
                product_dict[f"pca/{subdir}/{hist_title}"] = th1

        return product_dict
