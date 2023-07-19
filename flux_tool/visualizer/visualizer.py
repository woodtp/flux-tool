import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import ROOT
import uproot
from matplotlib.pyplot import Figure
from pandas import DataFrame

from flux_tool.flux_systematics_analysis import FluxSystematicsAnalysis
from flux_tool.config import AnalysisConfig
from flux_tool.visualizer.covariance_matrix import plot_covariance
from flux_tool.visualizer.fractional_uncertainties import plot_fractional_uncertainties
from flux_tool.visualizer.ppfx_correction import plot_ppfx_correction, plot_ppfx_correction_inset
from flux_tool.visualizer.ppfx_universes import plot_ppfx_universes
from flux_tool.visualizer.style import style
from flux_tool.visualizer.uncorrected_flux import plot_uncorrected_flux_logscale

np_histogram = tuple[np.ndarray, np.ndarray]


class Visualizer:
    __slots__ = (
        "ana",
        "bin_edges",
        "products_file",
        "plots_path",
        "subdirs",
        "xlim",
    )

    def __init__(
        self, analysis: FluxSystematicsAnalysis, config: AnalysisConfig
    ) -> None:
        plt.style.use(style)

        self.ana = analysis
        self.bin_edges = analysis.bin_edges
        self.products_file = config.products_file
        self.plots_path = config.plots_path
        self.xlim = (0, 6.0)
        subdirs = [
            "flux_spectra",
            "ppfx_universes",
            "matrices",
            "matrices/beam",
            "matrices/hadron",
            "hadron_systematics",
            "hadron_systematics/mesinc_daughter",
            "hadron_systematics/mesinc_projectile",
            "beam_systematics",
        ]
        self.subdirs = {dir: self.plots_path / dir for dir in subdirs}

        for dir in self.subdirs.values():
            dir.mkdir(parents=True, exist_ok=True)

    @property
    def nominal_flux(self) -> DataFrame:
        nom = self.ana.nominal_flux_df
        nom = nom.loc[(nom.run_id == 15) & (nom.category == "nominal")]
        nom = nom.pivot_table(
            columns=["horn_polarity", "neutrino_mode", "bin"],
            values="flux",
        )
        return nom

    @property
    def flux_universes(self) -> DataFrame:
        uni = self.ana.ppfx_correction_df
        uni = uni.loc[uni.category == "total"]
        uni = uni.pivot_table(
            index="universe",
            columns=["horn_polarity", "neutrino_mode", "bin"],
            values="flux",
        )

        return uni

    @property
    def ppfx_correction(self) -> DataFrame:
        return self.ana.hadron_systematics.ppfx_corrected_flux.loc["total"]

    @property
    def uncorrected_flux(self) -> dict[str, dict[str, np_histogram]]:
        def get_hists(dir):
            hists = {
                k.rsplit("_", 1)[-1]: h.to_numpy()
                for k, h in dir.items(filter_name="hnom_*", cycle=False)
            }
            return hists

        def normalize_to_pot(hists, pot: float) -> dict[str, np_histogram]:
            def scale_np_histogram(hist_, pot_):
                counts, bins = hist_
                return counts / pot_, bins

            hists = dict(
                map(
                    lambda items: (items[0], scale_np_histogram(items[1], pot)),
                    hists.items(),
                )
            )
            return hists

        def get_and_scale(dir, pot):
            hists = get_hists(dir)
            return normalize_to_pot(hists, pot)

        nom = {}
        with uproot.open(self.products_file) as f:
            for horn in self.ana.horn_modes:
                dir = f[f"ppfx_output/{horn}/nom"]
                exposure = f[f"ppfx_output/{horn}/hpot"].to_numpy()[0].max()
                nom[horn] = get_and_scale(dir, exposure)

        return nom

    def write_total_uncertainties(self):
        with open(self.plots_path / "total_uncertainties_table.tex", "w") as tex_file:
            table = self.ana.total_flux_uncertainties_table.to_latex(
                float_format="{:.2f}".format
            )
            tex_file.write(table)

    # def save_figure(self, subdir: str, filename: str, fig: Figure) -> tuple[Path, Path]:
    def save_figure(self, subdir: str, filename: str, fig: Figure) -> Path:
        output_dir = self.subdirs[subdir]
        full_path = f"{output_dir}/{filename}"
        # full_path_png = Path(full_path + ".png")
        full_path_pdf = Path(full_path + ".pdf")

        # logging.info(f"Saving plot to {full_path_png}...")
        # plt.savefig(full_path_png, dpi=300)

        logging.info(f"Saving plot to {full_path_pdf}...")
        plt.savefig(full_path_pdf)

        plt.close(fig)

        # return full_path_png, full_path_pdf
        return full_path_pdf

    def make_flux_simulation_plots(self):
        def calculate_flux_ratio(hist1, hist2):
            counts1, bins = hist1
            counts2, _ = hist2
            ratio = np.divide(
                counts1, counts2, out=np.zeros_like(counts1), where=counts2 != 0
            )
            return ratio, bins

        files: list[Path] = []
        for horn, flux in self.uncorrected_flux.items():
            is_fhc = horn == "fhc"

            numu_flux = [flux["numu"], flux["numubar"]]
            nue_flux = [flux["nue"], flux["nuebar"]]

            if is_fhc:
                numu_ratio = calculate_flux_ratio(*numu_flux)
                nue_ratio = calculate_flux_ratio(*nue_flux)
            else:
                numu_ratio = calculate_flux_ratio(numu_flux[1], numu_flux[0])
                nue_ratio = calculate_flux_ratio(nue_flux[1], nue_flux[0])

            fig, _ = plot_uncorrected_flux_logscale(
                numu_flux=numu_flux,
                nue_flux=nue_flux,
                numu_ratio=numu_ratio,
                nue_ratio=nue_ratio,
                fhc=is_fhc,
            )

            filename = f"{horn}_numi_flux_simulation_log"
            full_path_pdf = self.save_figure("flux_spectra", filename, fig)

            files.append(full_path_pdf)

        return files

    def make_ppfx_universe_plots(self):
        files: list[Path] = []

        groups = self.flux_universes.groupby(
            level=("horn_polarity", "neutrino_mode"), axis=1
        )

        for (horn, nu), unis in groups:
            nominal = self.nominal_flux.loc[:, (horn, nu)]
            correction = self.ppfx_correction.loc[horn, nu]
            fig, ax = plot_ppfx_universes(
                universes=unis,
                nominal_flux=nominal,
                ppfx_correction=correction,
                bins=self.bin_edges,
                nu=nu,
            )
            filename = f"{horn}_{nu}_universes"
            full_path_pdf = self.save_figure("ppfx_universes", filename, fig)
            files.append(full_path_pdf)

        return files

    def make_flux_correction_plots(self):
        def convert_to_numpy_list(df):
            nus = []
            nominal = []
            correction = []
            uncert = []

            for nu, h in df.groupby("neutrino_mode"):
                nus.append(nu)
                nominal.append(h["nominal"].to_numpy())
                correction.append(h["mean"].to_numpy())
                uncert.append(h["sigma"].to_numpy())

            return nominal, correction, uncert, nus

        nom = self.nominal_flux.T
        nom.columns = ["nominal"]

        df = pd.concat([nom, self.ppfx_correction], axis=1)

        for horn_polarity in self.ana.horn_modes:
            nue_nuebar = df.loc[horn_polarity, "nue":"nuebar", :]
            numu_numubar = df.loc[horn_polarity, "numu":"numubar", :]

            nominal1, correction1, uncert1, nus1 = convert_to_numpy_list(nue_nuebar)
            nominal2, correction2, uncert2, nus2 = convert_to_numpy_list(numu_numubar)

            is_fhc = horn_polarity == "fhc"

            files: list[Path] = []

            fig1, _ = plot_ppfx_correction(
                nominal1,
                correction1,
                uncert1,
                self.bin_edges,
                ["nue", "nuebar"],
                is_fhc=is_fhc,
                xlim=self.xlim,
            )

            full_path_pdf1 = self.save_figure(
                "flux_spectra", f"{horn_polarity}_nue_flux_prediction", fig1
            )

            files.append(full_path_pdf1)

            fig2, _ = plot_ppfx_correction(
                nominal2,
                correction2,
                uncert2,
                self.bin_edges,
                ["numu", "numubar"],
                is_fhc=is_fhc,
                xlim=self.xlim,
            )

            full_path_pdf2 = self.save_figure(
                "flux_spectra", f"{horn_polarity}_numu_flux_prediction", fig2
            )

            files.append(full_path_pdf2)

            fig3, _ = plot_ppfx_correction_inset(
                nominal1,
                nominal2,
                correction1,
                correction2,
                uncert1,
                uncert2,
                self.bin_edges,
                is_fhc=is_fhc,
            )

            full_path_pdf3 = self.save_figure(
                "flux_spectra", f"{horn_polarity}_inset_flux_prediction", fig3
            )

            files.append(full_path_pdf3)

        return files

    def make_correlation_matrix_plots(self, systematics, subdir: str):
        mats = systematics.correlation_matrices

        files = []

        if hasattr(systematics, "total_correlation_matrix"):
            corr_total = systematics.total_correlation_matrix  # .to_numpy()

            fig1, _ = plot_covariance(corr_total)

            full_path_total_pdf = self.save_figure(
                f"matrices/{subdir}", "total_correlation_matrix", fig1
            )

            files.append(full_path_total_pdf)

        for category, mat in mats.groupby("category"):
            fig, _ = plot_covariance(mat)  # .to_numpy())

            filename = f"{category}_correlation_matrix"

            full_path_pdf = self.save_figure(f"matrices/{subdir}", filename, fig)

            files.append(full_path_pdf)

        return files

    def make_total_correlation_matrix_plot(self):
        mat = self.ana.total_correlation_matrix  # .to_numpy()

        fig, _ = plot_covariance(mat)

        filename = "total_correlation_matrix"
        full_path_pdf = self.save_figure("matrices", filename, fig)

        return full_path_pdf

    def make_uncertainty_plots(self):
        groups = self.ana.hadron_systematics.fractional_uncertainties.groupby(
            level=["horn_polarity", "neutrino_mode"]
        )
        files = []
        for (horn, nu), uncerts in groups:
            frac_uncerts = {}
            projectile_uncerts = {}
            daughter_uncerts = {}
            for name, h in uncerts.items():
                if "projectile" in name:
                    projectile_uncerts[name] = h
                    continue
                if "daughter" in name:
                    daughter_uncerts[name] = h
                    continue
                if name == "mesinc":
                    projectile_uncerts[name] = h
                    daughter_uncerts[name] = h
                frac_uncerts[name] = h

            fig1, _ = plot_fractional_uncertainties(
                frac_uncerts, nu, horn, self.bin_edges, xlim=self.xlim
            )
            filename1 = f"{horn}_{nu}_fractional_uncertainties"
            full_path_pdf1 = self.save_figure("hadron_systematics", filename1, fig1)
            files.append(full_path_pdf1)

            fig2, _ = plot_fractional_uncertainties(
                projectile_uncerts, nu, horn, self.bin_edges, xlim=self.xlim
            )
            filename2 = f"{horn}_{nu}_mesinc_projectile_uncertainties"
            full_path_pdf2 = self.save_figure(
                "hadron_systematics/mesinc_projectile", filename2, fig2
            )
            files.append(full_path_pdf2)

            fig3, _ = plot_fractional_uncertainties(
                daughter_uncerts, nu, horn, self.bin_edges, xlim=self.xlim
            )
            filename3 = f"{horn}_{nu}_mesinc_daughter_uncertainties"
            full_path_pdf3 = self.save_figure(
                "hadron_systematics/mesinc_daughter", filename3, fig3
            )
            files.append(full_path_pdf3)

        return files

    def plot_files(self) -> None:
        # self.write_total_uncertainties()
        self.make_flux_simulation_plots()
        self.make_ppfx_universe_plots()
        self.make_flux_correction_plots()
        self.make_uncertainty_plots()
        self.make_correlation_matrix_plots(self.ana.hadron_systematics, "hadron")
        if self.ana.beam_systematics_is_initialized:
            self.make_correlation_matrix_plots(self.ana.beam_systematics, "beam")
        self.make_total_correlation_matrix_plot()

        # plot_set = (
        #     self.make_flux_simulation_plots(),
        #     self.make_ppfx_universe_plots(),
        #     self.make_correlation_matrix_plots(self.ana.hadron_systematics, "hadron"),
        #     self.make_correlation_matrix_plots(self.ana.beam_systematics, "beam"),
        #     [self.make_total_correlation_matrix_plot()],
        # )
        #
        # for plots in plot_set:
        #     yield from plots

    # def save_png_to_rootfile(self, plot: Path, tdirectory):
    #     img = ROOT.TImage.Open(str(plot))
    #     name = plot.stem
    #     subdir = plot.parent.stem
    #
    #     if not tdirectory.Get(subdir):
    #         tdirectory.mkdir(subdir)
    #     d = tdirectory.Get(subdir)
    #
    #     d.WriteObject(img, name)

    def save_plots(self) -> None:
        self.plot_files()
        # with open_tfile(self.products_file, "update") as f:
        #     f.mkdir("plots")
        #
        #     tdirectory = f.Get("plots")
        #
        #     for plot in self.plot_files():
        #         self.save_png_to_rootfile(plot, tdirectory)
