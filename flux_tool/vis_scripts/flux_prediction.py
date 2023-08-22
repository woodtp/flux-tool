from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import create_ylabel_with_scale, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (icarus_preliminary, neutrino_labels,
                                         place_header, xlabel_enu, ylabel_flux)


def plot_flux_prediction(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    flux_prediction = reader.flux_prediction

    for horn, nu in product(reader.horn_current, ["nue", "numu"]):
        flux = [
            flux_prediction[f"hflux_{horn}_{nu}"].to_pyroot(),
            flux_prediction[f"hflux_{horn}_{nu}bar"].to_pyroot(),
        ]

        max_flux = flux[0].GetMaximum()
        power = -1 * np.round(np.log10(max_flux))
        scale_factor = 10**power

        nominal = [
            reader[
                f"beam_samples/run_15_NOMINAL/hnom_{horn}_{nu}"
            ].to_pyroot(),  # type: ignore
            reader[
                f"beam_samples/run_15_NOMINAL/hnom_{horn}_{nu}bar"
            ].to_pyroot(),  # type: ignore
        ]

        for h in flux:
            h.Scale(scale_factor)

        for h in nominal:
            h.Scale(scale_factor)

        ylabel = create_ylabel_with_scale(int(power))

        prediction_label = (
            # "PPFX Mean "
            r"$\left(\pm"
            r"\mathrm{\sigma}_\mathsf{stat}"
            r"\oplus"
            r"\mathrm{\sigma}_\mathsf{syst}"
            r"\right)$"
        )
        prediction_labels = [
            f"Corrected {neutrino_labels[nu]} Flux {prediction_label}",
            f"Corrected {neutrino_labels[f'{nu}bar']} Flux {prediction_label}",
        ]

        nominal_labels = [
            f"Uncorrected {neutrino_labels[nu]} Flux",
            f"Uncorrected {neutrino_labels[f'{nu}bar']} Flux",
        ]

        fig, ax = plt.subplots(layout="constrained")  # , figsize=(12,12))

        marker = "o" if nu == "numu" else "s"

        color = ["C0", "C1"] if nu == "numu" else ["C2", "C3"]

        hep.histplot(
            H=flux,
            label=prediction_labels,
            ax=ax,
            histtype="errorbar",
            binwnorm=True,
            elinewidth=3,
            capsize=4,
            markersize=14,
            markerfacecolor=[None, "none"],
            color=color,
            marker=marker,
        )

        hep.histplot(
            H=nominal,
            label=nominal_labels,
            binwnorm=True,
            ax=ax,
            yerr=False,
            histtype="step",
            color=["k", "gray"],
            ls=["-", "--"],
            lw=2,
        )

        icarus_preliminary(ax, fontsize=24)  # type: ignore
        place_header(ax, f"NuMI Simulation ({horn.upper()})", (1.0, 1.0), ha="right")  # type: ignore

        ax.set_ylabel(ylabel)  # type: ignore
        ax.set_xlabel(xlabel_enu)  # type: ignore
        ax.legend(loc="best", fontsize=20)  # type: ignore

        ax.set_xlim(*xlim)  # type: ignore

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            file_stem = f"{prefix}_flux_prediction"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)


def plot_flux_uncorrected_logarithmic(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    flux_prediction = reader.nominal_spectra
    pot = reader.pot

    for horn in reader.horn_current:
        flux = [
            flux_prediction[f"{horn}/nom/hnom_numu"].to_pyroot(),
            flux_prediction[f"{horn}/nom/hnom_numubar"].to_pyroot(),
            flux_prediction[f"{horn}/nom/hnom_nue"].to_pyroot(),
            flux_prediction[f"{horn}/nom/hnom_nuebar"].to_pyroot(),
        ]

        for f in flux:
            f.Scale(1 / pot[horn])

        ylabel = [r"$\mathrm{\phi_\nu}$", r"$\mathrm{\phi_{\bar{\nu}}}$"]

        ratio_nus = [
            [neutrino_labels["numu"], neutrino_labels["numubar"]],
            [neutrino_labels["nue"], neutrino_labels["nuebar"]],
        ]
        right_sign_numu, wrong_sign_numu = 0, 1
        right_sign_nue, wrong_sign_nue = 2, 3

        if horn == "rhc":
            right_sign_numu, wrong_sign_numu = wrong_sign_numu, right_sign_numu
            right_sign_nue, wrong_sign_nue = wrong_sign_nue, right_sign_nue
            ylabel.reverse()
            ratio_nus = [
                [neutrino_labels["numubar"], neutrino_labels["numu"]],
                [neutrino_labels["nuebar"], neutrino_labels["nue"]],
            ]

        ratio_labels = map("/".join, ratio_nus)

        sign_contam = [flux[right_sign_numu].Clone(), flux[right_sign_nue].Clone()]

        sign_contam[0].Divide(flux[wrong_sign_numu])
        sign_contam[1].Divide(flux[wrong_sign_nue])

        prediction_labels = [
            neutrino_labels["nue"],
            neutrino_labels["nuebar"],
            neutrino_labels["numu"],
            neutrino_labels["numubar"],
        ]

        facecolor = [None, "none", None, "none"]
        markersize = 18
        markeredgewidth = [None, 2, None, 2]

        fig, axs = plt.subplots(
            2,
            1,
            layout="constrained",
            figsize=(12, 12),
            height_ratios=(3, 1),
            sharex=True,
            gridspec_kw={"hspace": 0.03},
        )

        ax1, ax2 = axs

        hep.histplot(
            ax=ax1,
            H=flux,
            label=prediction_labels,
            histtype="errorbar",
            markerfacecolor=facecolor,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            marker="o",
            binwnorm=True,
            yerr=False,
        )

        hep.histplot(
            ax=ax2,
            H=sign_contam,
            lw=3,
            label=ratio_labels,
            yerr=False,
            edges=False,
        )

        ax1.legend(loc="upper right", ncol=2)
        ax1.set_yscale("log")
        ax1.set_ylabel(ylabel_flux)

        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

        ax2.set_xlabel(xlabel_enu)
        ax2.legend(loc="upper right")
        ax2.axhline(1, ls="--", lw=2, color="k")
        ax2.set_ylabel(" / ".join(ylabel))

        icarus_preliminary(ax1, fontsize=24)
        place_header(ax1, f"NuMI Simulation (Uncorrected, {horn.upper()})", (1.0, 1.0), ha="right")  # type: ignore

        if output_dir is not None:
            file_stem = f"{horn}_flux_prediction_log"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)
