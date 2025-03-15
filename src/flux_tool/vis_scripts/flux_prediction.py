from functools import partial
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import create_ylabel_with_scale, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (
    colorscheme,
    neutrino_labels,
    xlabel_enu,
    ylabel_flux,
)


def plot_flux_prediction(
    reader: SpectraReader,
    nominal_id: str,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    label_drawer: Optional[partial] = None,
    bullets: bool = False,
):
    print("[TODO] This is broken right now. :(")
    return
    flux_prediction = reader.flux_prediction

    for horn, nu in product(reader.horn_current, ["nue", "numu"]):
        hist_title = f"hflux_{horn}_{nu}"
        nu_flux = flux_prediction[hist_title]
        nubar_flux = flux_prediction[f"{hist_title}bar"]

        flux = [nu_flux.to_numpy()[0], nubar_flux.to_numpy()[0]]

        bins = nu_flux.to_numpy()[1]

        max_flux = flux[0].max()
        power = -1 * np.round(np.log10(max_flux))
        scale_factor = 10**power

        nominal = [
            reader[f"beam_samples/run_{nominal_id}/hnom_{horn}_{nu}"].to_numpy()[0],  # type: ignore
            reader[f"beam_samples/run_{nominal_id}/hnom_{horn}_{nu}bar"].to_numpy()[0],  # type: ignore
        ]

        for h in flux:
            h /= scale_factor

        for h in nominal:
            h /= scale_factor

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

        fig, ax = plt.subplots(figsize=(12,12))

        ax.set_box_aspect(1)

        marker = "o" if nu == "numu" else "s"

        color = [colorscheme["blue"], colorscheme["vermillion"]]
        if nu == "nue":
            color = [colorscheme["bluishgreen"], colorscheme["reddishpurple"]]

        if bullets:
            hep.histplot(
                H=flux,
                bins=bins,
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
        else:
            hep.histplot(
                H=flux,
                bins=bins,
                label=prediction_labels,
                ax=ax,
                histtype="step",
                binwnorm=True,
                lw=2,
                color=color,
                yerr=False,
                edges=False,
            )
            fill = ["C0", "C1"] if nu == "numu" else ["C2", "C3"]
            for j, f in enumerate(flux):
                err_up = [0.0] + [
                    (f.GetBinContent(i) + f.GetBinError(i)) / f.GetBinWidth(i)
                    for i in range(1, f.GetNbinsX() + 1)
                ]
                err_low = [0.0] + [
                    (f.GetBinContent(i) - f.GetBinError(i)) / f.GetBinWidth(i)
                    for i in range(1, f.GetNbinsX() + 1)
                ]
                ax.fill_between(
                    bins, err_low, err_up, step="pre", color=fill[j], alpha=0.45
                )

        hep.histplot(
            H=nominal,
            bins=bins,
            label=nominal_labels,
            binwnorm=True,
            ax=ax,
            yerr=False,
            histtype="step",
            color=["k", "gray"],
            ls=["-", "--"],
            lw=2,
            edges=False,
        )

        if label_drawer is not None:
            label_drawer(ax=ax)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel_enu)
        ax.legend(loc="best", fontsize=23)

        ax.set_xlim(*xlim)

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            file_stem = f"{prefix}_flux_prediction"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)

        plt.close(fig)


def plot_flux_uncorrected_logarithmic(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
):
    flux_prediction = reader.nominal_spectra
    pot = reader.pot

    for horn in reader.horn_current:
        flux = [
            flux_prediction[f"{horn}/nom/hnom_numu"].to_numpy(),
            flux_prediction[f"{horn}/nom/hnom_numubar"].to_numpy(),
            flux_prediction[f"{horn}/nom/hnom_nue"].to_numpy(),
            flux_prediction[f"{horn}/nom/hnom_nuebar"].to_numpy(),
        ]

        for f in flux:
            f = f[0] / pot[horn], f[1]

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

        sign_contam = [flux[right_sign_numu], flux[right_sign_nue]]

        sign_contam[0] = (np.divide(flux[right_sign_numu][0], flux[wrong_sign_numu][0], out=np.zeros_like(flux[right_sign_numu][0]), where=flux[wrong_sign_numu][0]!=0), flux[right_sign_numu][1])
        sign_contam[0] = (np.divide(flux[right_sign_nue][0], flux[wrong_sign_nue][0], out=np.zeros_like(flux[right_sign_nue][0]), where=flux[wrong_sign_nue][0]!=0), flux[right_sign_nue][1])

        prediction_labels = [
            neutrino_labels["numu"],
            neutrino_labels["numubar"],
            neutrino_labels["nue"],
            neutrino_labels["nuebar"],
        ]

        # facecolor = [None, "none", None, "none"]
        # markersize = 18
        # markeredgewidth = [None, 2, None, 2]

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

        # ax1.set_box_aspect(1)
        # ax2.set_box_aspect(1)

        hep.histplot(
            ax=ax1,
            H=flux,
            label=prediction_labels,
            linestyle=["-", "--", "-", "--"],
            lw=3,
            # histtype="step",
            # markerfacecolor=facecolor,
            # markersize=markersize,
            # markeredgewidth=markeredgewidth,
            # marker="o",
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
            color=["C0", "C2"],
        )

        ax1.legend(loc="upper right", ncol=2)
        ax1.set_yscale("log")
        ax1.set_ylabel(ylabel_flux, fontsize='x-large')

        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(0, 60)

        ax2.set_xlabel(xlabel_enu, fontsize='x-large')
        ax2.legend(loc="upper right")
        ax2.axhline(1, ls="--", lw=2, color="k")
        ax2.set_ylabel(" / ".join(ylabel), fontsize='x-large')

        hep.label.exp_label(
            exp="NuMI",
            llabel=f"Simulation ({horn.upper()}, Uncorrected)",
            rlabel="",
            ax=ax1,
        )

        if output_dir is not None:
            file_stem = f"{horn}_uncorrected_flux_log"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)

        plt.close(fig)
