from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import create_ylabel_with_scale, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (neutrino_labels,
                                         neutrino_parent_labels, place_header,
                                         xlabel_enu)


def plot_parents(
    reader: SpectraReader,
    output_dir: Path,
    xlim: tuple[float, float] = (0, 20),
    stacked: bool = False,
) -> None:
    parents = {
        "numu": ["pip", "kp", "k0l", "mum"],
        "numubar": ["pim", "km", "k0l", "mup"],
        "nue": ["kp", "k0l", "mup"],
        "nuebar": ["km", "k0l", "mum"],
    }

    pot = reader.pot

    ppfx_correction = reader.ppfx_correction
    parent_spectra = reader.parent_spectra
    flux_weights = reader.flux_weights

    ylim_high = {
        "fhc": {"nue": 1.5, "nuebar": 1.5, "numu": 0.8, "numubar": 0.8},
        "rhc": {"nue": 1.5, "nuebar": 1.5, "numu": 0.8, "numubar": 0.8},
    }

    for horn, nu in reader.horns_and_nus:
        _, bins = ppfx_correction[f"htotal_{horn}_{nu}"].to_numpy()

        nu_correction = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()

        max_flux = nu_correction.GetMaximum()

        power = -1 * np.round(np.log10(max_flux))

        scale_factor = 10**power

        nu_correction.Scale(scale_factor)

        weights = flux_weights[f"hweights_{horn}_{nu}"].to_pyroot()

        numu_parents = [
            parent_spectra[f"{horn}/nom/parent/hnom_{nu}_pipm"].to_pyroot(),
            parent_spectra[f"{horn}/nom/parent/hnom_{nu}_kpm"].to_pyroot(),
            parent_spectra[f"{horn}/nom/parent/hnom_{nu}_k0l"].to_pyroot(),
            parent_spectra[f"{horn}/nom/parent/hnom_{nu}_mu"].to_pyroot(),
        ]

        if "nue" in nu:
            numu_parents = numu_parents[1:]

        H = []

        for parent in numu_parents:
            parent = parent.Rebin(len(bins) - 1, "", bins)
            parent.Scale(scale_factor / pot[horn])
            parent.Multiply(weights)
            H.append(parent)

        labels = [
            f"{neutrino_labels[nu]} from {neutrino_parent_labels[x]} decays"
            for x in parents[nu]
        ]

        fig, ax1 = plt.subplots()

        ax1.set_box_aspect(1)

        ax_ypos = 0.20 if "nue" in nu else 0.10

        axins = ax1.inset_axes((0.50, ax_ypos, 0.40, 0.40))  # type: ignore

        axins.set_box_aspect(1)

        axs = [ax1, axins]

        opts = dict(histtype="step", yerr=False, binwnorm=True, edges=False)

        correction_label = f"PPFX corrected {neutrino_labels[nu]} flux"

        ax1.set_xlim(xlim)  # type: ignore

        ax1.set_ylabel(create_ylabel_with_scale(int(power)))  # type: ignore
        ax1.set_xlabel(xlabel_enu)  # type: ignore

        # place_header(ax1, f"NuMI Simulation ({horn.upper()})")  # type: ignore

        hep.label.exp_label(exp="NuMI", llabel=f"Simulation ({horn.upper()})", rlabel="")

        for ax in axs:
            hep.histplot(
                H=nu_correction,
                label=correction_label,
                lw=3,
                color="k",
                ax=ax,
                zorder=10,
                **opts,  # type: ignore
            )

        file_suffix = ""
        if stacked:
            file_suffix = "_stacked"
            for ax in axs:
                hep.histplot(
                    H=H,
                    label=labels,
                    stack=True,
                    histtype="fill",
                    edgecolor="k",
                    lw=0.25,
                    ax=ax,
                    binwnorm=True,
                    edges=False,
                )
            handles, labs = plt.gca().get_legend_handles_labels()
            order = [0, 4, 3, 2, 1] if len(handles) == 5 else [0, 3, 2, 1]
            ax1.legend(  # type: ignore
                [handles[i] for i in order], [labs[i] for i in order], loc="best"
            )
        else:
            for ax in axs:
                hep.histplot(H=H, label=labels, lw=2, ax=ax, **opts)  # type: ignore
            ax1.legend(loc="best")  # type: ignore

        xlim2 = (2, 6) if "nue" in nu else (1, 4)
        axins.set_xlim(xlim2)

        axins.set_ylim(0, ylim_high[horn][nu])

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            file_stem = f"{prefix}_parent_composition{file_suffix}"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)
