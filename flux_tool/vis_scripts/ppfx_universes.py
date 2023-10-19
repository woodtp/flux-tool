from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import create_ylabel_with_scale, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (neutrino_labels, place_header,
                                         xlabel_enu)


def plot_ppfx_universes(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0.0, 20.0),
):
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)

    nominal_flux = reader.nominal_spectra
    universes = reader.universes
    ppfx_correction = reader.ppfx_correction

    pot = reader.pot

    for horn, nu in reader.horns_and_nus:
        unis = [universes[f"{horn}/{nu}_total/htotal_{nu}_{x}"] for x in range(100)]
        nom = nominal_flux[f"{horn}/nom/hnom_{nu}"]
        correction = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()

        _, bins = ppfx_correction[f"htotal_{horn}_{nu}"].to_numpy()

        max_flux = correction.GetMaximum()

        power = -1 * np.round(np.log10(max_flux))

        scale_factor = 10**power

        correction.Scale(scale_factor)

        def _scale(h):
            h = h.to_pyroot()
            h = h.Rebin(len(bins) - 1, "", bins)
            h.Scale(scale_factor / pot[horn])
            return h

        H = list(map(_scale, unis))

        nom_scaled = _scale(nom)

        fig, ax = plt.subplots()

        ax.set_box_aspect(1)

        ax.plot([], [], " ", label="PPFX Input")

        # nu_label = neutrino_labels[nu]

        hep.histplot(
            H=nom_scaled,
            label="Uncorrected Flux",
            # bins=bins,
            color="k",
            lw=3,
            yerr=False,
            edges=False,
            binwnorm=1,
            ls="--",
            zorder=10,
            ax=ax,
        )

        ax.plot([], [], " ", label="PPFX Output")

        hep.histplot(
            H=H[0],
            label="Flux Universes",
            color="C4",
            lw=1,
            yerr=False,
            edges=False,
            binwnorm=1,
            ax=ax,
        )
        hep.histplot(
            H=H[1:],
            color="C4",
            lw=1,
            yerr=False,
            edges=False,
            binwnorm=1,
            ax=ax,
        )

        hep.histplot(
            H=correction,
            label=r"Mean Flux ($\pm \mathrm{\sigma}$)",
            color="C0",
            histtype="errorbar",
            xerr=False,
            capsize=4,
            elinewidth=4,
            marker="o",
            binwnorm=1,
            ax=ax,
            zorder=15,
        )

        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel_enu)
        ax.set_ylabel(create_ylabel_with_scale(int(power)))

        handles, labels = ax.get_legend_handles_labels()

        leg = ax.legend(
            handles,
            labels,
            loc="best",
        )

        # place_header(ax, f"NuMI Simulation ({horn.upper()} {neutrino_labels[nu]})")
        hep.label.exp_label(
            exp="NuMI",
            llabel=f"Simulation ({horn.upper()} {neutrino_labels[nu]})",
            rlabel="",
        )

        # place_header(ax, "ICARUS Preliminary", (1.0, 1.0), ha="right")

        for item, label in zip(leg.legend_handles, leg.texts):
            if label._text in ["PPFX Input", "PPFX Output"]:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha("left")
                label.set_position((-2 * width, 0))

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            file_stem = f"{prefix}_ppfx_universes"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
