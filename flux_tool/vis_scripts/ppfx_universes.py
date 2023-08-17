from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import (_load_spectra,
                                          create_ylabel_with_scale,
                                          save_figure)
from flux_tool.vis_scripts.style import (neutrino_labels, place_header,
                                         xlabel_enu)


def plot_ppfx_universes(products_file: Path | str, output_dir: Optional[Path] = None):
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)

    spectra = _load_spectra(products_file)

    nominal_flux = spectra["nominal_spectra"]
    universes = spectra["universes"]
    ppfx_correction = spectra["ppfx_correction"]

    pot = {"fhc": spectra["fhc_pot"], "rhc": spectra["rhc_pot"]}
    horns = ["fhc", "rhc"]
    nus = ["nue", "nuebar", "numu", "numubar"]

    for horn, nu in product(horns, nus):
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

        ax.plot([], [], " ", label="PPFX Input")

        nu_label = neutrino_labels[nu]

        hep.histplot(
            H=nom_scaled,
            label=f"{nu_label} nominal",
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
            label=f"{nu_label} universes",
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
            label=f"{neutrino_labels[nu]} mean",
            color="C0",
            histtype="errorbar",
            xerr=True,
            capsize=4,
            elinewidth=4,
            marker=None,
            binwnorm=1,
            ax=ax,
        )

        ax.set_xlim(0, 6)
        ax.set_xlabel(xlabel_enu)
        ax.set_ylabel(create_ylabel_with_scale(int(power)))

        handles, labels = ax.get_legend_handles_labels()

        leg = ax.legend(
            handles,
            labels,
            loc="best",
        )

        place_header(ax, f"NuMI Simulation ({horn.upper()})")

        ax.text(
            0.1,
            0.925,
            "ICARUS Preliminary",
            fontweight="bold",
            fontsize=20,
            transform=ax.transAxes,
        )

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
