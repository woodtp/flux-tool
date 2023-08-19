from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import create_ylabel_with_scale, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (icarus_preliminary, neutrino_labels,
                                         place_header, xlabel_enu)


def plot_flux_prediction(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[int, int] = (0,20),
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

        fig, ax = plt.subplots(layout="constrained") #, figsize=(12,12))

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

        # icarus_preliminary(ax, fontsize=24)  # type: ignore
        place_header(ax, f"NuMI Simulation ({horn.upper()})", x_pos=0.58, fontsize=24)  # type: ignore

        ax.set_ylabel(ylabel)  # type: ignore
        ax.set_xlabel(xlabel_enu)  # type: ignore
        ax.legend(loc="best", fontsize=20)  # type: ignore

        ax.set_xlim(*xlim)  # type: ignore

        if output_dir is not None:
            # prefix = f"{horn}_{nu}"
            prefix = f"{horn}_{nu}"
            file_stem = f"{prefix}_flux_prediction"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)
