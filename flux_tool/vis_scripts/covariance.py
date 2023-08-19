from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
import seaborn as sns

from flux_tool.vis_scripts.helper import save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader


def plot_hadron_correlation_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    horn_currents = reader.horn_current
    matrices = reader.hadron_correlation_matrices

    for key, mat in matrices.items():
        fig, ax = plt.subplots(layout="constrained")

        # print(key)
        heatmap_kwargs = {
            "ax": ax,
            "cmap": "bwr",
            "cbar": True,
            "vmin": -1,
            "vmax": 1,
            "square": False,
            "cbar_kws": {"shrink": 0.81},
        }

        m = mat.to_numpy()[0]

        sns.heatmap(m, **heatmap_kwargs)

        ax.invert_yaxis()
        ax.set_aspect(m.shape[1] / m.shape[0])

        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            which="both",
        )

        if len(horn_currents) == 1:
            ax.text(0.5, -0.125, horn_currents[0].upper(), fontweight="bold", transform=ax.transAxes)
            ax.text(-0.20, 0.5, horn_currents[0].upper(), fontweight="bold", transform=ax.transAxes)
            ax.text(-0.055, 0.083, r"$\mathrm{\nu_e}$", transform=ax.transAxes)
            ax.text(-0.055, 0.25, r"$\mathrm{\bar{\nu}_e}$", transform=ax.transAxes)
            ax.text(-0.055, 0.50, r"$\mathrm{\nu_\mu}$", transform=ax.transAxes)
            ax.text(-0.055, 0.80, r"$\mathrm{\bar{\nu}_\mu}$", transform=ax.transAxes)
            ax.text(0.083, -0.045, r"$\mathrm{\nu_e}$", transform=ax.transAxes)
            ax.text(0.25, -0.045, r"$\mathrm{\bar{\nu}_e}$", transform=ax.transAxes)
            ax.text(0.50, -0.045, r"$\mathrm{\nu_\mu}$", transform=ax.transAxes)
            ax.text(0.80, -0.045, r"$\mathrm{\bar{\nu}_\mu}$", transform=ax.transAxes)
        else:
            ax.text(0.19, -0.125, "FHC", fontweight="bold", transform=ax.transAxes)
            ax.text(0.68, -0.125, "RHC", fontweight="bold", transform=ax.transAxes)
            ax.text(
                -0.125, 0.21, "FHC", fontweight="bold", transform=ax.transAxes, rotation=90
            )
            ax.text(
                -0.125, 0.71, "RHC", fontweight="bold", transform=ax.transAxes, rotation=90
            )
            ax.text(-0.055, 0.05, r"$\mathrm{\nu_e}$", transform=ax.transAxes)
            ax.text(-0.055, 0.17, r"$\mathrm{\bar{\nu}_e}$", transform=ax.transAxes)
            ax.text(-0.055, 0.30, r"$\mathrm{\nu_\mu}$", transform=ax.transAxes)
            ax.text(-0.055, 0.43, r"$\mathrm{\bar{\nu}_\mu}$", transform=ax.transAxes)
            ax.text(-0.055, 0.56, r"$\mathrm{\nu_e}$", transform=ax.transAxes)
            ax.text(-0.055, 0.69, r"$\mathrm{\bar{\nu}_e}$", transform=ax.transAxes)
            ax.text(-0.055, 0.80, r"$\mathrm{\nu_\mu}$", transform=ax.transAxes)
            ax.text(-0.055, 0.92, r"$\mathrm{\bar{\nu}_\mu}$", transform=ax.transAxes)
            ax.text(0.04, -0.045, r"$\mathrm{\nu_e}$", transform=ax.transAxes)
            ax.text(0.17, -0.045, r"$\mathrm{\bar{\nu}_e}$", transform=ax.transAxes)
            ax.text(0.30, -0.045, r"$\mathrm{\nu_\mu}$", transform=ax.transAxes)
            ax.text(0.43, -0.045, r"$\mathrm{\bar{\nu}_\mu}$", transform=ax.transAxes)
            ax.text(0.545, -0.045, r"$\mathrm{\nu_e}$", transform=ax.transAxes)
            ax.text(0.675, -0.045, r"$\mathrm{\bar{\nu}_e}$", transform=ax.transAxes)
            ax.text(0.80, -0.045, r"$\mathrm{\nu_\mu}$", transform=ax.transAxes)
            ax.text(0.92, -0.045, r"$\mathrm{\bar{\nu}_\mu}$", transform=ax.transAxes)

        # for n in range(1, 8):
        #     ax.axhline(n * 15, c="k", lw=1)
        #     ax.axvline(n * 15, c="k", lw=1)

        if output_dir is not None:
            category = key.split("/")[1]
            file_stem = f"{category}_correlation_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
