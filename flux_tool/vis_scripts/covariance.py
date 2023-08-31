from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from flux_tool.vis_scripts.helper import save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import neutrino_labels


def plot_matrices(matrices: dict[str, Any], horn_currents: list[str]):
    for key, mat in matrices.items():
        fig, ax = plt.subplots(layout="constrained")

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

        ax.invert_yaxis()  # type: ignore
        ax.set_aspect(m.shape[1] / m.shape[0])  # type: ignore

        ax.tick_params(  # type: ignore
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            which="both",
        )

        nue = neutrino_labels["nue"]
        nueb = neutrino_labels["nuebar"]
        numu = neutrino_labels["numu"]
        numub = neutrino_labels["numubar"]

        kwargs = {
            "transform": ax.transAxes,  # type: ignore
            "fontsize": 32,
            "fontweight": "bold",
            # "ha": "center",
            # "va": "top",
        }
        kwargs2 = {
            "transform": ax.transAxes,  # type: ignore
            "fontsize": 32,
            "fontweight": "bold",
            # "va": "center",
            # "ha": "right",
        }

        # ax.annotate(  # type: ignore
        #     "ICARUS Preliminary",
        #     (0.0, 1.0),
        #     xytext=(0, 3),
        #     xycoords="axes fraction",
        #     textcoords="offset points",
        #     ha="left",
        #     va="bottom",
        #     fontweight="bold",
        # )  # type: ignore

        if len(horn_currents) == 1:
            xpos = -0.015
            nu_coords = [[xpos, 0.09], [xpos, 0.25], [xpos, 0.50], [xpos, 0.85]]
            horn = horn_currents[0].upper()
            ax.text(0.50, -0.085, horn, ha="center", va="top", **kwargs)  # type: ignore
            ax.text(-0.085, 0.50, horn, rotation=90, ha="right", va="center", **kwargs2)  # type: ignore

            for (x, y), nu in zip(nu_coords, neutrino_labels.values()):
                ax.text(x, y, nu, ha="right", va="center", **kwargs)  # type: ignore
                ax.text(y, x, nu, ha="center", va="top", **kwargs)  # type: ignore
        else:
            ax.text(0.245, -0.085, "FHC", ha="center", va="top", **kwargs)  # type: ignore
            ax.text(0.765, -0.085, "RHC", ha="center", va="top", **kwargs)  # type: ignore
            ax.text(  # type: ignore
                -0.085, 0.245, "FHC", rotation=90, ha="right", va="center", **kwargs2  # type: ignore
            )
            ax.text(  # type: ignore
                -0.085, 0.765, "RHC", rotation=90, ha="right", va="center", **kwargs2  # type: ignore
            )

            xpos = -0.015
            nu_coords = [
                [xpos, 0.05, nue],
                [xpos, 0.18, nueb],
                [xpos, 0.31, numu],
                [xpos, 0.44, numub],
                [xpos, 0.57, nue],
                [xpos, 0.70, nueb],
                [xpos, 0.83, numu],
                [xpos, 0.96, numub],
            ]

            for x, y, nu in nu_coords:
                ax.text(x, y, nu, ha="right", va="center", **kwargs2)  # type: ignore
                ax.text(y, x, nu, ha="center", va="top", **kwargs)  # type: ignore

        yield key, fig


def plot_hadron_correlation_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    horn_currents = reader.horn_current
    matrices = reader.hadron_correlation_matrices

    figures = plot_matrices(matrices, horn_currents)

    if output_dir is not None:
        for key, fig in figures:
            category = key.split("/")[1]
            file_stem = f"{category}_correlation_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
            plt.close(fig)


def plot_beam_correlation_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    horn_currents = reader.horn_current
    matrices = reader.beam_correlation_matrices

    figures = plot_matrices(matrices, horn_currents)

    if output_dir is not None:
        for key, fig in figures:
            category = key
            if "/" in category:
                category = category.split("/")[1]
            category = category.split("_", 1)[1]
            file_stem = f"{category}_correlation_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
            plt.close(fig)
