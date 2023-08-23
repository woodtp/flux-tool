import logging
from itertools import product
from pathlib import Path
from typing import Generator, Optional

from hist.hist import Hist
from matplotlib.pyplot import Axes, Figure
from numpy import log10
from numpy.typing import NDArray
from ROOT import TH1D  # type: ignore

from flux_tool.vis_scripts.style import ylabel_flux


def get_all_neutrinos() -> Generator[tuple[str, str], None, None]:
    yield from product(["fhc", "rhc"], ["nue", "nuebar", "numu", "numubar"])


def save_figure(
    fig: Figure, fig_name: str, output_dir: Path | str, tex_caption: str, tex_label: str
) -> None:
    for ext in get_plot_extensions():
        file_name = f"{output_dir}/{fig_name}.{ext}"
        logging.debug(f"Saving image {file_name}...")
        fig.savefig(file_name)

    tex_figure = build_latex_figure(f"{fig_name}.pdf", tex_caption, tex_label)

    tex_filename = f"{output_dir}/{fig_name}.tex"

    logging.debug(f"Writing figure to {tex_filename}...")

    with open(tex_filename, "w") as texfile:
        texfile.write(tex_figure)


def get_plot_extensions() -> Generator[str, None, None]:
    yield from ("png", "pdf")


def build_latex_figure(image_path: str, caption: str, label: str) -> str:
    lines: list[str] = [
        r"\begin{figure}",
        r"    \centering",
        f"    \\includegraphics[width=\\textwidth]{{{image_path}}}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{fig:{label}}}",
        r"\end{figure}",
    ]
    fig: str = "\n".join(lines)

    return fig


def get_hist_scale(h: NDArray | Hist) -> int:
    if isinstance(h, Hist):
        hmax = h.counts().max()
    else:
        hmax = h.max()
    power = log10(hmax)
    return int(round(power))


def create_ylabel_with_scale(scale_factor: int) -> str:
    prefix, units = ylabel_flux.split(" ", 1)
    return prefix + f" $\\mathrm{{\\times 10^{{{scale_factor}}}}}$ " + units


def scale_hist(histogram: NDArray, errors: Optional[NDArray] = None, scale_factor=None):
    if scale_factor is None:
        scale_factor = -1 * get_hist_scale(histogram)
    scaled_histogram = (10**scale_factor) * histogram
    # scaled_ylabel = insert_yscale(scale_factor)

    if errors is not None:
        scaled_errors = (10**scale_factor) * errors
        return scaled_histogram, scaled_errors
    return scaled_histogram


def make_legend_no_errorbars(ax: Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, **kwargs)


def absolute_uncertainty(total_flux: TH1D, fractional_uncertainty: TH1D) -> TH1D:
    return total_flux * fractional_uncertainty
