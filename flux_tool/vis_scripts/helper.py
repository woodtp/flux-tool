from typing import Optional

from hist.hist import Hist
from matplotlib.pyplot import Axes
from numpy import log10, ndarray
from ROOT import TH1D  # type: ignore

from flux_tool.vis_scripts.style import ylabel_flux


def get_hist_scale(h: ndarray | Hist) -> int:
    if isinstance(h, Hist):
        hmax = h.counts().max()
    else:
        hmax = h.max()
    power = log10(hmax)
    return int(round(power))


def create_ylabel_with_scale(scale_factor: int) -> str:
    prefix, units = ylabel_flux.split(" ", 1)
    return prefix + f" $\\mathrm{{\\times 10^{{{scale_factor}}}}}$ " + units


def scale_hist(histogram: ndarray, errors: Optional[ndarray] = None, scale_factor=None):
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
