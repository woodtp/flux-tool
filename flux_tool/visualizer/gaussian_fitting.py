import mplhep
import numpy as np
import matplotlib.pyplot as plt

from flux_tool.hadron_production_systematics import FluxUniverseFit
from flux_tool.visualizer.style import neutrino_labels


def plot_gaussian_fitting(ax, fit_result: FluxUniverseFit, nu: str, e_range: str) -> None:
    hist = fit_result.flux_histogram

    xmin = fit_result.universe_mean - 4 * fit_result.universe_sigma
    xmax = fit_result.universe_mean + 4 * fit_result.universe_sigma

    X = np.linspace(0.9 * xmin, 1.1 * xmax, num=200)
    y = [fit_result.eval_fit_function(x) for x in X]

    mplhep.histplot(H=hist, yerr=False, lw=3, color="k", label=r"PPFX universes", ax=ax)

    ax.plot(X, y, lw=4, color="C1", label="Gaussian fit")

    ax.set_ylim(0, 24)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("Universe Count", loc="center")
    ax.set_xlabel(r"$\mathrm{\phi_\nu}$ (m$^{-2}$ POT$^{-1}$)", loc="center")

    ax.legend(loc="upper right", fontsize=24)

    ax.text(0.57, 1.015, e_range, transform=ax.transAxes)

    ax.text(
        0,
        1.015,
        neutrino_labels[nu],
        fontstyle="italic",
        fontweight="bold",
        transform=ax.transAxes,
    )

    res_text = (
        r"$\frac{\mathrm{\langle \phi \rangle}_{fit} - \mathrm{\langle \phi \rangle}_{ppfx}}{\mathrm{\langle \phi \rangle}_{ppfx}} =$"
        + f"{fit_result.mean_fractional_error:0.1%}"
        "\n"
        r"$\mathrm{\chi^2}/ndf$ = " + f"{fit_result.chi2ndf:0.2f}"
    )

    ax.text(0.57, 0.64, res_text, transform=ax.transAxes)

    plt.tight_layout()
