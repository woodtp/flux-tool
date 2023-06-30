import matplotlib.pyplot as plt
import mplhep

from flux_tool.visualizer.style import neutrino_labels, place_header, ppfx_labels, xlabel_enu, okabe_ito


def plot_fractional_uncertainties(
    uncerts, nu, horn_current, bins, labels=None, ylim=(0.0,0.2), xlim=(0.0,20.0), **kwargs
):

    fig, ax = plt.subplots(constrained_layout=True)

    if labels is None:
        labels = (ppfx_labels[key] for key in uncerts if key != "total")

    hists = [h for key, h in uncerts.items() if key != "total"]

    histplot_opts = {
        "ax": ax,
        "lw": 2,
        "edges": False,
        "yerr": False,
    }
    histplot_opts.update(kwargs)
    n_spectra = len(hists)
    ls = None
    if n_spectra > len(okabe_ito):
        n_solid = n_spectra // 2
        n_dash = n_spectra - n_solid
        ls = n_solid * ["-"] + n_dash * ["--"]

    mplhep.histplot(H=hists, bins=bins, label=labels, ls=ls, **histplot_opts)
    if "total" in uncerts:
        mplhep.histplot(
            H=uncerts["total"], bins=bins, label="Total", color="k", ls="-", **histplot_opts
        )

    ax.set_xlabel(xlabel_enu)
    ax.set_ylabel(r"Fractional Uncertainty ($\mathrm{\sigma}$ / $\mathrm{\phi}$)")

    ax.legend(loc="upper right", ncol=2, fontsize=20, columnspacing=0.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    place_header(ax, f"{horn_current.upper()} {neutrino_labels[nu]}")

    # plt.tight_layout()

    return fig, ax
