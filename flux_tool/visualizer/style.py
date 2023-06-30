import matplotlib.pyplot as plt
import mplhep
from cycler import Cycler, cycler
from matplotlib.axes import Axes

style = {
    "axes.formatter.limits": [-3, 3],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 40,
    "axes.linewidth": 1.0,
    "axes.prop_cycle": cycler(
        "color",
        [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#56B4E9",
            "#E69F00",
            "#000000",
            "#F0E442",
        ],
    ),
    "axes.unicode_minus": False,
    "axes.titlesize": 40,
    "figure.figsize": [10.0, 10.0],
    "font.sans-serif": "TeX Gyre Heros",
    "font.family": "sans-serif",
    "font.size": 26,
    "grid.alpha": 0.8,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "image.cmap": "cividis",
    "lines.linewidth": 2.0,
    "legend.fontsize": 32,
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
    "savefig.transparent": False,
    "xaxis.labellocation": "center",
    "yaxis.labellocation": "center",
    "xtick.labelsize": "small",
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.labelsize": "small",
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
}

okabe_ito = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluishgreen": "#009E73",
    "reddishpurple": "#CC79A7",
    "skyblue": "#56B4E9",
    "orange": "#E69F00",
    "black": "#000000",
    "yellow": "#F0E442",
}

neutrino_labels = {
    "nue": r"$\mathrm{\nu_{e}}$",
    "nuebar": r"$\mathrm{\bar{\nu}_{e}}$",
    "numu": r"$\mathrm{\nu_{\mu}}$",
    "numubar": r"$\mathrm{\bar{\nu}_{\mu}}$",
}

neutrino_parent_labels = {
    "kp": r"$\mathrm{K^{+}}$",
    "km": r"$\mathrm{K^{-}}$",
    "kpm": r"$\mathrm{K^{\pm}}$",
    "k0l": r"$\mathrm{K^{0}_{L}}$",
    "mup": r"$\mathrm{\mu^{+}}$",
    "mum": r"$\mathrm{\mu^{-}}$",
    "mu": r"$\mathrm{\mu^{\pm}}$",
    "pipm": r"$\mathrm{\pi^{\pm}}$",
    "pip": r"$\mathrm{\pi^{+}}$",
    "pim": r"$\mathrm{\pi^{-}}$",
}

ppfx_labels = {
    "pCpi": r"$p + C \to \mathrm{\pi}^\pm + X$",
    "pCnu": r"$p + C \to N + X$",
    "pCk": r"$p + C \to K + X$",
    "pCQEL": r"$p + C$ (QEL)",
    "nCpi": r"$n + C \to \mathrm{\pi}^\pm + X$",
    "mesinc": r"$(\mathrm{\pi}^\pm, K) + A \to (\mathrm{\pi}^\pm, K, N) + X$",
    "mesinc_projectile_K0": r"$K^0 + A \to (\mathrm{\pi}^\pm, K, N) + X$",
    "mesinc_projectile_Km": r"$K^{-} + A \to (\mathrm{\pi}^\pm, K, N) + X$",
    "mesinc_projectile_Kp": r"$K^{+} + A \to (\mathrm{\pi}^\pm, K, N) + X$",
    "mesinc_projectile_pip": r"$\pi^{+} + A \to (\mathrm{\pi}^\pm, K, N) + X$",
    "mesinc_projectile_pim": r"$\pi^{-} + A \to (\mathrm{\pi}^\pm, K, N) + X$",
    "mesinc_daughter_K0": r"$(\mathrm{\pi}^\pm, K) + A \to K^0 + X$",
    "mesinc_daughter_Km": r"$(\mathrm{\pi}^\pm, K) + A \to K^{-} + X$",
    "mesinc_daughter_Kp": r"$(\mathrm{\pi}^\pm, K) + A \to K^{+} + X$",
    "mesinc_daughter_pip": r"$(\mathrm{\pi}^\pm, K) + A \to \pi^{+} + X$",
    "mesinc_daughter_pim": r"$(\mathrm{\pi}^\pm, K) + A \to \pi^{-} + X$",
    "nua": r"$N + A \to X$",
    "nuAlFe": r"$N + (Al, Fe)\to X$",
    "att": "Attenuation",
    "others": "Others",
    "total": "Total",
}


xlabel_enu = r"E$_\mathrm{\nu}$ [GeV]"
ylabel_flux = r"$\mathrm{\phi_\nu}$ [m$^{-2}$ GeV$^{-1}$ POT$^{-1}$]"


def get_prop_cycler() -> Cycler:
    """Returns an axis prop cycler, which cycles first through solid lines with the Okabe Ito colorscheme, then dashed lines.
    Excludes the black color.
    """
    prop_cycle = cycler(linestyle=["-", "--", "-."]) * cycler(
        color=[v for k, v in okabe_ito.items() if k != "black"]
    )

    return prop_cycle


def place_header(ax: Axes, header: str) -> None:
    ax.text(
        0.0,
        1.015,
        header,
        fontweight="bold",
        fontstyle="italic",
        fontsize=28,
        transform=ax.transAxes,
    )


def apply_style() -> None:
    plt.style.use(mplhep.style.ROOT)

    default_cycler = cycler(linestyle=["-", "--", "-."]) * cycler(
        color=list(okabe_ito.values())
    )

    mpl_opts = {
        "axes.prop_cycle": default_cycler,
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": [-3, 3],
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        "font.family": "TeX Gyre Heros",
        "font.size": 32,
        "mathtext.fontset": "stixsans",
    }

    plt.rcParams.update(mpl_opts)
