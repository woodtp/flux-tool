from itertools import cycle

import matplotlib.pyplot as plt
import mplhep
from cycler import Cycler, cycler
from matplotlib.axes import Axes

# Okabe-Ito Colors
colorscheme = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluishgreen": "#009E73",
    "reddishpurple": "#CC79A7",
    "skyblue": "#56B4E9",
    "orange": "#E69F00",
    "yellow": "#F0E442",
}

style = {
    "axes.formatter.limits": [-3, 3],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 40,
    "axes.linewidth": 1.0,
    "axes.prop_cycle": cycler(linestyle=["-", "--", "-."])
    * cycler(
        color=[
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#56B4E9",
            "#E69F00",
            "#F0E442",
            # "#000000",
        ]
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
    "savefig.transparent": True,
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
    "pCpi": r"$\mathrm{p + C \to \pi^\pm + X}$",
    "pCnu": r"$\mathrm{p + C \to N + X}$",
    "pCk": r"$\mathrm{p + C \to K + X}$",
    "pCQEL": r"$\mathrm{p + C}$ (QEL)",
    "nCpi": r"$\mathrm{n + C \to \pi^\pm + X}$",
    "mesinc": r"$\mathrm{\left( \pi^\pm, K \right) + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_K0": r"$\mathrm{K^0 + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_Km": r"$\mathrm{K^{-} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_Kp": r"$\mathrm{K^{+} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_pip": r"$\mathrm{\pi^{+} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_pim": r"$\mathrm{\pi^{-} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_daughter_K0": r"$\mathrm{\left( \pi^\pm, K \right) + A \to K^0 + X}$",
    "mesinc_daughter_Km": r"$\mathrm{\left( \pi^\pm, K \right) + A \to K^{-} + X}$",
    "mesinc_daughter_Kp": r"$\mathrm{\left( \pi^\pm, K \right) + A \to K^{+} + X}$",
    "mesinc_daughter_pip": r"$\mathrm{\left( \pi^\pm, K \right) + A \to \pi^{+} + X}$",
    "mesinc_daughter_pim": r"$\mathrm{\left( \pi^\pm, K \right) + A \to \pi^{-} + X}$",
    "nua": r"$\mathrm{N + A \to X}$",
    "nua_datavol": r"$\mathrm{N + A \to X}$ (Carbon Volumes, $\mathrm{-0.25 \leq x_F < 0}$)",
    "nua_datavol_negxF": r"$\mathrm{N + A \to X}$ (Carbon Volumes, OOPS)",
    "nua_othervol": r"$\mathrm{N + A \to X}$ (Non-C Volumes, !OOPS)",
    "nua_other": r"$\mathrm{N + A \to X}$ (Everything else)",
    "nuAlFe": r"$\mathrm{N + (Al, Fe)\to X}$",
    "att": "Attenuation",
    "others": "Others",
    "total": "Total",
}

ppfx_mesinc_labels = {
    "mesinc_projectile_K0": r"$\mathrm{K^0 + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_Km": r"$\mathrm{K^{-} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_Kp": r"$\mathrm{K^{+} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_pip": r"$\mathrm{\pi^{+} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_projectile_pim": r"$\mathrm{\pi^{-} + A \to \left( \pi^\pm, K, N \right) + X}$",
    "mesinc_daughter_K0": r"$\mathrm{\left( \pi^\pm, K \right) + A \to K^0 + X}$",
    "mesinc_daughter_Km": r"$\mathrm{\left( \pi^\pm, K \right) + A \to K^{-} + X}$",
    "mesinc_daughter_Kp": r"$\mathrm{\left( \pi^\pm, K \right) + A \to K^{+} + X}$",
    "mesinc_daughter_pip": r"$\mathrm{\left( \pi^\pm, K \right) + A \to \pi^{+} + X}$",
    "mesinc_daughter_pim": r"$\mathrm{\left( \pi^\pm, K \right) + A \to \pi^{-} + X}$",
}

ppfx_colors = {
    k: c
    for k, c in zip(ppfx_labels, cycle(colorscheme.values()))
    if "projectile" not in k and "daughter" not in k
}
ppfx_mesinc_colors = {
    k: c
    for k, c in zip(ppfx_mesinc_labels, cycle(colorscheme.values()))
    if "projectile" in k or "daughter" in k
}

ppfx_lines = {
    k: ls
    for k, ls in zip(ppfx_labels, cycle(["-", "--", ":"]))
    if "projectile" not in k and "daughter" not in k
}

beam_syst_labels = {
    "beam_shift_y_plus": r"Beam Shift $y+1$ mm",
    "beam_shift_y_minus": r"Beam Shift $y-1$ mm",
    "beam_shift_y": r"Beam Shift $y \pm 1$ mm",
    "beam_shift_x_plus": r"Beam Shift $x+1$ mm",
    "beam_shift_x_minus": r"Beam Shift $x-1$ mm",
    "beam_shift_x": r"Beam Shift $x \pm 1$ mm",
    "beam_spot": r"Beam Spot Size $\pm 0.2$ cm",
    "horn_current_plus": "Horn Current $+ 2$ kA",
    "horn_current_minus": "Horn Current $- 2$ kA",
    "horn_current": r"Horn Current $\pm 2$ kA",
    "horn1_y": r"Horn 1 y-Position $\pm 0.3$ cm",
    "horn1_x": r"Horn 1 x-Position $\pm 0.3$ cm",
    "water_layer": r"Horn Water Layer $\pm 1$ mm",
    "beam_div": r"54 $\mathrm{\mu}$rad Beam Divergence",
    "total": "Total",
}

beam_syst_colors = {k: c for k, c in zip(beam_syst_labels, cycle(colorscheme.values()))}

beam_syst_lines = {
    k: ls
    for k, ls in zip(beam_syst_labels, cycle(["-", "--", ":"]))
    if "total" not in k
}


xlabel_enu = r"E$_\mathrm{\nu}$ [GeV]"
ylabel_flux = r"$\mathrm{\phi_\nu}$  / GeV / POT"


def get_prop_cycler() -> Cycler:
    """Returns an axis prop cycler, which cycles first through solid lines with the Okabe Ito colorscheme, then dashed lines.
    Excludes the black color.
    """
    prop_cycle = cycler(linestyle=["-", "--", "-."]) * cycler(
        color=[v for k, v in okabe_ito.items() if k != "black"]
    )

    return prop_cycle


def place_header(
    ax: Axes, header: str, xy: tuple[float, float] = (0.0, 1.0), ha="left", **kwargs
) -> None:
    ax.annotate(
        header,
        xy,
        xytext=(0, 3),
        xycoords="axes fraction",
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontweight="bold",
        **kwargs,
    )


def icarus_preliminary(
    ax: Axes, xy: tuple[float, float] = (0.0, 1.0), **kwargs
) -> None:
    """
    Annotates a Matplotlib axes object with the label "ICARUS Preliminary".

    This function adds a text annotation to the specified axes at the given
    coordinates (xy). The text "ICARUS Preliminary" is used as the annotation,
    and it is positioned with an offset from the specified coordinates.

    Parameters:
    ax (Axes): The Matplotlib axes object to which the annotation will be added.
    xy (tuple[float, float], optional): The coordinates (x, y) where the annotation
        arrow will point. Default is (0.0, 1.0).
    **kwargs: Additional keyword arguments that are passed to the `annotate` function
        of Matplotlib.

    Returns:
    None

    Example:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    icarus_preliminary(ax)  # Annotate with "ICARUS Preliminary" label
    plt.show()
    """
    ax.annotate(
        "ICARUS Preliminary",
        xy,
        xytext=(0, 3),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontweight="bold",
        **kwargs,
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
