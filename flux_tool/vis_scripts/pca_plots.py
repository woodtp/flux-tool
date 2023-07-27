from itertools import product
from pathlib import Path
from functools import reduce

import matplotlib.pyplot as plt
import mplhep as hep
import uproot

from flux_tool.vis_scripts.helper import absolute_uncertainty
from flux_tool.vis_scripts.style import (neutrino_labels, place_header,
                                         ppfx_labels, style, xlabel_enu)


def plot_hadron_systs_and_pca_variances(
    products_file: Path | str, output_dir: str = "plots/pca"
) -> None:
    plt.style.use(style)

    xaxis_lim = (0, 6)
    yaxis_lim = (0, 0.03)

    npcs = 8

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}

    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        product(["fhc", "rhc"], ["numu", "numubar", "nue", "nuebar"]),
        ["daughter", "projectile"],
    )

    all_versions_list = [(*v[0], v[1]) for v in all_versions]
    with uproot.open(products_file) as f:
        for horn, nu, ver in all_versions_list:
            flux = f[f"ppfx_corrected_flux/total/htotal_{horn}_{nu}"].to_pyroot()

            total_uncert = f[
                f"fractional_uncertainties/hadron/total/hfrac_hadron_total_{horn}_{nu}"
            ].to_pyroot()

            hadron_uncerts = {
                key.split("/")[0]: h.to_pyroot()
                for key, h in f["fractional_uncertainties/hadron/"].items(
                    filter_name=f"*{horn}*{nu}", cycle=False
                )
                if "total" not in key and ver not in key and "mesinc/" not in key
            }

            pcs = [
                f[f"pca/principal_components/hpc_{x}_{horn}_{nu}"].to_pyroot()
                for x in range(npcs)
            ]

            eigenvals, _ = f["pca/heigenvals_frac"].to_numpy()

            eigenvals = eigenvals[:npcs]

            absolute_uncertainties = {
                k: absolute_uncertainty(flux, h) for k, h in hadron_uncerts.items()
            }

            total_variance = total_uncert * total_uncert

            hadron_variances = {k: h * h for k, h in hadron_uncerts.items()}

            sorted_hadron_variances = {
                k: v
                for k, v in sorted(
                    hadron_variances.items(),
                    key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
                    reverse=True,
                )
            }

            total_variance_from_had_systs = reduce(lambda h1, h2: h1+h2, hadron_variances.values())

            hadron_labels = [ppfx_labels[k] for k in sorted_hadron_variances]

            pcs_variances = [pc * pc for pc in pcs]

            pc_labels = [
                f"$\\mathrm{{\\lambda_{i}}}$ ({var*100:0.1f}%)"
                for i, var in enumerate(eigenvals)
            ]

            if ver == "daughter":
                actual = "projectile"
            else:
                actual = "daughter"

            _, axs = plt.subplots(
                1, 2, sharey=True, figsize=(26, 14), gridspec_kw={"wspace": 0.03}
            )

            hep.histplot(
                ax=axs[0],
                H=total_variance_from_had_systs,
                yerr=False,
                histtype="fill",
                linestyle="--",
                lw=3,
                color="w",
                edgecolor="gray",
                hatch="//",
                zorder=0,
                label=r"Total of all channels",
            )

            hep.histplot(
                ax=axs[0],
                H=total_variance,
                yerr=False,
                histtype="step",
                linestyle="--",
                lw=3,
                color="k",
                label="PPFX Total",
                zorder=10,
            )

            hep.histplot(
                ax=axs[0],
                H=list(sorted_hadron_variances.values())[:npcs],
                yerr=False,
                histtype="fill",
                stack=True,
                edges=False,
                lw=3,
                label=hadron_labels[:npcs],
            )

            hep.histplot(
                ax=axs[1],
                H=total_variance,
                yerr=False,
                histtype="fill",
                linestyle="--",
                lw=3,
                color="w",
                edgecolor="k",
                hatch="//",
                zorder=0,
                label=r"PPFX Total",
            )

            hep.histplot(
                ax=axs[1],
                H=total_variance,
                yerr=False,
                histtype="step",
                linestyle="--",
                lw=3,
                color="k",
                zorder=10,
            )

            hep.histplot(
                ax=axs[1],
                H=pcs_variances,
                yerr=False,
                histtype="fill",
                stack=True,
                edges=False,
                lw=3,
                label=pc_labels,
            )

            for ax in axs:
                ax.set_xlim(*xaxis_lim)
                ax.set_ylim(*yaxis_lim)
                ax.legend(loc="upper center", fontsize=24, ncol=2)
                ax.set_xlabel(xlabel_enu)

            axs[0].set_ylabel(
                r"Fractional Variance $\mathrm{\left( \sigma / \phi \right)^2}$"
            )

            place_header(axs[0], f"{header[horn]} {neutrino_labels[nu]}")
            axs[1].text(
                0.75,
                1.015,
                r"$\mathrm{\sum \, \lambda_n =}$" + f" {eigenvals.sum()*100:0.1f}%",
                fontweight="bold",
                fontstyle="italic",
                fontsize=28,
                transform=axs[1].transAxes,
            )

            plt.savefig(
                f"{output_dir}/{horn}_{nu}_{version[actual]}_hadron_systs_and_pca_variances.pdf"
            )