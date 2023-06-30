import numpy as np
import pandas as pd


def extract_statistical_uncertainties(
    nominal_dataframe: pd.DataFrame, flux_weights: pd.DataFrame, normalized=False
) -> pd.Series:
    index = ("category", "horn_polarity", "neutrino_mode", "bin")
    pivot_table = pd.pivot_table(
        nominal_dataframe, index=index, values=["flux", "stat_uncert"]
    )

    pt = pivot_table.loc["nominal"].mul(flux_weights, axis=0)

    stats = pt["stat_uncert"]

    if normalized:
        return stats / pt["flux"]

    return stats


def flux_uncertainty(cov: pd.DataFrame, total_flux: float) -> float:
    return np.sqrt(cov.sum().sum() / total_flux**2)


def ratio_uncertainty(
    cov: pd.DataFrame, total_neutrino1_flux: float, total_neutrino2_flux: float
) -> float:
    reordered_mat = cov.stack("neutrino_mode").swaplevel(i=1, j=2).sort_index()
    if reordered_mat is None:
        raise ValueError("cov is not of the expected format.")

    mat_groups = reordered_mat.groupby(level=(0, 1))

    total = 0.0
    for (nu1, nu2), mat in mat_groups:  # type: ignore
        cov_sum = np.array(mat).sum()
        if nu1 in ["nue", "nuebar"] and nu2 in ["nue", "nuebar"]:
            div = total_neutrino1_flux**-2
            coeff = 1
        elif nu1 in ["numu", "numubar"] and nu2 in ["numu", "numubar"]:
            div = total_neutrino2_flux**-2
            coeff = 1
        else:
            div = 1 / (total_neutrino1_flux * total_neutrino2_flux)
            coeff = -1
        total += coeff * cov_sum * div

    return np.sqrt(total)
