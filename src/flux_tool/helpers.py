from typing import Any, Callable, Optional, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from ROOT import TH1D, TH2D, TMatrixDSym  # type: ignore


def get_bin_edges_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    energy_columns = ["E_low", "E_high"]
    if not set(energy_columns).issubset(df.columns):
        raise ValueError(
            "DataFrame does not contain the lower and upper energy bin edges."
        )
    return np.unique(df[energy_columns].to_numpy().flatten())


def calculate_correlation_matrix(
    covariance_matrix: pd.DataFrame | NDArray[Any],
) -> pd.DataFrame | NDArray[Any]:
    """Calculates the correlation matrix for a given covariance matrix."""
    variance = np.sqrt(np.diag(covariance_matrix))
    outer_product = np.outer(variance, variance)
    correlation_matrix = np.divide(
        covariance_matrix,
        outer_product,
        out=np.zeros(covariance_matrix.shape),
        where=outer_product != 0,
    )
    return correlation_matrix

def rebin_within_xlim(hist: TH1D, binning: NDArray, xlim: tuple[float, float]) -> TH1D:
    new_binning = binning[(binning >= xlim[0]) & (binning <= xlim[1])]
    hist = hist.Rebin(len(new_binning) - 1, hist.GetName(), new_binning)
    return hist

def convert_pandas_to_th1(
    series: pd.Series | pd.DataFrame,
    bin_edges: Iterable[float],
    hist_title: str = "",
    hist_name: Optional[str] = None,
    uncerts: Optional[pd.Series] = None,
) -> TH1D:
    if hist_name is None:
        hist_name = str(uuid4())

    th1 = TH1D(hist_name, hist_title, len(bin_edges) - 1, np.asarray(bin_edges))

    if uncerts is None:
        for b, val in enumerate(series.values):
            th1.SetBinContent(b + 1, val)
    else:
        for b, (val, unc) in enumerate(zip(series.values, uncerts.values)):
            th1.SetBinContent(b + 1, val)
            th1.SetBinError(b + 1, unc)

    return th1


def convert_pandas_to_th2(dataframe: pd.DataFrame, hist_name: str) -> TH2D:
    rows = dataframe.index
    columns = dataframe.columns

    xbins = np.arange(0, len(rows) + 1, dtype=float)
    ybins = np.arange(0, len(columns) + 1, dtype=float)
    nbinsx = xbins.shape[0] - 1
    nbinsy = ybins.shape[0] - 1

    th2 = TH2D(hist_name, "", nbinsx, xbins, nbinsy, ybins)

    for ii, (_, row) in enumerate(dataframe.iterrows()):
        for jj, (_, col) in enumerate(row.items()):
            th2.SetBinContent(ii + 1, jj + 1, col)

    it = zip(enumerate(rows), enumerate(columns))

    for (ii, row), (jj, col) in it:
        labelx = f"{row[0]}-{row[1]}-{row[2]}"
        labely = f"{col[0]}-{col[1]}-{col[2]}"
        th2.GetXaxis().SetBinLabel(ii + 1, labelx)
        th2.GetYaxis().SetBinLabel(jj + 1, labely)

    return th2


def convert_symmetric_ndarray_to_tmatrix(matrix: NDArray) -> TMatrixDSym:
    m = TMatrixDSym(matrix.shape[0])
    for i, x in enumerate(matrix):
        for j, y in enumerate(x):
            if i == j:
                m[i, i] = y
            elif i < j:
                m[i, j] = y
                m[j, i] = y
    return m


def convert_groups_to_dict(
    df_groups: SeriesGroupBy | DataFrameGroupBy,
    bins: dict[str, NDArray],
    hist_name_builder: Callable[..., str],
    hist_title: str,
    directory_builder: Callable[[str], str] = lambda _: "",
    has_uncerts: bool = False,
) -> dict[str, TH1D]:
    out_dict: dict[str, TH1D] = {}

    for idx, hist in df_groups:
        nu = idx[-1]  # type: ignore
        hist_name = hist_name_builder(*idx)  # type: ignore
        if has_uncerts:
            th1 = convert_pandas_to_th1(
                hist.iloc[:, 0], bins[nu], hist_name, hist_title, hist.iloc[:, 1]
            )
        else:
            th1 = convert_pandas_to_th1(hist, bins[nu], hist_name, hist_title)
        out_dict[f"{directory_builder(idx[0])}/{hist_name}"] = th1  # type: ignore

    return out_dict
