from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from ROOT import TH1D, TH2D, TMatrixD  # type: ignore


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


def convert_pandas_to_th1(
    series: pd.Series,
    bin_edges: np.ndarray,
    hist_title: Optional[str] = None,
    uncerts: Optional[pd.Series] = None,
) -> TH1D:
    if hist_title is None:
        hist_title = str(uuid4())

    th1 = TH1D(hist_title, "", len(bin_edges) - 1, bin_edges)

    if uncerts is None:
        for b, val in enumerate(series.values):
            th1.SetBinContent(b + 1, val)
    else:
        for b, (val, unc) in enumerate(zip(series.values, uncerts.values)):
            th1.SetBinContent(b + 1, val)
            th1.SetBinError(b + 1, unc)

    return th1


def convert_pandas_to_th2(dataframe: pd.DataFrame, hist_title: str) -> TH2D:
    rows = dataframe.index
    columns = dataframe.columns

    xbins = np.arange(0, len(rows) + 1, dtype=float)
    ybins = np.arange(0, len(columns) + 1, dtype=float)
    nbinsx = xbins.shape[0] - 1
    nbinsy = ybins.shape[0] - 1

    th2 = TH2D(hist_title, "", nbinsx, xbins, nbinsy, ybins)

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


def convert_pandas_to_tmatrix(matrix: pd.DataFrame) -> TMatrixD:
    nrow, ncol = matrix.shape
    return TMatrixD(nrow, ncol, matrix.values, "D")
