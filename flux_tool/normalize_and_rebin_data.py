from contextlib import contextmanager
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import cppyy
import numpy as np
import pandas as pd
import uproot
from ROOT import TFile  # type: ignore

TH1D = cppyy.gbl.TH1D


class HistInfo(NamedTuple):
    category: str
    neutrino: str
    universe: Optional[int] = None


def parse_th1_name(name: str) -> HistInfo:
    name = name[1:]
    split = name.split("_")

    match split:
        case ["nom", neutrino]:
            return HistInfo(category="nominal", neutrino=neutrino)
        case ["cv", neutrino]:
            return HistInfo(category="central_value", neutrino=neutrino)
        case ["nom", neutrino, category]:
            return HistInfo(category=category, neutrino=neutrino)
        case ["thin", *category, neutrino, universe]:
            return HistInfo(
                category="_".join(category), neutrino=neutrino, universe=int(universe)
            )
        case [category, neutrino, universe]:
            return HistInfo(
                category=category, neutrino=neutrino, universe=int(universe)
            )
        case _:
            raise ValueError(f"Cannot parse TH1 name: {name}")


@contextmanager
def open_tfile(file_path: Path, mode: str = "read"):
    """Context manager for opening and closing a ROOT TFile."""
    tfile = TFile(str(file_path), mode)
    try:
        yield tfile
    finally:
        tfile.Close()


def calculate_df(h: TH1D, horn: str, run_id: int, parsed: HistInfo) -> pd.DataFrame:
    nbins = len(h) - 2

    flux = np.zeros(nbins)
    stat_uncert = np.zeros(nbins)

    for i in range(1, nbins + 1):
        flux[i - 1] = h.GetBinContent(i)
        stat_uncert[i - 1] = h.GetBinError(i)

    df = pd.DataFrame(
        {
            "flux": flux,
            "stat_uncert": stat_uncert,
            "bin": range(1, nbins + 1),
            "category": parsed.category,
            "neutrino_mode": parsed.neutrino,
            "horn_polarity": horn,
            "run_id": run_id,
        }
    )

    if parsed.universe is not None:
        df["universe"] = parsed.universe

    return df


def normalize_flux_to_pot(
    input_file: Path,
    horn: str,
    run_id: int,
    bin_edges: Optional[dict[str, np.ndarray]] = None,
    hist_name_filter: Optional[Callable[[str], bool]] = None,
) -> pd.DataFrame:
    """Normalizes flux histograms to POT and saves the data in a Pandas DataFrame.

    Args:
        input_file: Path to the input ROOT file containing the flux histograms and POT.
        horn: Horn polarity of the neutrino beam (either "FHC" or "RHC").
        run_id: ID number of the run for which the histograms were produced.
        bin_edges: Optional array of bin edges for rebinning the histograms.
        hist_name_filter: Optional function for filtering the histogram names.

    Returns:
        A Pandas DataFrame with columns for the flux, statistical uncertainty, bin number
    """

    with uproot.open(input_file) as f:  # type: ignore
        histkeys = f.keys(
            cycle=False,
            filter_classname="TH1D",
            filter_name=hist_name_filter,
        )

    with open_tfile(input_file) as tfile:
        pot = tfile.Get("hpot").GetMaximum()

        hlist = []

        for key in histkeys:
            _, hist_name = key.rsplit("/", 1)

            parsed: HistInfo = parse_th1_name(hist_name)

            h = tfile.Get(key)

            if bin_edges is not None:
                bins = bin_edges[parsed.neutrino]
                h = h.Rebin(len(bins) - 1, hist_name, bins)

            h.Scale(1.0 / pot)

            df = calculate_df(h, horn, run_id, parsed)

            hlist.append(df)

    return pd.concat(hlist)
