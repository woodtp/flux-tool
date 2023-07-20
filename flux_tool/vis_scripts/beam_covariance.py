from pathlib import Path

import uproot
import mplhep as hep
import matplotlib.pyplt as plt

from flux_tool.style import style

def plot_beam_covariance_matrices(products_file: Path | str):
    plt.style.use(plt)
    ...
