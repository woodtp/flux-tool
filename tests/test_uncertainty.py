import numpy as np
import pandas as pd

from flux_tool.uncertainty import flux_uncertainty


class TestUncertainty:
    def test_flux_uncertainty(self):
        flux: float = 1.0
        cov = pd.DataFrame(np.diag([1, 2, 3, 4]))
        uncert_true = np.sqrt(cov.sum().sum() / flux**2)

        uncert = flux_uncertainty(cov, flux)

        assert uncert == uncert_true
