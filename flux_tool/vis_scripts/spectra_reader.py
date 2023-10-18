from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Iterable

from numpy.typing import NDArray
import uproot


class SpectraReader:
    def __init__(self, products_file: Path | str, binning: dict[str, NDArray]):
        self._f = uproot.open(products_file)

        self.binning = binning

        horns_nus = [
            key.split("_")[0].split("/")
            for key in self._f["ppfx_output"].keys(filter_name="*/nu*", cycle=False)  # type: ignore
        ]


        self.horn_current = list({item[0] for item in horns_nus})
        self.neutrinos = list({item[1] for item in horns_nus})

        self.horns_and_nus = list(product(self.horn_current, self.neutrinos))

    def __del__(self):
        self._f.close()  # type: ignore

    def __getitem__(self, key):
        return self._f[key]

    def load_cache(self) -> None:
        self.beam_covariance_matrices
        self.beam_correlation_matrices
        self.beam_systematic_shifts
        self.beam_uncertainties
        self.flux_prediction
        self.flux_weights
        self.hadron_covariance_matrices
        self.hadron_correlation_matrices
        self.hadron_uncertainties
        self.nominal_spectra
        self.parent_spectra
        self.pot
        self.ppfx_correction
        self.pca_eigenvalues
        self.principal_components
        self.universes

    @cached_property
    def beam_uncertainties(self):
        return {
            key: h
            for key, h in self._f["fractional_uncertainties/beam/"].items(
                cycle=False, filter_classname="TH1D"  # type: ignore
            )
        }

    @cached_property
    def beam_systematic_shifts(self):
        return {key: h for key, h in self._f["beam_systematic_shifts"].items(cycle=False)}  # type: ignore

    @cached_property
    def flux_prediction(self):
        return {key: h for key, h in self._f["flux_prediction"].items(cycle=False)}  # type: ignore

    @cached_property
    def flux_weights(self):
        return {
            key: h for key, h in self._f["ppfx_flux_weights"].items(cycle=False)  # type: ignore
        }

    @cached_property
    def hadron_covariance_matrices(self):
        return {
            key: h
            for key, h in self._f["covariance_matrices/hadron"].items(
                filter_name="*/hcov*", filter_classname="TH2D", cycle=False  # type: ignore
            )
        }

    @cached_property
    def hadron_correlation_matrices(self):
        return {
            key: h
            for key, h in self._f["covariance_matrices/hadron"].items(
                filter_name="*/hcor*", filter_classname="TH2D", cycle=False  # type: ignore
            )
        }

    @cached_property
    def beam_covariance_matrices(self):
        return {
            key: h
            for key, h in self._f["covariance_matrices/beam"].items(
                filter_name="*hcov*", filter_classname="TH2D", cycle=False  # type: ignore
            )
        }

    @cached_property
    def beam_correlation_matrices(self):
        return {
            key: h
            for key, h in self._f["covariance_matrices/beam"].items(
                filter_name="*hcor*", filter_classname="TH2D", cycle=False  # type: ignore
            )
        }

    @cached_property
    def hadron_uncertainties(self):
        return {
            key: h
            for key, h in self._f["fractional_uncertainties/hadron/"].items(
                cycle=False, filter_classname="TH1D"  # type: ignore
            )
        }

    @cached_property
    def nominal_spectra(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                cycle=False, filter_name="*/nom/hnom*"  # type: ignore
            )
        }

    @cached_property
    def parent_spectra(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                cycle=False, filter_name="*/nom/parent/*"  # type: ignore
            )
        }

    @cached_property
    def pot(self):
        pot = {
            horn: self._f[f"ppfx_output/{horn}/hpot"].values().max()  # type: ignore
            for horn in self.horn_current
        }
        return pot

    @cached_property
    def ppfx_correction(self):
        return {
            key: h for key, h in self._f["ppfx_corrected_flux/total"].items(cycle=False)  # type: ignore
        }

    @cached_property
    def principal_components(self):
        return {
            key: h
            for key, h in self._f["pca/principal_components"].items(
                cycle=False, filter_name="*hpc_*"  # type: ignore
            )
        }

    @cached_property
    def pca_eigenvalues(self):
        return self._f["pca/heigenvals_frac"]

    @cached_property
    def universes(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                filter_name="*/*_total/*", cycle=False  # type: ignore
            )
        }
