from functools import cached_property
from itertools import product
from pathlib import Path

import uproot


class SpectraReader:
    def __init__(self, products_file: Path | str):
        self._f = uproot.open(products_file)

        horns_nus = [
            key.split("_")[0].split("/")
            for key in self._f["ppfx_output"].keys(filter_name="*/nu*", cycle=False)  # type: ignore
        ]

        self.horn_current = list({item[0] for item in horns_nus})
        self.neutrinos = list({item[1] for item in horns_nus})

        self.horns_and_nus = product(self.horn_current, self.neutrinos)

    def __del__(self):
        self._f.close()  # type: ignore

    def __getitem__(self, key):
        return self._f[key]

    def load_cache(self) -> None:
        self.flux_prediction
        self.flux_weights
        self.hadron_covariance_matrices
        self.hadron_correlation_matrices
        self.hadron_uncertainties
        self.nominal_spectra
        self.parent_spectra
        self.pot
        self.ppfx_correction
        self.principal_components
        self.universes

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
        return {
            "fhc": self._f["ppfx_output/fhc/hpot"].values().max(),  # type: ignore
            "rhc": self._f["ppfx_output/rhc/hpot"].values().max(),  # type: ignore
        }

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
    def universes(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                filter_name="*/*_total/*", cycle=False  # type: ignore
            )
        }
