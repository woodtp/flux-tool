from pathlib import Path

import uproot


class SpectraReader:
    __slots__ = (
        "_f",
        "flux_weights",
        "hadron_uncertainties",
        "nominal_spectra",
        "parent_spectra",
        "pot",
        "ppfx_correction",
        "principal_components",
        "universes",
    )

    def __init__(self, products_file: Path | str):
        self._f = uproot.open(products_file)

    def __del__(self):
        self._f.close()  # type: ignore

    def __getitem__(self, key):
        return self._f[key]

    def load_cache(self) -> None:
        self.flux_weights = self._flux_weights()
        self.hadron_uncertainties = self._hadron_uncertainties()
        self.nominal_spectra = self._nominal_spectra()
        self.parent_spectra = self._parent_spectra()
        self.pot = self._pot()
        self.ppfx_correction = self._ppfx_correction()
        self.principal_components = self._principal_components()
        self.universes = self._universes()

    def _flux_weights(self):
        return {
            key: h for key, h in self._f["ppfx_flux_weights"].items(cycle=False)  # type: ignore
        }

    def _hadron_uncertainties(self):
        return {
            key: h
            for key, h in self._f["fractional_uncertainties/hadron/"].items(
                cycle=False, filter_classname="TH1D"  # type: ignore
            )
        }

    def _nominal_spectra(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                cycle=False, filter_name="*/nom/hnom*"  # type: ignore
            )
        }

    def _parent_spectra(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                cycle=False, filter_name="*/nom/parent/*"  # type: ignore
            )
        }

    def _pot(self):
        return {
            "fhc": self._f["ppfx_output/fhc/hpot"].values().max(),  # type: ignore
            "rhc": self._f["ppfx_output/rhc/hpot"].values().max(),  # type: ignore
        }

    def _ppfx_correction(self):
        return {
            key: h for key, h in self._f["ppfx_corrected_flux/total"].items(cycle=False)  # type: ignore
        }

    def _principal_components(self):
        return {
            key: h
            for key, h in self._f["pca/principal_components"].items(
                cycle=False, filter_name="*hpc_*"  # type: ignore
            )
        }

    def _universes(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                filter_name="*/*_total/*", cycle=False  # type: ignore
            )
        }
