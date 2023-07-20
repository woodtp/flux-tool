from ROOT import TH1D  # type: ignore


def absolute_uncertainty(total_flux: TH1D, fractional_uncertainty: TH1D) -> TH1D:
    return total_flux * fractional_uncertainty
