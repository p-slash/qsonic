import numpy as np
from scipy.optimize import minimize

def _continuum_chi2(a, b, wave_rf, norm_flux, ivar):
    chi2 = 0

    for arm in spectrum.arms:
        wv = wave_rf[arm]
        if wv.size == 0:
            continue

        Slope = np.log(wave_rf)
        Slope = (Slope-Slope[0])/(Slope[-1]-Slope[0])

        chi2 += np.sum(
            ivar[arm]*(norm_flux[arm]-(a+b*Slope))**2
        )

    return chi2

class ContinuumFitter(object):
    def __init__(self, w1rf, w2rf, dwrf):
        self.nbins = int((w2rf-w1rf)/dwrf)+1
        self.rfwave = w1rf + np.arange(self.nbins)*dwrf

        self.mean_cont = np.ones(self.nbins)

    def get_normalized_flux(self, spectrum):
        norm_flux = {}

        for arm in spectrum.arms:
            wv = spectrum.forestwave[arm]/(1+spectrum.z_qso)
            cont_est = np.interp(
                wv,
                self.rfwave,
                self.mean_cont
            )
            # Multiply with resolution
            norm_flux[arm] = spectrum.forestflux[arm]/cont_est

        return norm_flux

    def fit_continuum(self, spectrum):
        wave_rf = spectrum.forestwave[arm]
        norm_flux = self.get_normalized_flux(spectrum)

        a0, b0 = spectrum.cont_params['a'], spectrum.cont_params['b']
        result = minimize(_continuum_chi2, (a0, b0),
            args=(wave_rf, norm_flux, ivar),
            method=None,
            jac=None
        )

        spectrum.cont_params['valid'] = result.success
        if result.success:
            spectrum.cont_params['a'] = result.x[0]
            spectrum.cont_params['b'] = result.x[1]


        

