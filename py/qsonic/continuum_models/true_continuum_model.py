import logging
import numpy as np

from qsonic.mathtools import FastCubic1DInterp
from qsonic.continuum_models.base_continuum_model import BaseContinuumModel


class TrueContinuumModel(BaseContinuumModel):
    """True continuum model class for mock analysis. Uses fiducials for mean
    flux and varlss interpolation. Continuum is interpolated using a cubic
    spline.

    Parameters
    ----------
    meanflux_interp: FastLinear1DInterp
        Interpolator for mean flux. If fiducial is not set, this equals to 1.
    varlss_interp: FastLinear1DInterp or FastCubic1DInterp
        Cubic spline for var_lss if fitting. Linear if from file.
    """

    def __init__(self, meanflux_interp, varlss_interp):
        self.meanflux_interp = meanflux_interp
        self.varlss_interp = varlss_interp

    def fit_continuum(self, spec):
        """True continuum reduction. Uses fiducials for mean flux and varlss
        interpolation. Continuum is interpolated using a cubic spline.

        Arguments
        ---------
        spec: Spectrum
            Spectrum object to add true continuum.
        """
        w1 = spec.cont_params['true_data_w1']
        dwave = spec.cont_params['true_data_dwave']
        tcont = spec.cont_params['true_data']

        tcont_interp = FastCubic1DInterp(w1, dwave, tcont)

        for arm, wave_arm in spec.forestwave.items():
            cont_est = tcont_interp(wave_arm)
            cont_est *= self.meanflux_interp(wave_arm)
            spec.cont_params['cont'][arm] = cont_est

        spec.set_forest_weight(self.varlss_interp)
        spec.calc_continuum_chi2()

    def init_spectra(self, spectra_list):
        """ Initializes
        :attr:`cont_params <qsonic.spectrum.Spectrum.cont_params>` for a list
        of Spectrum objects.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects.
        """
        logging.info("Initializing true continuum.")

        for spec in spectra_list:
            spec.cont_params['method'] = 'true'
            spec.cont_params['valid'] = True
            spec.cont_params['x'] = np.zeros(1)
            spec.cont_params['xcov'] = np.eye(1)
            spec.cont_params['dof'] = spec.get_real_size()
            spec.cont_params['cont'] = {}

    def stacks_residual_flux(self):
        """:meth:`stack_spectra` stacks actual continuum values.

        Returns
        -------
        False: bool
        """
        return False
