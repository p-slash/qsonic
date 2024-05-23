import logging
import numpy as np
from iminuit import Minuit
from scipy.optimize import minimize

from qsonic import QsonicException
from qsonic.mathtools import mypoly1d
from qsonic.continuum_models.base_continuum_model import BaseContinuumModel


class PiccaContinuumModel(BaseContinuumModel):
    """Picca continuum model class.

    Parameters
    ----------
    meancont_interp: FastCubic1DInterp
        Fast cubic spline object for the mean continuum.
    meanflux_interp: FastLinear1DInterp
        Interpolator for mean flux. If fiducial is not set, this equals to 1.
    varlss_interp: FastLinear1DInterp or FastCubic1DInterp
        Cubic spline for var_lss if fitting. Linear if from file.
    eta_interp: FastCubic1DInterp
        Interpolator for eta. Returns one if fiducial var_lss is set.
    cont_order: int
        Order of continuum polynomial from ``args.cont_order``.
    rfwave0: float
        First rest-frame wavelength center for the mean continuum.
    denom: float
        Denominator for the slope term in the continuum model.
    minimizer: str
        ``iminuit`` or ``l_bfgs_b`` to select the minimizer function.

    Attributes
    ----------
    minimizer: function
        Function that points to one of the minimizer options.
    """

    def __init__(
            self, meancont_interp, meanflux_interp, varlss_interp,
            eta_interp, cont_order, rfwave0, denom, minimizer
    ):
        self.meancont_interp = meancont_interp
        self.meanflux_interp = meanflux_interp
        self.varlss_interp = varlss_interp
        self.eta_interp = eta_interp
        self.cont_order = cont_order
        self.rfwave0 = rfwave0
        self.denom = denom

        if minimizer == "iminuit":
            self.minimizer = self._iminuit_minimizer
        elif minimizer == "l_bfgs_b":
            self.minimizer = self._scipy_l_bfgs_b_minimizer
        else:
            raise QsonicException(
                "Undefined minimizer. Developer forgot to implement.")

    def _continuum_costfn(self, x, wave, flux, ivar_sm, z_qso):
        """Cost function to minimize for each quasar.

        This is a modified chi2 where amplitude is also part of minimization.
        Cost of each arm is simply added to the total cost.

        Arguments
        ---------
        x: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Polynomial coefficients for quasar diversity.
        wave: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
            Observed-frame wavelengths.
        flux: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
            Flux.
        ivar_sm: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
            Smooth inverse variance.
        z_qso: float
            Quasar redshift.

        Returns
        ---------
        cost: float
            Cost (modified chi2) for a given ``x``.
        """
        cost = 0

        for arm, wave_arm in wave.items():
            cont_est = self.get_continuum_model(x, wave_arm / (1 + z_qso))
            # no_neg = np.sum(cont_est<0)
            # penalty = wave_arm.size * no_neg**2

            cont_est *= self.meanflux_interp(wave_arm)

            var_lss = self.varlss_interp(wave_arm) * cont_est**2
            eta = self.eta_interp(wave_arm)
            weight = ivar_sm[arm] / (eta + ivar_sm[arm] * var_lss)
            w = weight > 0

            cost += np.dot(
                weight, (flux[arm] - cont_est)**2
            ) - np.log(weight[w]).sum()  # + penalty

        return cost

    def get_continuum_model(self, x, wave_rf_arm):
        """Returns interpolated continuum model.

        Arguments
        ---------
        x: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Polynomial coefficients for quasar diversity.
        wave_rf_arm: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Rest-frame wavelength per arm.

        Returns
        ---------
        cont: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Continuum at `wave_rf_arm` values given `x`.
        """
        slope = np.log(wave_rf_arm / self.rfwave0) / self.denom

        cont = self.meancont_interp(wave_rf_arm) * mypoly1d(x, 2 * slope - 1)
        # Multiply with resolution
        # Edges are difficult though

        return cont

    def _iminuit_minimizer(self, spec, a0):
        def _cost(x):
            return self._continuum_costfn(
                x, spec.forestwave, spec.forestflux, spec.forestivar_sm,
                spec.z_qso)

        x0 = np.zeros_like(spec.cont_params['x'])
        x0[0] = a0
        mini = Minuit(_cost, x0)
        mini.errordef = Minuit.LEAST_SQUARES
        mini.migrad()

        result = {}

        result['valid'] = mini.valid
        result['x'] = np.array(mini.values)
        result['xcov'] = np.array(mini.covariance)

        return result

    def _scipy_l_bfgs_b_minimizer(self, spec, a0):
        x0 = np.zeros_like(spec.cont_params['x'])
        x0[0] = a0
        mini = minimize(
            self._continuum_costfn,
            x0,
            args=(spec.forestwave,
                  spec.forestflux,
                  spec.forestivar_sm,
                  spec.z_qso),
            method='L-BFGS-B',
            bounds=None,
            jac=None
        )

        result = {}

        result['valid'] = mini.success
        result['x'] = mini.x
        result['xcov'] = mini.hess_inv.todense()

        return result

    def stack_spectra(self, valid_spectra_list, flux_stacker):
        """Stacks spectra in the observed and rest-frame. Observed-frame and
        rest-frame stacking is performed over residuals f/C.

        Arguments
        ---------
        valid_spectra_list: list(Spectrum)
            Valid spectra objects to iterate.
        flux_stacker: FluxStacker
            Flux stacker object.
        """
        for spec in valid_spectra_list:
            for arm, wave_arm in spec.forestwave.items():
                wave_rf_arm = wave_arm / (1 + spec.z_qso)

                cont = spec.cont_params['cont'][arm]
                weight = spec.forestweight[arm] * cont**2
                weighted_flux = weight * spec.forestflux[arm] / cont

                flux_stacker.add(
                    wave_arm, wave_rf_arm, weighted_flux, weighted_flux, weight
                )

    def fit_continuum(self, spec):
        """Fits the continuum for a single Spectrum.

        This function uses
        :attr:`forestivar_sm <qsonic.spectrum.Spectrum.forestivar_sm>` in
        inverse variance, which must be smoothed beforehand.
        It also modifies
        :attr:`cont_params <qsonic.spectrum.Spectrum.cont_params>`
        dictionary's ``valid, cont, x, xcov, chi2, dof`` keys.
        If the best-fitting continuum is **negative at any point**, the fit is
        **invalidated**. Chi2 is set separately without using the
        :meth:`cost function <._continuum_costfn>`.
        ``x`` key is the best-fitting parameter, and ``xcov`` is their inverse
        Hessian ``hess_inv`` given by
        :external+scipy:func:`scipy.optimize.minimize` using 'L-BFGS-B' method.

        Arguments
        ---------
        spec: Spectrum
            Spectrum object to fit.
        """
        # We can precalculate meanflux and varlss here,
        # and store them in respective keys to spec.cont_params

        def get_a0():
            a0 = 0
            n0 = 1e-6
            for arm, ivar_arm in spec.forestivar_sm.items():
                a0 += np.dot(spec.forestflux[arm], ivar_arm)
                n0 += np.sum(ivar_arm)

            return a0 / n0

        result = self.minimizer(spec, get_a0())
        spec.cont_params['valid'] = result['valid']

        if spec.cont_params['valid']:
            spec.cont_params['cont'] = {}
            for arm, wave_arm in spec.forestwave.items():
                cont_est = self.get_continuum_model(
                    result['x'], wave_arm / (1 + spec.z_qso))

                if any(cont_est < 0):
                    spec.cont_params['valid'] = False
                    break

                cont_est *= self.meanflux_interp(wave_arm)
                # cont_est *= self.flux_stacker(wave_arm)
                spec.cont_params['cont'][arm] = cont_est

        spec.set_forest_weight(self.varlss_interp, self.eta_interp)
        # We can further eliminate spectra based chi2
        spec.calc_continuum_chi2()

        if spec.cont_params['valid']:
            spec.cont_params['x'] = result['x']
            spec.cont_params['xcov'] = result['xcov']
        else:
            spec.cont_params['cont'] = None
            spec.cont_params['chi2'] = -1

    def init_spectra(self, spectra_list):
        """ Initializes
        :attr:`cont_params <qsonic.spectrum.Spectrum.cont_params>` for a list
        of Spectrum objects.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects to fit.
        """
        logging.info("Initializing Picca continuum fitting.")

        for spec in spectra_list:
            spec.cont_params['method'] = 'picca'
            spec.cont_params['x'] = np.append(
                spec.cont_params['x'][0], np.zeros(self.cont_order))
            spec.cont_params['xcov'] = np.eye(self.cont_order + 1)
            spec.cont_params['dof'] = \
                spec.get_real_size() - self.cont_order - 1

    def stacks_residual_flux(self):
        """:meth:`stack_spectra` stacks residual flux values f/C.

        Returns
        -------
        True: bool
        """
        return True
