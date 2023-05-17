"""Continuum fitting module."""
import argparse

import numpy as np
from numba import njit
import fitsio
from iminuit import Minuit
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.special import legendre

from mpi4py import MPI

from qsonic import QsonicException
from qsonic.spectrum import valid_spectra
from qsonic.mpi_utils import logging_mpi, warn_mpi, mpi_fnc_bcast, MPISaver
from qsonic.mathtools import (
    mypoly1d, block_covariance_of_square,
    FastLinear1DInterp, FastCubic1DInterp,
    SubsampleCov
)


def add_picca_continuum_parser(parser=None):
    """ Adds PiccaContinuumFitter related arguments to parser. These
    arguments are grouped under 'Continuum fitting options'. All of them
    come with defaults, none are required.

    Arguments
    ---------
    parser: argparse.ArgumentParser, default: None

    Returns
    ---------
    parser: argparse.ArgumentParser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cont_group = parser.add_argument_group('Continuum fitting options')

    cont_group.add_argument(
        "--rfdwave", type=float, default=0.8,
        help="Rest-frame wave steps. Complies with forest limits")
    cont_group.add_argument(
        "--no-iterations", type=int, default=10,
        help="Number of iterations for continuum fitting.")
    cont_group.add_argument(
        "--fiducial-meanflux", help="Fiducial mean flux FITS file.")
    cont_group.add_argument(
        "--fiducial-varlss", help="Fiducial var_lss FITS file.")
    cont_group.add_argument(
        "--cont-order", type=int, default=1,
        help="Order of continuum fitting polynomial.")
    cont_group.add_argument(
        "--var-fit-eta", action="store_true",
        help="Fit for noise calibration (eta).")
    cont_group.add_argument(
        "--var-use-cov", action="store_true",
        help="Use covariance in varlss-eta fitting.")
    cont_group.add_argument(
        "--normalize-stacked-flux", action="store_true",
        help="NOT IMPLEMENTED: Force stacked flux to be one at the end.")
    cont_group.add_argument(
        "--minimizer", default="iminuit", choices=["iminuit", "l_bfgs_b"],
        help="Minimizer to fit the continuum.")

    return parser


class PiccaContinuumFitter():
    """ Picca continuum fitter class.

    Fits spectra without coadding. Pipeline inverse variance preferably should
    be smoothed before fitting. Mean continuum and var_lss are smoothed using
    inverse weights and cubic spline to help numerical stability.

    When fitting for var_lss, number of wavelength bins in the observed frame
    for variance fitting ``nwbins`` are calculated by demanding 120 A steps
    between bins as closely as possible by :class:`VarLSSFitter`.

    Contruct an instance, then call :meth:`iterate` with local spectra.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace. Wavelength values are taken to be the edges (not centers).
        See respective parsers for default values.

    Attributes
    ----------
    nbins: int
        Number of bins for the mean continuum in the rest frame.
    rfwave: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Rest-frame wavelength centers for the mean continuum.
    _denom: float
        Denominator for the slope term in the continuum model.
    meancont_interp: FastCubic1DInterp
        Fast cubic spline object for the mean continuum.
    minimizer: function
        Function that points to one of the minimizer options.
    comm: MPI.COMM_WORLD
        MPI comm object to reduce, broadcast etc.
    mpi_rank: int
        Rank of the MPI process.
    meanflux_interp: FastLinear1DInterp
        Interpolator for mean flux. If fiducial is not set, this equals to 1.
    flux_stacker: FluxStacker
        Stacks flux. Set up with 8 A wavelength bin size.
    varlss_fitter: VarLSSFitter or None
        None if fiducials are set for var_lss.
    varlss_interp: FastLinear1DInterp or FastCubic1DInterp
        Cubic spline for var_lss if fitting. Linear if from file.
    eta_interp: FastCubic1DInterp
        Interpolator for eta. Returns one if fiducial var_lss is set.
    niterations: int
        Number of iterations from `args.no_iterations`.
    cont_order: int
        Order of continuum polynomial from `args.cont_order`.
    outdir: str or None
        Directory to save catalogs. If None or empty, does not save.
    fit_eta: bool
        True if fitting eta and fiducial var_lss is not set.
    """

    def _get_fiducial_interp(self, fname, col2read):
        """ Return an interpolator for mean flux or var_lss.

        FITS file must have a 'STATS' extention, which must have 'LAMBDA',
        'MEANFLUX' and 'VAR' columns. This is the same format as raw_io output
        from picca. 'LAMBDA' must be linearly and equally spaced.
        This function sets up ``col2read`` as FastLinear1DInterp object.

        Arguments
        ---------
        fname: str
            Filename of the FITS file.
        col2read: str
            Should be 'MEANFLUX' or 'VAR'.

        Returns
        -------
        FastLinear1DInterp

        Raises
        ------
        QsonicException
            If 'LAMBDA' is not equally spaced or ``col2read`` is not in the
            file.
        """
        def _read(fname, col2read):
            with fitsio.FITS(fname) as fts:
                data = fts['STATS'].read()

            waves = data['LAMBDA']
            waves_0 = waves[0]
            dwave = waves[1] - waves[0]
            nsize = waves.size

            if not np.allclose(np.diff(waves), dwave):
                raise Exception(
                    "Failed to construct fiducial mean flux or varlss from "
                    f"{fname}::LAMBDA is not equally spaced.")

            data = np.array(data[col2read], dtype='d')

            return FastLinear1DInterp(waves_0, dwave, data, ep=np.zeros(nsize))

        return mpi_fnc_bcast(
            _read, self.comm, self.mpi_rank,
            f"Failed to construct fiducial mean flux or varlss from {fname}.",
            fname, col2read)

    def __init__(self, args):
        # We first decide how many bins will approximately satisfy
        # rest-frame wavelength spacing. Then we create wavelength edges, and
        # transform these edges into centers
        self.nbins = int(round(
            (args.forest_w2 - args.forest_w1) / args.rfdwave))
        edges, self.dwrf = np.linspace(
            args.forest_w1, args.forest_w2, self.nbins + 1, retstep=True)
        self.rfwave = (edges[1:] + edges[:-1]) / 2
        self._denom = np.log(self.rfwave[-1] / self.rfwave[0])

        self.meancont_interp = FastCubic1DInterp(
            self.rfwave[0], self.dwrf, np.ones(self.nbins),
            ep=np.zeros(self.nbins))

        if args.minimizer == "iminuit":
            self.minimizer = self._iminuit_minimizer
        elif args.minimizer == "l_bfgs_b":
            self.minimizer = self._scipy_l_bfgs_b_minimizer
        else:
            raise QsonicException(
                "Undefined minimizer. Developer forgot to implement.")

        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()

        if args.fiducial_meanflux:
            self.meanflux_interp = self._get_fiducial_interp(
                args.fiducial_meanflux, 'MEANFLUX')
        else:
            self.meanflux_interp = FastLinear1DInterp(
                args.wave1, args.wave2 - args.wave1, np.ones(3))

        self.flux_stacker = FluxStacker(
            args.wave1, args.wave2, 8., comm=self.comm)

        if args.fiducial_varlss:
            self.varlss_fitter = None
            self.varlss_interp = self._get_fiducial_interp(
                args.fiducial_varlss, 'VAR')
        else:
            self.varlss_fitter = VarLSSFitter(
                args.wave1, args.wave2, use_cov=args.var_use_cov,
                comm=self.comm)
            self.varlss_interp = FastCubic1DInterp(
                self.varlss_fitter.waveobs[0], self.varlss_fitter.dwobs,
                0.1 * np.ones(self.varlss_fitter.nwbins),
                ep=np.zeros(self.varlss_fitter.nwbins))

        self.eta_interp = FastCubic1DInterp(
            self.varlss_interp.xp0, self.varlss_interp.dxp,
            np.ones_like(self.varlss_interp.fp),
            ep=np.zeros_like(self.varlss_interp.fp))

        self.niterations = args.no_iterations
        self.cont_order = args.cont_order
        self.outdir = args.outdir
        self.fit_eta = args.var_fit_eta

    def _continuum_costfn(self, x, wave, flux, ivar_sm, z_qso):
        """ Cost function to minimize for each quasar.

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
        """ Returns interpolated continuum model.

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
        slope = np.log(wave_rf_arm / self.rfwave[0]) / self._denom

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

    def fit_continuum(self, spec):
        """ Fits the continuum for a single Spectrum.

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

    def fit_continua(self, spectra_list):
        """ Fits all continua for a list of Spectrum objects.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects to fit.

        Raises
        ------
        QsonicException
            If there are no valid fits.
        RuntimeWarning
            If more than 20% spectra have invalid fits.
        """
        no_valid_fits = 0
        no_invalid_fits = 0

        # For each forest fit continuum
        for spec in spectra_list:
            self.fit_continuum(spec)

            if not spec.cont_params['valid']:
                no_invalid_fits += 1
            else:
                no_valid_fits += 1

        no_valid_fits = self.comm.allreduce(no_valid_fits)
        no_invalid_fits = self.comm.allreduce(no_invalid_fits)
        logging_mpi(f"Number of valid fits: {no_valid_fits}", self.mpi_rank)
        logging_mpi(f"Number of invalid fits: {no_invalid_fits}",
                    self.mpi_rank)

        if no_valid_fits == 0:
            raise QsonicException("Crucial error: No valid continuum fits!")

        invalid_ratio = no_invalid_fits / (no_valid_fits + no_invalid_fits)
        if invalid_ratio > 0.2:
            warn_mpi("More than 20% spectra have invalid fits.", self.mpi_rank)

    def _project_normalize_meancont(self, new_meancont):
        """ Project out higher order Legendre polynomials from the new mean
        continuum since these are degenerate with the free fitting parameters.
        Returns a normalized mean continuum. Integrals are calculated using
        ``np.trapz`` with ``ln lambda_RF`` as x array.

        Arguments
        ---------
        new_meancont: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            First estimate of the new mean continuum.

        Returns
        ---------
        new_meancont: :class:`ndarray <numpy.ndarray>`
            Legendere polynomials projected out and normalized mean continuum.
        mean: float
            Normalization of the mean continuum.
        """
        x = np.log(self.rfwave / self.rfwave[0]) / self._denom

        for ci in range(1, self.cont_order + 1):
            norm = 2 * ci + 1
            leg_ci = legendre(ci)(2 * x - 1)

            B = norm * np.trapz(new_meancont * leg_ci, x=x)
            new_meancont -= B * leg_ci

        # normalize
        mean = np.trapz(new_meancont, x=x)
        new_meancont /= mean

        return new_meancont, mean

    def update_mean_cont(self, spectra_list, noupdate):
        """ Update the global mean continuum and stacked flux.

        Uses :attr:`forestivar_sm <qsonic.spectrum.Spectrum.forestivar_sm>`
        in inverse variance, but must be set beforehand.
        Raw mean continuum estimates are smoothed with a weighted
        :external+scipy:py:class:`scipy.interpolate.UnivariateSpline`. The mean
        continuum is removed from higher Legendre polynomials and normalized by
        the mean. This function updates
        :attr:`meancont_interp.fp <.meancont_interp>` if noupdate is False.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects to fit.
        noupdate: bool
            Does not update :attr:`meancont_interp.fp <.meancont_interp>` if
            True (last iteration).

        Returns
        ---------
        has_converged: bool
            True if all continuum updates on every point are less than 0.33
            times the error estimates.
        """
        norm_flux = np.zeros(self.nbins)
        std_flux = np.empty(self.nbins)
        counts = np.zeros(self.nbins)
        self.flux_stacker.reset()

        for spec in valid_spectra(spectra_list):
            for arm, wave_arm in spec.forestwave.items():
                wave_rf_arm = wave_arm / (1 + spec.z_qso)
                bin_idx = (
                    (wave_rf_arm - self.rfwave[0]) / self.dwrf + 0.5
                ).astype(int)

                cont = spec.cont_params['cont'][arm]
                flux = spec.forestflux[arm] / cont
                weight = spec.forestweight[arm] * cont**2
                # Deconvolve resolution matrix ?

                norm_flux += np.bincount(
                    bin_idx, weights=flux * weight, minlength=self.nbins)
                counts += np.bincount(
                    bin_idx, weights=weight, minlength=self.nbins)

                self.flux_stacker.add(wave_arm, flux, weight)

        self.flux_stacker.calculate()
        self.comm.Allreduce(MPI.IN_PLACE, norm_flux)
        self.comm.Allreduce(MPI.IN_PLACE, counts)
        w = counts > 0

        if w.sum() != self.nbins:
            warn_mpi(
                "Extrapolating empty bins in the mean continuum.",
                self.mpi_rank)

        norm_flux[w] /= counts[w]
        norm_flux[~w] = np.mean(norm_flux[w])
        std_flux[w] = 1 / np.sqrt(counts[w])
        std_flux[~w] = 10 * np.mean(std_flux[w])

        # Smooth new estimates
        spl = UnivariateSpline(self.rfwave, norm_flux, w=1 / std_flux)
        new_meancont = spl(self.rfwave)
        new_meancont *= self.meancont_interp.fp

        # remove tilt and higher orders and normalize
        new_meancont, mean_ = self._project_normalize_meancont(new_meancont)

        norm_flux = new_meancont / self.meancont_interp.fp - 1
        std_flux /= mean_

        if not noupdate:
            self.meancont_interp.reset(new_meancont, ep=std_flux)

        all_pt_test = np.all(np.abs(norm_flux) < 0.33 * std_flux)
        chi2_change = np.sum((norm_flux / std_flux)**2) / self.nbins
        has_converged = (chi2_change < 1e-3) | all_pt_test

        if self.mpi_rank != 0:
            return has_converged

        text = ("Continuum updates\n" "rfwave\t| update\t| error\n")

        sl = np.s_[::max(1, self.nbins // 10)]
        for w, n, e in zip(self.rfwave[sl], norm_flux[sl], std_flux[sl]):
            text += f"{w:7.2f}\t| {n:7.2e}\t| +- {e:7.2e}\n"

        text += f"Change in chi2: {chi2_change*100:.4e}%"
        logging_mpi(text, 0)

        return has_converged

    def update_var_lss_eta(self, spectra_list, noupdate):
        """ Fit and update var_lss and eta if enabled. See
        :class:`VarLSSFitter` for fitting details.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects to fit.
        noupdate: bool
            Does not update `self.varlss_interp.fp` if True (last iteration).
        """
        if self.varlss_fitter is None:
            return

        self.varlss_fitter.reset()

        for spec in valid_spectra(spectra_list):
            for arm, wave_arm in spec.forestwave.items():
                cont = spec.cont_params['cont'][arm]
                delta = spec.forestflux[arm] / cont - 1
                ivar = spec.forestivar[arm] * cont**2

                self.varlss_fitter.add(wave_arm, delta, ivar)

        # Else, fit for var_lss
        if self.fit_eta:
            text = "Fitting var_lss and eta"
            initial_guess = np.vstack(
                (self.varlss_interp.fp, self.eta_interp.fp)).T
        else:
            text = "Fitting var_lss"
            initial_guess = self.varlss_interp.fp

        logging_mpi(text, self.mpi_rank)
        y, ep = self.varlss_fitter.fit(initial_guess)

        if not noupdate and not self.fit_eta:
            self.varlss_interp.reset(y, ep=ep)
        if not noupdate and self.fit_eta:
            self.varlss_interp.reset(y[:, 0], ep=ep[:, 0])
            self.eta_interp.reset(y[:, 1], ep=ep[:, 1])

        if self.mpi_rank != 0:
            return

        step = max(1, self.varlss_fitter.nwbins // 10)
        text = ("------------------------------\n"
                "wave\t| var_lss +-  error \t|   eta   +-  error \n")

        for i in range(0, self.varlss_fitter.nwbins, step):
            w = self.varlss_fitter.waveobs[i]
            v = self.varlss_interp.fp[i]
            ve = self.varlss_interp.ep[i]
            n = self.eta_interp.fp[i]
            ne = self.eta_interp.ep[i]
            text += \
                f"{w:7.2f}\t| {v:7.2e} +- {ve:7.2e}\t| {n:7.2e} +- {ne:7.2e}\n"
        text += "------------------------------"
        logging_mpi(text, 0)

    def iterate(self, spectra_list):
        """Main function to fit continua and iterate.

        Consists of three major steps: initializing, fitting, updating global
        variables. The initialization sets ``cont_params`` variable of every
        Spectrum object. Continuum polynomial order is carried by setting
        ``cont_params[x]``. At each iteration:

        1. Global variables (mean continuum, var_lss) are saved to file
           (attributes.fits) file. This ensures the order of what is used in
           each iteration.
        2. All spectra are fit.
        3. Mean continuum is updated by stacking, smoothing and removing
           degenarate modes. Check for convergence if update is small.
        4. If fitting for var_lss, fit and update by calculating variance
           statistics.

        At the end of requested iterations or convergence, a chi2 catalog is
        created that includes information regarding chi2, mean_snr, targetid,
        etc.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects to fit.
        """
        has_converged = False

        for spec in spectra_list:
            spec.cont_params['method'] = 'picca'
            spec.cont_params['x'] = np.append(
                spec.cont_params['x'][0], np.zeros(self.cont_order))
            spec.cont_params['xcov'] = np.eye(self.cont_order + 1)
            spec.cont_params['dof'] = spec.get_real_size()

        fname = f"{self.outdir}/attributes.fits" if self.outdir else ""
        fattr = MPISaver(fname, self.mpi_rank)

        for it in range(self.niterations):
            is_last_it = it == self.niterations - 1
            logging_mpi(
                f"Fitting iteration {it+1}/{self.niterations}", self.mpi_rank)

            self.save(fattr, f"-{it + 1}")

            # Fit all continua one by one
            self.fit_continua(spectra_list)
            # Stack all spectra in each process
            # Broadcast and recalculate global functions
            has_converged = self.update_mean_cont(spectra_list, is_last_it)

            self.update_var_lss_eta(spectra_list, is_last_it)

            if has_converged:
                logging_mpi("Iteration has converged.", self.mpi_rank)
                break

        if not has_converged:
            warn_mpi("Iteration has NOT converged.", self.mpi_rank)

        self.save(fattr)
        self.varlss_fitter.write(fattr)
        fattr.close()
        logging_mpi("All continua are fit.", self.mpi_rank)

    def save(self, fattr, suff=''):
        """Save mean continuum and var_lss (if fitting) to a fits file.

        Arguments
        ---------
        fattr: MPISaver
            File handler to save only on master node.
        suff: str
            Suffix for the current iteration number.
        """
        fattr.write(
            [self.rfwave, self.meancont_interp.fp, self.meancont_interp.ep],
            names=['lambda_rf', 'mean_cont', 'e_mean_cont'],
            extname=f'CONT{suff}')

        fattr.write(
            [self.flux_stacker.waveobs, self.flux_stacker.stacked_flux],
            names=['lambda', 'stacked_flux'],
            extname=f'STACKED_FLUX{suff}')

        if self.varlss_fitter is None:
            return

        fattr.write(
            [self.varlss_fitter.waveobs,
             self.varlss_interp.fp, self.varlss_interp.ep,
             self.eta_interp.fp, self.eta_interp.ep],
            names=['lambda', 'var_lss', 'e_var_lss', 'eta', 'e_eta'],
            extname=f'VAR_FUNC{suff}')

    def save_contchi2_catalog(self, spectra_list):
        """Save chi2 catalog if ``self.outdir`` is set. All values are gathered
        and saved on the master node.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects to fit.
        """
        if not self.outdir:
            return

        logging_mpi("Saving continuum chi2 catalog.", self.mpi_rank)
        corder = self.cont_order + 1

        dtype = np.dtype([
            ('TARGETID', 'int64'), ('Z', 'f4'), ('HPXPIXEL', 'i8'),
            ('ARMS', 'U5'), ('MEANSNR', 'f4', 3), ('RSNR', 'f4'),
            ('MPI_RANK', 'i4'),
            ('CONT_valid', bool), ('CONT_chi2', 'f4'), ('CONT_dof', 'i4'),
            ('CONT_x', 'f4', corder),
            ('CONT_xcov', 'f4', corder**2)
        ])
        local_catalog = np.empty(len(spectra_list), dtype=dtype)

        for i, spec in enumerate(spectra_list):
            mean_snrs = np.zeros(3)
            _x = list(spec.mean_snr.values())
            mean_snrs[:len(_x)] = _x

            row = local_catalog[i]
            row['TARGETID'] = spec.targetid
            row['Z'] = spec.z_qso
            row['HPXPIXEL'] = spec.catrow['HPXPIXEL']
            row['ARMS'] = ",".join(spec.mean_snr.keys())
            row['MEANSNR'] = mean_snrs
            row['RSNR'] = spec.rsnr
            row['MPI_RANK'] = self.mpi_rank
            for lbl in ['valid', 'x', 'chi2', 'dof']:
                row[f'CONT_{lbl}'] = spec.cont_params[lbl]
            row['CONT_xcov'] = spec.cont_params['xcov'].ravel()

        all_catalogs = self.comm.gather(local_catalog)
        if self.mpi_rank == 0:
            all_catalog = np.concatenate(all_catalogs)
            fts = fitsio.FITS(
                f"{self.outdir}/continuum_chi2_catalog.fits", 'rw',
                clobber=True)
            fts.write(all_catalog, extname='CHI2_CAT')
            fts.close()


@njit("f8[:, :](i8[:], f8[:], f8[:], i8)")
def _fast_weighted_vector_bincount(x, delta, var, minlength):
    xvec = np.zeros((4, minlength), dtype=np.float_)
    y = delta**2
    y2 = y**2

    for i in range(x.size):
        xvec[0, x[i]] += delta[i]
        xvec[1, x[i]] += y[i]
        xvec[2, x[i]] += y2[i]
        xvec[3, x[i]] += var[i]

    return xvec


class VarLSSFitter():
    """ Variance fitter for the large-scale fluctuations.

    Input wavelengths and variances are the bin edges, so centers will be
    shifted. Valid bins require at least 500 pixels from 50 quasars. Assumes no
    spectra has `wave < w1obs` or `wave > w2obs`.

    .. note::

        This class is designed to be used in a linear fashion. You create it,
        add statistics to it and finally fit. After :meth:`fit` is called,
        :meth:`fit` and :meth:`add` **cannot** be called again. You may
        :meth:`reset` and start over.

    Usage::

        ...
        varfitter = VarLSSFitter(
            wave1, wave2, nwbins,
            var1, var2, nvarbins,
            nsubsamples=100, comm=comm)
        # Change static minimum numbers for valid statistics
        VarLSSFitter.min_no_pix = min_no_pix
        VarLSSFitter.min_no_qso = min_no_qso

        for delta in deltas_list:
            varfitter.add(delta.wave, delta.delta, delta.ivar)

        logging_mpi("Fitting variance for VarLSS and eta", mpi_rank)
        fit_results = np.ones((nwbins, 2))
        fit_results[:, 0] = 0.1
        fit_results, std_results = varfitter.fit(fit_results)

        varfitter.save("variance-file.fits")

        # You CANNOT call ``fit`` again!

    Parameters
    ----------
    w1obs: float
        Lower observed wavelength edge.
    w2obs: float
        Upper observed wavelength edge.
    nwbins: int, default: None
        Number of wavelength bins. If none, automatically calculated to yield
        120 A wavelength spacing.
    var1: float, default: 1e-4
        Lower variance edge.
    var2: float, default: 20
        Upper variance edge.
    nvarbins: int, default: 100
        Number of variance bins.
    nsubsamples: int, default: None
        Number of subsamples for the Jackknife covariance. Auto determined by
        number of bins (``nvarbins^2``) if None.
    use_cov: bool, default: False
        Use the Jackknife covariance when fitting.
    comm: MPI.COMM_WORLD or None, default: None
        MPI comm object to allreduce if enabled.

    Attributes
    ----------
    waveobs: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Wavelength centers in the observed frame.
    ivar_edges: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Inverse variance edges.
    minlength: int
        Minimum size of the combined bin count array. It includes underflow and
        overflow bins for both wavelength and variance bins.
    wvalid_bins: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Bool array slicer to get non-overflow bins of 1D arrays.
    subsampler: SubsampleCov
        Subsampler object that stores mean_delta in axis=0, var_delta in axis=1
        , var2_delta in axis=2, mean bin variance in axis=3.
    comm: MPI.COMM_WORLD or None, default: None
        MPI comm object to allreduce if enabled.
    mpi_rank: int
        Rank of the MPI process if ``comm!=None``. Zero otherwise.
    """
    min_no_pix = 500
    """int: Minimum number of pixels a bin must have to be valid."""
    min_no_qso = 50
    """int: Minimum number of quasars a bin must have to be valid."""

    @staticmethod
    def variance_function(var_pipe, var_lss, eta=1):
        """Variance model to be fit.

        .. math::

            \\sigma^2_\\mathrm{obs} = \\eta \\sigma^2_\\mathrm{pipe} +
            \\sigma^2_\\mathrm{LSS}

        Arguments
        ---------
        var_pipe: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Pipeline variance.
        var_lss: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Large-scale structure variance.
        eta: float
            Pipeline variance calibration scalar.

        Returns
        -------
        :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Expected variance of deltas.
        """
        return eta * var_pipe + var_lss

    def __init__(
            self, w1obs, w2obs, nwbins=None,
            var1=1e-4, var2=20., nvarbins=100,
            nsubsamples=None, use_cov=False, comm=None
    ):
        if nwbins is None:
            nwbins = int(round((w2obs - w1obs) / 120.))

        self.nwbins = nwbins
        self.nvarbins = nvarbins
        self.use_cov = use_cov
        self.comm = comm

        # Set up wavelength and inverse variance bins
        wave_edges, self.dwobs = np.linspace(
            w1obs, w2obs, nwbins + 1, retstep=True)
        self.waveobs = (wave_edges[1:] + wave_edges[:-1]) / 2
        self.ivar_edges = np.logspace(
            -np.log10(var2), -np.log10(var1), nvarbins + 1)

        # Set up arrays to store statistics
        self.minlength = (self.nvarbins + 2) * (self.nwbins + 2)
        # Bool array slicer for get non-overflow bins in 1D array
        self.wvalid_bins = np.zeros(self.minlength, dtype=bool)
        for iwave in range(self.nwbins):
            i1 = (iwave + 1) * (self.nvarbins + 2) + 1
            i2 = i1 + self.nvarbins
            wbinslice = np.s_[i1:i2]
            self.wvalid_bins[wbinslice] = True

        self._num_pixels = np.zeros(self.minlength, dtype=int)
        self._num_qso = np.zeros(self.minlength, dtype=int)

        # If ran with MPI, save mpi_rank first
        # Then shift each container to remove possibly over adding to 0th bin.
        if comm is not None:
            self.mpi_rank = comm.Get_rank()
            mpi_size = comm.Get_size()
        else:
            self.mpi_rank = 0
            mpi_size = 1

        if nsubsamples is None:
            nsubsamples = self.nvarbins**2
        istart = (self.mpi_rank * nsubsamples) // mpi_size
        # Axis 0 is mean
        # Axis 1 is var_delta
        # Axis 2 is var2_delta
        # Axis 3 is var_centers
        self.subsampler = SubsampleCov(
            (4, self.minlength), nsubsamples, istart)

    def reset(self):
        """Reset delta and num arrays to zero."""
        self.subsampler.reset()
        self._num_pixels *= 0
        self._num_qso *= 0

    def add(self, wave, delta, ivar):
        """Add statistics of a single spectrum. Updates delta and num arrays.

        Assumes no spectra has ``wave < w1obs`` or ``wave > w2obs``.

        Arguments
        ---------
        wave: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Wavelength array.
        delta: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Delta array.
        ivar: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Inverse variance array.
        """
        # add 1 to match searchsorted/bincount output/input
        wave_indx = ((wave - self.waveobs[0]) / self.dwobs + 1.5).astype(int)
        ivar_indx = np.searchsorted(self.ivar_edges, ivar)
        all_indx = ivar_indx + wave_indx * (self.nvarbins + 2)
        var = np.zeros_like(ivar)
        w = ivar > 0
        var[w] = 1. / ivar[w]

        npix = np.bincount(all_indx, minlength=self.minlength)
        self._num_pixels += npix
        xvec = _fast_weighted_vector_bincount(
            all_indx, delta, var, self.minlength)
        self.subsampler.add_measurement(xvec, npix)

        npix[npix > 0] = 1
        self._num_qso += npix

    def _allreduce(self):
        """ Sums statistics from all MPI process in place."""
        if self.comm is None:
            return

        self.subsampler.allreduce(self.comm, MPI.IN_PLACE)

        self.comm.Allreduce(MPI.IN_PLACE, self._num_pixels)
        self.comm.Allreduce(MPI.IN_PLACE, self._num_qso)

    def _calc_subsampler_stats(self):
        """ Calculates mean, variance and error on the variance.

        It also calculates the delete-one Jackknife variance over
        ``nsubsamples``.
        The variance on var_delta is updated with the variance on the mean
        estimates, then it is regularized by calculated var2_delta
        (Gaussian estimates), where if Jackknife variance is smaller than
        the Gaussian estimate, it is replaced by the Gaussian estimate.

        Covariance is calculated if :attr:`use_cov` is set to True in init.
        Covariance of mean_delta^2 is propagated using the formula in
        :func:`qsonic.mathtools.block_covariance_of_square`.
        """
        self.subsampler.get_mean_n_var()
        if self.use_cov:
            self.subsampler.get_mean_n_cov(
                indices=[0, 1], blockdim=self.nvarbins + 2)

        m2 = self.subsampler.mean[0]**2
        self.subsampler.mean[1] -= m2
        self.subsampler.mean[2] -= self.subsampler.mean[1]**2

        w = self._num_pixels > 0
        self.subsampler.mean[2, w] /= self._num_pixels[w]

        # Update variance / covariance
        if self.use_cov:
            self.subsampler.covariance[1] += block_covariance_of_square(
                self.subsampler.mean[0],
                self.subsampler.variance[0],
                self.subsampler.covariance[0])
        else:
            self.subsampler.variance[1] += 4 * m2 * self.subsampler.variance[0]
            self.subsampler.variance[1] += 2 * self.subsampler.variance[0]**2
            # Regularized jackknife errors
            self.subsampler.variance[1] = np.where(
                self.subsampler.variance[1] > self.subsampler.mean[2],
                self.subsampler.variance[1],
                self.subsampler.mean[2]
            )

    def _smooth_fit_results(self, fit_results, std_results):
        w = fit_results > 0

        # Smooth new estimates
        if fit_results.ndim == 1:
            spl = UnivariateSpline(
                self.waveobs[w], fit_results[w], w=1 / std_results[w])

            fit_results = spl(self.waveobs)
        # else ndim == 2
        else:
            w = w[:, 0]
            spl1 = UnivariateSpline(
                self.waveobs[w], fit_results[w, 0], w=1 / std_results[w, 0])
            spl2 = UnivariateSpline(
                self.waveobs[w], fit_results[w, 1], w=1 / std_results[w, 1])

            fit_results[:, 0] = spl1(self.waveobs)
            fit_results[:, 1] = spl2(self.waveobs)

        return fit_results

    def _fit_array_shape_assert(self, arr):
        assert (arr.ndim == 1 or arr.ndim == 2)
        assert (arr.shape[0] == self.nwbins)
        if arr.ndim == 2:
            assert (arr.shape[1] == 2)

    def fit(self, initial_guess, smooth=True):
        """ Syncronize all MPI processes and fit for ``var_lss`` and ``eta``.

        Second axis always contains ``eta`` values. Example::

            var_lss = initial_guess[:, 0]
            eta = initial_guess[:, 1]

        This implemented using :func:`scipy.optimize.curve_fit` with
        :attr:`e_var_delta` or :attr:`cov_var_delta` as absolute errors. Domain
        is bounded to ``(0, 2)``. These fits are then smoothed via
        :external+scipy:py:class:`scipy.interpolate.UnivariateSpline`
        using weights from ``curve_fit``,
        while missing values or failed wavelength bins are extrapolated.

        Arguments
        ---------
        initial_guess: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Initial guess for var_lss and eta. If 1D array, eta is fixed to
            one. If 2D, its shape must be ``(nwbins, 2)``.
        smooth: bool, default: True
            Smooth results using UnivariateSpline.

        Returns
        ---------
        fit_results: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Smoothed fit results at observed wavelengths where missing values
            are extrapolated. 1D array containing LSS variance if
            ``initial_guess`` is 1D. 2D containing eta values on the second
            axis if ``initial_guess`` is 2D ndarray.
        std_results: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Error on ``var_lss`` from sqrt of ``curve_fit`` output. Same
            behavior as ``fit_results``.
        """
        self._fit_array_shape_assert(initial_guess)
        self._allreduce()
        self._calc_subsampler_stats()

        nfails = 0
        fit_results = np.zeros_like(initial_guess)
        std_results = np.zeros_like(initial_guess)

        w_gtr_min = ((self.num_pixels > VarLSSFitter.min_no_pix)
                     & (self.num_qso > VarLSSFitter.min_no_qso))

        if self.use_cov:
            err_base = self.cov_var_delta
        else:
            err_base = self.e_var_delta

        for iwave in range(self.nwbins):
            i1 = iwave * self.nvarbins
            i2 = i1 + self.nvarbins
            wave_slice = np.s_[i1:i2]
            w = w_gtr_min[wave_slice]

            if w.sum() == 0:
                nfails += 1
                warn_mpi(
                    "Not enough statistics for VarLSSFitter at"
                    f" wave_obs: {self.waveobs[iwave]:.2f}.",
                    self.mpi_rank)
                continue

            if self.use_cov:
                err2use = err_base[iwave][:, w][w, :]
            else:
                err2use = err_base[wave_slice][w]

            try:
                pfit, pcov = curve_fit(
                    VarLSSFitter.variance_function,
                    self.var_centers[wave_slice][w],
                    self.var_delta[wave_slice][w],
                    p0=initial_guess[iwave],
                    sigma=err2use,
                    absolute_sigma=True,
                    check_finite=True,
                    bounds=(0, 2)
                )
            except Exception as e:
                nfails += 1
                warn_mpi(
                    "VarLSSFitter failed at wave_obs: "
                    f"{self.waveobs[iwave]:.2f}. "
                    f"Reason: {e}. Extrapolating.",
                    self.mpi_rank)
            else:
                fit_results[iwave] = pfit
                std_results[iwave] = np.sqrt(np.diag(pcov))

        invalid_ratio = nfails / fit_results.shape[0]
        if invalid_ratio > 0.4:
            raise QsonicException(
                "VarLSSFitter failed at more than 40% points.")

        if nfails > 0:
            warn_mpi(
                f"VarLSSFitter failed and extrapolated at {nfails} points.",
                self.mpi_rank)

        # Smooth new estimates
        if smooth:
            fit_results = self._smooth_fit_results(fit_results, std_results)

        return fit_results, std_results

    def write(self, mpi_saver, min_snr=0, max_snr=100):
        """ Write variance statistics to FITS file in 'VAR_STATS' extention.

        Arguments
        ---------
        mpi_saver: MPISaver
            MPI FITS file handler.
        min_snr: float, default: 0
            Minimum SNR in this sample to be written into header.
        max_snr: float, default: 100
            Maximum SNR in this sample to be written into header.
        """
        hdr_dict = {
            'MINNPIX': VarLSSFitter.min_no_pix,
            'MINNQSO': VarLSSFitter.min_no_qso,
            'MINSNR': min_snr,
            'MAXSNR': max_snr,
            'WAVE1': self.waveobs[0],
            'WAVE2': self.waveobs[-1],
            'NWBINS': self.nwbins,
            'IVAR1': self.ivar_edges[0],
            'IVAR2': self.ivar_edges[-1],
            'NVARBINS': self.nvarbins
        }

        data_to_write = [
            np.repeat(self.waveobs, self.nvarbins),
            self.var_centers, self.e_var_centers,
            self.var_delta, self.e_var_delta,
            self.mean_delta, self.var2_delta, self.num_pixels, self.num_qso]

        names = ['wave', 'var_pipe', 'e_var_pipe', 'var_delta', 'e_var_delta',
                 'mean_delta', 'var2_delta', 'num_pixels', 'num_qso']

        if self.use_cov:
            cov_data = self.cov_var_delta.reshape(
                self.nwbins * self.nvarbins, self.nvarbins)
            data_to_write.append(cov_data)
            names.append("cov_var_delta")

        mpi_saver.write(
            data_to_write, names=names, extname="VAR_STATS", header=hdr_dict)

    @property
    def num_pixels(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Number of pixels in bins."""
        return self._num_pixels[self.wvalid_bins]

    @property
    def num_qso(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Number of quasars in bins."""
        return self._num_qso[self.wvalid_bins]

    @property
    def mean_delta(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`: Mean delta."""
        return self.subsampler.mean[0][self.wvalid_bins]

    @property
    def var_delta(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Variance delta."""
        return self.subsampler.mean[1][self.wvalid_bins]

    @property
    def e_var_delta(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Error on variance delta using delete-one Jackknife. The error of
        :attr:`mean_delta` is propagated, then regularized using
        :attr:`var2_delta`. See source code of :meth:`_calc_subsampler_stats`.
        """
        return np.sqrt(self.subsampler.variance[1][self.wvalid_bins])

    @property
    def cov_var_delta(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        3D array of shape ``(nwbins, nvarbins, nvarbins)`` for the covariance
        on variance delta using delete-one Jackknife. The covariance
        of :attr:`mean_delta` is propagated. See
        :func:`qsonic.mathtools.block_covariance_of_square` for details.
        None if :attr:`use_cov` is False."""
        if not self.use_cov:
            return None
        cov = self.subsampler.covariance[1]
        return cov[1:-1, 1:-1, 1:-1]

    @property
    def var2_delta(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Variance delta^2."""
        return self.subsampler.mean[2][self.wvalid_bins]

    @property
    def var_centers(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Mean variance for the bin centers in **descending** order."""
        return self.subsampler.mean[3][self.wvalid_bins]

    @property
    def e_var_centers(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`:
        Error on the mean variance for the bin centers."""
        return np.sqrt(self.subsampler.variance[3][self.wvalid_bins])


class FluxStacker():
    """ The class to stack flux values to obtain IGM mean flux and other
    problems.

    This object can be called. Stacked flux is initialized to one. Reset before
    adding statistics.

    Parameters
    ----------
    w1obs: float
        Lower observed wavelength edge.
    w2obs: float
        Upper observed wavelength edge.
    dwobs: float
        Wavelength spacing.
    comm: MPI.COMM_WORLD or None, default: None
        MPI comm object to allreduce if enabled.

    Attributes
    ----------
    waveobs: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Wavelength centers in the observed frame.
    nwbins: int
        Number of wavelength bins
    dwobs: float
        Wavelength spacing. Usually same as observed grid.
    _interp: FastLinear1DInterp
        Interpolator. Saves stacked_flux in fp and weights in ep.
    comm: MPI.COMM_WORLD or None, default: None
        MPI comm object to allreduce if enabled.
    """

    def __init__(self, w1obs, w2obs, dwobs, comm=None):
        # Set up wavelength and inverse variance bins
        self.nwbins = int(round((w2obs - w1obs) / dwobs))
        wave_edges, self.dwobs = np.linspace(
            w1obs, w2obs, self.nwbins + 1, retstep=True)
        self.waveobs = (wave_edges[1:] + wave_edges[:-1]) / 2

        self.comm = comm

        self._interp = FastLinear1DInterp(
            self.waveobs[0], self.dwobs,
            np.ones(self.nwbins), ep=np.zeros(self.nwbins))

    def __call__(self, wave):
        return self._interp(wave)

    def add(self, wave, flux, weight):
        """ Add statistics of a single spectrum.

        Updates :attr:`stacked_flux` and :attr:`weights`. Assumes no spectra
        has ``wave < w1obs`` or ``wave > w2obs``.

        Arguments
        ---------
        wave: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Wavelength array.
        flux: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Flux array. Specifically f/C.
        weight: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Weight array.
        """
        wave_indx = ((wave - self.waveobs[0]) / self.dwobs + 0.5).astype(int)

        self._interp.fp += np.bincount(
            wave_indx, weights=flux * weight, minlength=self.nwbins)
        self._interp.ep += np.bincount(
            wave_indx, weights=weight, minlength=self.nwbins)

    def calculate(self):
        """Calculate stacked flux by allreducing if necessary."""
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, self._interp.fp)
            self.comm.Allreduce(MPI.IN_PLACE, self._interp.ep)

        w = self._interp.ep > 0
        self._interp.fp[w] /= self._interp.ep[w]
        self._interp.fp[~w] = 0

    def reset(self):
        """Reset :attr:`stacked_flux` and :attr:`weights` arrays to zero."""
        self._interp.fp *= 0
        self._interp.ep *= 0

    @property
    def stacked_flux(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`: Stacked flux."""
        return self._interp.fp

    @property
    def weights(self):
        """:external+numpy:py:class:`ndarray <numpy.ndarray>`: Weights."""
        return self._interp.ep
