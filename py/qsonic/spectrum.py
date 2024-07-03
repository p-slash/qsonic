import argparse

import numpy as np

from qsonic.mathtools import (
    _zero_function, _one_function, get_smooth_ivar, get_median_outlier_mask)


def add_wave_region_parser(parser=None):
    """ Adds wavelength analysis related arguments to parser. These
    arguments are grouped under 'Wavelength analysis region'. All of them
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

    wave_group = parser.add_argument_group('Wavelength analysis region')
    wave_group.add_argument(
        "--wave1", type=float, default=3600.,
        help="First observed wavelength edge.")
    wave_group.add_argument(
        "--wave2", type=float, default=6000.,
        help="Last observed wavelength edge.")
    wave_group.add_argument(
        "--forest-w1", type=float, default=1050.,
        help="First forest wavelength edge.")
    wave_group.add_argument(
        "--forest-w2", type=float, default=1180.,
        help="Last forest wavelength edge.")

    return parser


def generate_spectra_list_from_data(cat_by_survey, data):
    return [
        Spectrum.from_dictionary(catrow, data, idx)
        for idx, catrow in enumerate(cat_by_survey)
    ]


def valid_spectra(spectra_list):
    """Generator for continuum valid spectra."""
    return (spec for spec in spectra_list if spec.cont_params['valid'])


class Spectrum():
    """An object to represent one spectrum.

    Parameters
    ----------
    catrow: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog row.
    wave: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Dictionary of arrays specifying the wavelength grid. Static variable!
    flux: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Dictionary of arrays specifying the flux.
    ivar: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Dictionary of arrays specifying the inverse variance.
    mask: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Dictionary of arrays specifying the bitmask. Not stored
    reso: dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Dictionary of 2D arrays specifying the resolution matrix.
    idx: int
        Index to access in flux, ivar, mask and reso that corresponds to the
        quasar in `catrow`.

    Attributes
    ----------
    rsnr: float
        Average SNR above Lya. Calculated in :meth:`set_forest_region`.
    mean_snr: dict(float)
        Mean signal-to-noise ratio in the forest.
    _f1, _f2: dict(int)
        Forest indices. Set up using :meth:`set_forest_region` method. Then use
        property functions to access forest wave, flux, ivar instead.
    cont_params: dict
        Continuum parameters. Initial estimates are constructed.
    """
    WAVE_LYA_A = 1215.67
    """float: Lya wavelength in A."""
    _wave = None
    """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Common
    wavelength grid for **all** Spectra."""
    _coadd_wave = None
    """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Common
    **coadded** wavelength grid for **all** Spectra."""
    _dwave = None
    """float: Wavelength spacing."""
    _blinding = None
    """str or None: Blinding. Must be set for certain data."""
    _fits_colnames = ['LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT']
    """list(str): Column names to save in delta files."""

    @staticmethod
    def _set_wave(wave, check_consistency=False):
        """Set the common wavelength grid.

        Arguments
        ---------
        check_consistency: bool
            Asserts each time key and values are the same if True.
        """
        if not Spectrum._wave:
            Spectrum._wave = wave.copy()
            arm = list(wave.keys())[0]
            wave_arm = wave[arm]
            Spectrum._dwave = wave_arm[1] - wave_arm[0]
        elif check_consistency:
            for arm, wave_arm in Spectrum._wave.items():
                assert (arm in wave.keys())
                assert (np.allclose(Spectrum._wave[arm], wave_arm))

    @staticmethod
    def _set_coadd_wave():
        if Spectrum._coadd_wave:
            return

        min_wave = np.min([wave[0] for wave in Spectrum._wave.values()])
        max_wave = np.max([wave[-1] for wave in Spectrum._wave.values()])

        # Z arm in mocks are shifted by 0.4 A, so we need these extra steps
        # to make sure dwave is preseved.
        nwaves = round((max_wave - min_wave) / Spectrum._dwave + 0.1)
        coadd_wave = np.linspace(
            min_wave, min_wave + nwaves * Spectrum._dwave, nwaves + 1)
        Spectrum._coadd_wave = {'brz': coadd_wave}

    @staticmethod
    def set_blinding(maxlastnight, args):
        """Set the blinding strategy.

        'LASTNIGHT' column, args.mock_analysis, args.forest_w1 decide the
        blinding strategy. Mock, side band and SV analyses are not blinded.

        Arguments
        ---------
        maxlastnight: int
            Maximum LASTNIGHT column of the entire quasar catalog.
        args: argparse.Namespace
            Should have ``mock_analysis (bool)`` and ``forest_w1 (floar)``.
        """
        # do not blind mocks or metal forests
        if args.mock_analysis or args.forest_w1 > Spectrum.WAVE_LYA_A:
            Spectrum._blinding = "none"
        # sv data, no blinding
        elif maxlastnight < 20210514:
            Spectrum._blinding = "none"
        elif maxlastnight < 20210801:
            Spectrum._blinding = "desi_m2"
        elif maxlastnight < 20220801:
            Spectrum._blinding = "desi_y1"
        else:
            Spectrum._blinding = "desi_y3"

        if Spectrum._blinding != "none":
            Spectrum._fits_colnames[1] = 'DELTA_BLIND'

        if not args.skip_resomat:
            Spectrum._fits_colnames.append('RESOMAT')

    @staticmethod
    def blinding_not_set():
        """bool: ``True`` if blinding is not set."""
        return Spectrum._blinding is None

    @classmethod
    def from_dictionary(cls, catrow, data, idx):
        """Create a Spectrum from dictionary. See :class:`Spectrum` for
        argument details.

        If ``cont`` key is present in ``data``, :attr:`cont_params` dictionary
        gains the following::

            cont_params['true_data_w1'] (float): First wavelength
            cont_params['true_data_dwave'] (float): Wavelength spacing
            cont_params['true_data'] (ndarray): True continuum

        Returns
        -------
        Spectrum
        """
        spec = cls(catrow, data['wave'], data['flux'], data['ivar'],
                   data['mask'], data['reso'], idx)

        if "cont" in data.keys():
            spec.cont_params['true_data_w1'] = data['cont']['w1']
            spec.cont_params['true_data_dwave'] = data['cont']['dwave']
            spec.cont_params['true_data'] = data['cont']['data'][idx]

        return spec

    def __init__(self, catrow, wave, flux, ivar, mask, reso, idx):
        self.catrow = catrow
        Spectrum._set_wave(wave)

        self._current_wave = Spectrum._wave
        self.flux = {}
        self.ivar = {}
        self.reso = {}

        self.rsnr = None
        self.mean_snr = None
        self._f1 = {}
        self._f2 = {}
        self._forestwave = {}
        self._forestflux = {}
        self._forestivar = {}
        self._forestivar_sm = {}
        self._forestreso = {}
        self._forestweight = {}

        self._smoothing_scale = 0

        for arm, wave_arm in self.wave.items():
            self.flux[arm] = flux[arm][idx].copy()
            self.ivar[arm] = ivar[arm][idx].copy()
            w = (mask[arm][idx] != 0) | np.isnan(self.flux[arm]) \
                | np.isnan(self.ivar[arm]) | (self.ivar[arm] < 0)
            self.flux[arm][w] = 0
            self.ivar[arm][w] = 0

            if not reso:
                continue
            elif reso[arm].ndim == 2:
                self.reso[arm] = reso[arm].copy()
            else:
                self.reso[arm] = reso[arm][idx].copy()

        self.cont_params = {}
        self.cont_params['method'] = ''
        self.cont_params['valid'] = False
        self.cont_params['x'] = np.array([1., 0.])
        self.cont_params['xcov'] = np.eye(2)
        self.cont_params['chi2'] = -1.
        self.cont_params['dof'] = 0
        self.cont_params['cont'] = {}

        self._set_rsnr()

    def _set_rsnr(self):
        """Calculates and sets SNR above Lya."""
        self.rsnr = 0
        rsnr_weight = 1e-6

        for arm, wave_arm in self.wave.items():
            # Calculate SNR above Lya
            ii1 = np.searchsorted(
                wave_arm, (1 + self.z_qso) * Spectrum.WAVE_LYA_A)
            weight = np.sqrt(self.ivar[arm][ii1:])
            self.rsnr += np.dot(self.flux[arm][ii1:], weight)
            rsnr_weight += np.sum(weight > 0)

        self.rsnr /= rsnr_weight

    def _set_forest_related_parameters(self):
        """Calculates the mean SNR in the forest region and an initial guess
        for the continuum amplitude."""
        self.cont_params['x'][0] = 0
        cont_params_weight = 1e-6

        self.mean_snr = {}

        for arm, ivar_arm in self.forestivar.items():
            flux_arm = self.forestflux[arm]
            w = flux_arm > 0

            self.cont_params['x'][0] += np.dot(flux_arm[w], ivar_arm[w])
            cont_params_weight += np.sum(ivar_arm[w])

            self.mean_snr[arm] = 0
            armpix = np.sum(ivar_arm > 0)
            if armpix == 0:
                continue

            self.mean_snr[arm] = np.dot(np.sqrt(ivar_arm), flux_arm) / armpix

        self.cont_params['x'][0] /= cont_params_weight

    def slice(self, arm, i1, i2):
        if i1 == 0 and i2 == self._forestwave[arm].size:
            return

        self._forestwave[arm] = self._forestwave[arm][i1:i2]
        self._forestflux[arm] = self._forestflux[arm][i1:i2]
        self._forestivar[arm] = self._forestivar[arm][i1:i2]
        self._forestivar_sm[arm] = self._forestivar_sm[arm][i1:i2]
        self._forestweight[arm] = self._forestweight[arm][i1:i2]

        if arm in self._forestreso:
            self._forestreso[arm] = self._forestreso[arm][:, i1:i2]

        if arm in self.cont_params['cont']:
            self.cont_params['cont'][arm] = \
                self.cont_params['cont'][arm][i1:i2]

    def set_forest_region(self, w1, w2, lya1, lya2):
        """ Sets slices for the forest region. Masks outliers in each arm
        separately based on moving median statistics
        (see :func:`qsonic.mathtools.get_median_outlier_mask`). Also calculates
        the mean SNR in the forest and an initial guess for the continuum
        amplitude.

        Arguments
        ---------
        w1, w2: float
            Observed wavelength range
        lya1, lya2: float
            Rest-frame wavelength for the forest
        """
        l1 = max(w1, (1 + self.z_qso) * lya1)
        l2 = min(w2, (1 + self.z_qso) * lya2)

        for arm, wave_arm in self.wave.items():
            # Slice to forest limits
            ii1, ii2 = np.searchsorted(wave_arm, [l1, l2])
            real_size_arm = np.sum(self.ivar[arm][ii1:ii2] > 0)
            if real_size_arm == 0:
                continue

            self._f1[arm], self._f2[arm] = ii1, ii2

            # See https://numpy.org/doc/stable/user/basics.copies.html
            # Slicing creates views, not copies. Bases of views are not removed
            # from memory!
            self._forestwave[arm] = wave_arm[ii1:ii2].copy()
            self._forestflux[arm] = self.flux[arm][ii1:ii2].copy()
            self._forestivar[arm] = self.ivar[arm][ii1:ii2].copy()
            if self.reso:
                self._forestreso[arm] = self.reso[arm][:, ii1:ii2].copy()

            w = get_median_outlier_mask(
                self._forestflux[arm], self._forestivar[arm])
            self._forestflux[arm][w] = 0
            self._forestivar[arm][w] = 0

        self._forestivar_sm = self._forestivar
        self._forestweight = self._forestivar
        self._set_forest_related_parameters()

    def drop_arm(self, arm):
        self._forestwave.pop(arm, None)
        self._forestflux.pop(arm, None)
        self._forestivar.pop(arm, None)
        self._forestreso.pop(arm, None)
        self._forestivar_sm.pop(arm, None)
        self._forestweight.pop(arm, None)
        self.cont_params['cont'].pop(arm, None)

    def drop_short_arms(self, lya1=0, lya2=0, skip_ratio=0):
        """Arms that have less than ``skip_ratio`` pixels are removed from
        forest dictionary.

        Arguments
        ---------
        lya1, lya2: float
            Rest-frame wavelength for the forest
        skip_ratio: float
            Remove arms if they have less than this ratio of pixels
        """
        npixels_expected = (1 + self.z_qso) * (lya2 - lya1) / self.dwave
        npixels_expected = int(skip_ratio * npixels_expected) + 1
        short_arms = [arm for arm, ivar in self.forestivar.items()
                      if np.sum(ivar > 0) < npixels_expected]
        for arm in short_arms:
            self.drop_arm(arm)

    def remove_nonforest_pixels(self):
        """ Remove non-forest pixels from storage.

        This sets :attr:`flux`, :attr:`ivar` and :attr:`reso` to empty
        dictionary, but :attr:`wave` is not modified,
        since it is a static variable. Good practive is to loop using, e.g.,
        ``for arm, wave_arm in self.forestwave.items():``.
        """
        self._current_wave = {}
        self.flux = {}
        self.ivar = {}
        self.reso = {}

    def get_real_size(self):
        """
        Returns
        -------
        int: Sum of number of pixels with ``forestivar > 0`` for all arms.
        """
        size = 0
        for ivar_arm in self.forestivar.values():
            size += np.sum(ivar_arm > 0)

        return size

    def get_effective_meansnr(self):
        """ Calculate a weighted average of :attr:`mean_snr` over arms. Only
        call if :meth:`set_forest_region` has been called.

        .. math::
            \\langle\\mathrm{SNR}\\rangle = \\sum_{i} \\mathrm{SNR}_i^3 \\bigg/
            {\\sum_{i} \\mathrm{SNR}_i^2}

        Returns
        -------
        float: Effective mean SNR.
        """
        msnr_eff = 0
        norm = 1e-8

        for val in self.mean_snr.values():
            msnr_eff += val**3
            norm += val**2

        return msnr_eff / norm

    def is_long(self, dforest_wave, skip_ratio):
        """Determine if spectrum is long enough to be accepted.

        The condition is :meth:`get_real_size` > ``skip_ratio * npixels``,
        where ``npixels`` :math:`=(1 + z_\\mathrm{qso}) \\times`
        ``dforest_wave`` :math:`/ \\mathrm{d}\\lambda` and
        :math:`\\mathrm{d}\\lambda` is wavelength spacing in the observed frame
        in A.

        Arguments
        ---------
        dforest_wave: float
            Length of the forest in the rest-frame in A.
        skip_ratio: float
            Minimum ratio that needs to be present and unmasked to keep the
            spectrum.

        Returns
        -------
        bool
        """
        npixels = (1 + self.z_qso) * dforest_wave / self.dwave
        return self.get_real_size() > skip_ratio * npixels

    def set_smooth_forestivar(self, smoothing_size=16.):
        """ Set :attr:`forestivar_sm` to smoothed inverse variance. Before this
        call :attr:`forestivar_sm` points to :attr:`forestivar`. If
        ``smoothing_size <= 0``, smoothing is undone such that ivar_sm
        points to ivar. Also, :attr:`forestweight` points to this.

        ``smoothing_size`` is saved to a private :attr:`_smoothing_scale`
        variable for future use.

        Arguments
        ---------
        smoothing_size: float, default: 16
            Gaussian smoothing spread in A.
        """
        if smoothing_size <= 0:
            self._smoothing_scale = 0
            self._forestivar_sm = self._forestivar
        else:
            self._smoothing_scale = smoothing_size
            sigma_pix = smoothing_size / self.dwave
            self._forestivar_sm = {
                arm: get_smooth_ivar(ivar_arm, sigma_pix)
                for arm, ivar_arm in self.forestivar.items()
            }

        self._forestweight = self._forestivar_sm

    def set_forest_weight(
            self, varlss_interp=_zero_function, eta_interp=_one_function
    ):
        """ Sets :attr:`forestweight` for a given var_lss and eta correction.
        Always uses :attr:`forestivar_sm`, which is not actually smoothed if
        :meth:`set_smooth_forestivar` is not called.

        .. math::

            w = i / (\\eta + i \\sigma^2_\\mathrm{LSS} C^2),

        where i is IVAR and C is the continuum.

        Arguments
        ---------
        varlss_interp: Callable[[ndarray], ndarray], default: 0
            LSS variance interpolator.
        eta_interp: Callable[[ndarray], ndarray], default: 1
            eta interpolator.
        """
        if not self.cont_params['valid'] or not self.cont_params['cont']:
            self._forestweight = self._forestivar_sm
            return

        self._forestweight = {}
        for arm, wave_arm in self.forestwave.items():
            cont_est = self.cont_params['cont'][arm]
            var_lss = varlss_interp(wave_arm) * cont_est**2
            eta = eta_interp(wave_arm)
            ivar_arm = self.forestivar_sm[arm]
            self._forestweight[arm] = ivar_arm / (eta + ivar_arm * var_lss)

    def calc_continuum_chi2(self):
        """ Calculate the chi2 of the continuum fitting. This is just a sum
        of weight * (flux - cont)^2.

        Returns
        -------
        chi2: float
        """
        if not self.cont_params['valid'] or not self.cont_params['cont']:
            self.cont_params['chi2'] = -1
            return -1

        chi2 = 0
        for arm, wave_arm in self.forestwave.items():
            cont_est = self.cont_params['cont'][arm]
            weight = self.forestweight[arm]
            flux = self.forestflux[arm]
            chi2 += np.dot(weight, (flux - cont_est)**2)

        self.cont_params['chi2'] = chi2

        return chi2

    def simple_coadd(self):
        """Coadding without continuum and var_lss terms on the full spectrum.
        Weights, forests etc. will not be set. Replaces :attr:`wave`,
        :attr:`flux`, :attr:`ivar` and :attr:`reso` attributes with
        dictionaries that has a single arm ``brz`` as key to access the coadded
        data.

        We first set the static coadded wavelength grid if it is not set.
        Then we add each arm using :attr:`ivar` (not smoothed). If :attr:`reso`
        is set, we coadd the resolution matrix using the same inverse variance
        weights. If these weights are zero for both arms, resolution matrix
        coadding reverts to equal weights. Final private assignments are done
        at the end to keep using arm by arm values of :attr:`ivar`.
        """
        Spectrum._set_coadd_wave()

        min_wave = Spectrum._coadd_wave['brz'][0]
        nwaves = Spectrum._coadd_wave['brz'].size

        coadd_flux = np.zeros(nwaves)
        coadd_ivar = np.zeros(nwaves)
        coadd_norm = np.zeros(nwaves)

        idxes = {}
        for arm, wave_arm in self.wave.items():
            idx = ((wave_arm - min_wave) / self.dwave + 0.1).astype(int)
            idxes[arm] = idx

            weight = self.ivar[arm]

            var = np.zeros_like(weight)
            w = self.ivar[arm] > 0
            var[w] = 1 / self.ivar[arm][w]

            coadd_flux[idx] += weight * self.flux[arm]
            coadd_ivar[idx] += weight**2 * var
            coadd_norm[idx] += weight

        w = coadd_norm > 0
        coadd_flux[w] /= coadd_norm[w]
        coadd_ivar[w] = coadd_norm[w]**2 / coadd_ivar[w]

        if self.reso:
            max_ndia = np.max([reso.shape[0] for reso in self.reso.values()])
            coadd_reso = np.zeros((max_ndia, nwaves))
            coadd_norm *= 0

            for arm, reso_arm in self.reso.items():
                weight = self.ivar[arm].copy()
                weight[weight == 0] = 1e-8

                ddia = max_ndia - reso_arm.shape[0]
                # Assumption ddia cannot be odd
                ddia = ddia // 2
                if ddia > 0:
                    reso_arm = np.pad(reso_arm, ((ddia, ddia), (0, 0)))

                coadd_reso[:, idxes[arm]] += weight * reso_arm
                coadd_norm[idxes[arm]] += weight

            coadd_reso /= coadd_norm
            self.reso = {'brz': coadd_reso}

        self._current_wave = Spectrum._coadd_wave
        self.flux = {'brz': coadd_flux}
        self.ivar = {'brz': coadd_ivar}

    def coadd_arms_forest(
            self, varlss_interp=_zero_function, eta_interp=_one_function
    ):
        """ Coadds different arms using :attr:`forestweight`. Interpolators are
        needed to reset :attr:`forestweight`.

        Replaces ``forest`` variables and ``cont_params['cont']`` with a
        dictionary that has a single arm ``brz`` as key to access coadded data.

        Arguments
        ---------
        varlss_interp: Callable[[ndarray], ndarray], default: 0
            LSS variance interpolator or function.
        eta_interp: Callable[[ndarray], ndarray], default: 1
            eta interpolator or function.
        """
        min_wave = np.min([wave[0] for wave in self.forestwave.values()])
        max_wave = np.max([wave[-1] for wave in self.forestwave.values()])

        nwaves = int((max_wave - min_wave) / self.dwave + 0.1) + 1
        coadd_wave = np.linspace(min_wave, max_wave, nwaves)
        coadd_flux = np.zeros(nwaves)
        coadd_ivar = np.zeros(nwaves)
        coadd_norm = np.zeros(nwaves)

        idxes = {}
        for arm, wave_arm in self.forestwave.items():
            idx = ((wave_arm - min_wave) / self.dwave + 0.1).astype(int)
            idxes[arm] = idx

            weight = self.forestweight[arm]

            var = np.zeros_like(weight)
            w = self.forestivar[arm] > 0
            var[w] = 1 / self.forestivar[arm][w]

            coadd_flux[idx] += weight * self.forestflux[arm]
            coadd_ivar[idx] += weight**2 * var
            coadd_norm[idx] += weight

        w = coadd_norm > 0
        coadd_flux[w] /= coadd_norm[w]
        coadd_ivar[w] = coadd_norm[w]**2 / coadd_ivar[w]

        self._forestwave = {'brz': coadd_wave}
        self._forestflux = {'brz': coadd_flux}
        self._forestivar = {'brz': coadd_ivar}

        if self.cont_params['cont']:
            coadd_cont = np.empty(nwaves)

            for arm, idx in idxes.items():
                # continuum needs not weighting
                coadd_cont[idx] = self.cont_params['cont'][arm]

            self.cont_params['cont'] = {'brz': coadd_cont}

        if self.forestreso:
            max_ndia = np.max(
                [reso.shape[0] for reso in self.forestreso.values()])
            coadd_reso = np.zeros((max_ndia, nwaves))
            coadd_norm *= 0

            for arm, reso_arm in self.forestreso.items():
                weight = self.forestweight[arm].copy()
                weight[weight == 0] = 1e-8

                ddia = max_ndia - reso_arm.shape[0]
                # Assumption ddia cannot be odd
                ddia = ddia // 2
                if ddia > 0:
                    reso_arm = np.pad(reso_arm, ((ddia, ddia), (0, 0)))

                coadd_reso[:, idxes[arm]] += weight * reso_arm
                coadd_norm[idxes[arm]] += weight

            coadd_reso /= coadd_norm
            self._forestreso = {'brz': coadd_reso}

        self.set_smooth_forestivar(self._smoothing_scale)
        self.set_forest_weight(varlss_interp, eta_interp)

        mean_snr = np.dot(
            np.sqrt(coadd_ivar), coadd_flux) / np.sum(coadd_ivar > 0)
        self.mean_snr = {'brz': mean_snr}

    def mean_resolution(self, arm, weight=None):
        """ Returns the weighted mean Gaussian sigma of the spectrograph
        resolution in the forest.

        Arguments
        ---------
        arm: str
            Arm.
        weight: None or ndarray, default: None
            Weights. If ``None``, :attr:`forestweight` is used.

        Returns
        -------
        mean_reso: float or None
            Gaussian sigma. None if forestreso is not set.
        """
        if not self.forestreso:
            return None

        if weight is None:
            weight = self.forestweight[arm]

        total_weight = np.sum(weight)
        reso = np.dot(self.forestreso[arm], weight) / total_weight
        lambda_eff = np.dot(self.forestwave[arm], weight) / total_weight

        central_idx = reso.argmax()
        off_idx = np.array([-2, -1, 1, 2], dtype=int)
        ratios = reso[central_idx] / reso[central_idx + off_idx]
        ratios = np.log(ratios)
        w2 = ratios > 0
        norm = np.sum(w2)
        new_ratios = np.zeros_like(ratios)
        new_ratios[w2] = 1. / np.sqrt(ratios[w2])

        rms_in_pixel = np.abs(off_idx).dot(new_ratios) / np.sqrt(2.) / norm
        return rms_in_pixel * 3e5 * self.dwave / lambda_eff

    def write(self, fts_file):
        """Writes each arm to FITS file separately.

        Writes 'LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT' columns and
        'RESOMAT' column if resolution matrix is present to extension name
        ``targetid-arm``. FITS file must be initialized before.
        Each arm has its own `MEANSNR`.

        Arguments
        ---------
        fts_file: FITS file
            The file handler, not filename.
        """
        hdr_dict = {
            'LOS_ID': self.targetid,
            'TARGETID': self.targetid,
            'RA': np.radians(self.ra),
            'DEC': np.radians(self.dec),
            'Z': self.z_qso,
            'BLINDING': Spectrum._blinding,
            'WAVE_SOLUTION': "lin",
            'MEANSNR': 0.,
            'RSNR': self.rsnr,
            'DELTA_LAMBDA': self.dwave,
            'SMSCALE': self._smoothing_scale
        }

        for key in set(
                ['TILEID', 'FIBER', 'PETAL_LOC', 'EXPID', 'NIGHT']
        ).intersection(self.catrow.dtype.names):
            hdr_dict[key] = self.catrow[key]

        if 'EXPID' in self.catrow.dtype.names:
            expid = f"-{self.catrow['EXPID']}"
        else:
            expid = ""

        for arm, wave_arm in self.forestwave.items():
            if self.mean_snr[arm] == 0:
                continue

            hdr_dict['MEANSNR'] = self.mean_snr[arm]

            cont_est = self.cont_params['cont'][arm]
            delta = self.forestflux[arm] / cont_est - 1
            ivar = self.forestivar[arm] * cont_est**2
            weight = self.forestweight[arm] * cont_est**2
            delta[ivar == 0] = 0

            cols = [wave_arm, delta, ivar, weight, cont_est]
            if self.forestreso:
                hdr_dict['MEANRESO'] = self.mean_resolution(arm)
                cols.append(self.forestreso[arm].T.astype('f8'))

            fts_file.write(
                cols, names=Spectrum._fits_colnames, header=hdr_dict,
                extname=f"{self.targetid}-{arm}{expid}")

    @property
    def z_qso(self):
        """float: Quasar redshift."""
        return self.catrow['Z']

    @property
    def targetid(self):
        """int: Unique TARGETID identifier."""
        return self.catrow['TARGETID']

    @property
    def hpix(self):
        """int: Healpix."""
        return self.catrow['HPXPIXEL']

    @property
    def ra(self):
        """float: Right ascension."""
        return self.catrow['RA']

    @property
    def dec(self):
        """float: Declination"""
        return self.catrow['DEC']

    @property
    def wave(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Original
        wavelength grid in A."""
        return self._current_wave

    @property
    def dwave(self):
        """float: Wavelength step size in A."""
        return Spectrum._dwave

    @property
    def forestwave(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Forest
        wavelength field in A."""
        return self._forestwave

    @property
    def forestflux(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Forest
        flux field."""
        return self._forestflux

    @property
    def forestivar(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Forest
        inverse variance field."""
        return self._forestivar

    @property
    def forestivar_sm(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Forest
        smoothed inverse variance field.

        Initially equal to :attr:`.forestivar`. Smoothed if
        :meth:`.set_smooth_forestivar` is called."""
        return self._forestivar_sm

    @property
    def forestweight(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Forest
        weight field. Initially equal to :attr:`.forestivar`."""
        return self._forestweight

    @property
    def forestreso(self):
        """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Resolution
        matrix in the forest."""
        return self._forestreso


class Delta():
    """An object to read one delta from HDU.

    Parameters
    ----------
    hdu: fitsio.TableHDU
        Table containing delta data.

    Raises
    ---------
    RuntimeError
        If ``hdu`` doesn't have neither "LAMBDA" nor "LOGLAM" columns.
    RuntimeError
        If ``hdu`` doesn't have neither "DELTA" nor "DELTA_BLIND" columns.
    RuntimeError
        If ``hdu`` doesn't have none of "MOCKID", "TARGETID" and "THING_ID"
        header keys.

    Attributes
    ----------
    wave: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Wavelength array in A.
    delta: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Deltas.
    ivar: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Inverse variance.
    weight: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Weights, which includes var_lss.
    cont: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Continuum.
    reso: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Resolution matrix. ``None`` if not present.
    header: FITS header
        Header.
    targetid: int
        TARGETID, MOCKID or THING_ID from header.
    mean_snr: float
        MEANSNR from header.
    """
    _accepted_wave_columns = set(["LAMBDA", "LOGLAM"])
    """set: Supported column names for wavelength."""
    _accepted_delta_columns = set(['DELTA', 'DELTA_BLIND'])
    """set: Supported column names for delta."""
    _accepted_targetid_keys = set(['MOCKID', 'TARGETID', 'THING_ID'])
    """set: Supported header keys for unique ID."""
    _accepted_colums_map = {
        "wave": _accepted_wave_columns,
        "delta": _accepted_delta_columns
    }

    @staticmethod
    def _check_hdu(colnames, attr):
        req_map = Delta._accepted_colums_map[attr]
        key = req_map.intersection(colnames)
        if not key:
            raise RuntimeError(
                "One of these must be present in delta files: "
                f"{', '.join(req_map)} for {attr}!")

        return key.pop()

    def __init__(self, hdu):
        self.header = hdu.read_header()
        key = Delta._accepted_targetid_keys.intersection(self.header.keys())
        if not key:
            raise RuntimeError(
                "One of these must be present in delta file header: "
                f"{', '.join(Delta._accepted_targetid_keys)} for TARGETID!")

        key = key.pop()
        self.targetid = self.header[key]
        self.mean_snr = self.header['MEANSNR']

        colnames = hdu.get_colnames()
        data = hdu.read()

        key = Delta._check_hdu(colnames, "wave")
        if key == "LOGLAM":
            self.wave = 10**data['LOGLAM']
        else:
            self.wave = data[key].astype("f8")

        key = Delta._check_hdu(colnames, "delta")
        self._is_blinded = key == "DELTA_BLIND"
        self.delta = data[key].astype("f8")
        self.ivar = data['IVAR'].astype("f8")
        self.weight = data['WEIGHT'].astype("f8")
        self.cont = data['CONT'].astype("f8")

        self.delta[self.ivar == 0] = 0

        if 'RESOMAT' in colnames:
            self.reso = data['RESOMAT'].T.astype("f8")
        else:
            self.reso = None

    def write(self, fts_file):
        """Writes to FITS file. This function is aimed at saving coadded
        deltas.

        Writes 'LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT' columns and
        'RESOMAT' column if resolution matrix is present to extension name
        ``targetid``. FITS file must be initialized before. Note that ``arm``
        is lost in the extension name.

        Arguments
        ---------
        fts_file: FITS file
            The file handler, not filename.
        """
        hdr_dict = self.header

        cols = [self.wave, self.delta, self.ivar, self.weight, self.cont]
        names = ['LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT']

        if self._is_blinded:
            names[1] = 'DELTA_BLIND'

        if self.reso is not None:
            cols.append(self.reso.T)
            names.append('RESOMAT')

        fts_file.write(
            cols, names=names, header=hdr_dict,
            extname=f"{self.targetid}")

    def _coadd_reso(self, other, nwaves, idxes):
        max_ndia = max(self.reso.shape[0], other.reso.shape[0])
        coadd_reso = np.zeros((max_ndia, nwaves))
        coadd_norm = np.zeros(nwaves)

        for j, obj in enumerate([self, other]):
            weight = obj.weight.copy()
            weight[weight == 0] = 1e-8

            ddia = max_ndia - obj.reso.shape[0]
            # Assumption ddia cannot be odd
            ddia = ddia // 2
            if ddia > 0:
                obj.reso = np.pad(obj.reso, ((ddia, ddia), (0, 0)))

            coadd_reso[:, idxes[j]] += weight * obj.reso
            coadd_norm[idxes[j]] += weight

        coadd_reso /= coadd_norm
        self.reso = coadd_reso

    def coadd(self, other, dwave=0.8):
        min_wave = min(self.wave[0], other.wave[0])
        max_wave = max(self.wave[-1], other.wave[-1])

        nwaves = int((max_wave - min_wave) / dwave + 0.1) + 1
        coadd_wave = np.linspace(min_wave, max_wave, nwaves)
        coadd_delta = np.zeros(nwaves)
        coadd_ivar = np.zeros(nwaves)
        coadd_lss = np.zeros(nwaves)
        coadd_cont = np.zeros(nwaves)
        coadd_norm = np.zeros(nwaves)

        idxes = [None, None]
        for j, obj in enumerate([self, other]):
            i0 = ((obj.wave[0] - min_wave) / dwave + 0.1).astype(int)
            idx = np.s_[i0:i0 + obj.wave.size]
            idxes[j] = idx

            var = np.zeros_like(obj.weight)
            w = (obj.weight > 0) & (obj.ivar > 0)
            var[w] = 1 / obj.ivar[w]

            coadd_delta[idx] += obj.weight * obj.delta
            coadd_cont[idx] += obj.weight * obj.cont
            coadd_ivar[idx] += obj.weight**2 * var
            coadd_lss[idx] += 1 - obj.weight * var
            coadd_norm[idx] += obj.weight

        w = coadd_norm > 0
        coadd_delta[w] /= coadd_norm[w]
        coadd_cont[w] /= coadd_norm[w]
        coadd_lss[w] /= coadd_norm[w]
        coadd_ivar[w] = coadd_norm[w]**2 / coadd_ivar[w]

        if self.reso is not None:
            self._coadd_reso(other, nwaves, idxes)

        self.wave = coadd_wave
        self.delta = coadd_delta
        self.ivar = coadd_ivar
        self.weight = self.ivar / (1 + self.ivar * coadd_lss)
        self.cont = coadd_cont

        self.mean_snr = np.dot(
            np.sqrt(coadd_ivar), coadd_delta + 1) / np.sum(coadd_ivar > 0)
        self.header['MEANSNR'] = self.mean_snr

    @property
    def ra(self):
        """float: Right ascension in degrees."""
        return np.degrees(self.header['RA'])

    @property
    def dec(self):
        """float: Declination in degrees"""
        return np.degrees(self.header['DEC'])
