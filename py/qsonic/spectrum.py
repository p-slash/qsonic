import argparse

import numpy as np

from qsonic import QsonicException
from qsonic.mathtools import _zero_function, _one_function, get_smooth_ivar


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
        "--forest-w1", type=float, default=1040.,
        help="First forest wavelength edge.")
    wave_group.add_argument(
        "--forest-w2", type=float, default=1200.,
        help="Last forest wavelength edge.")

    return parser


def generate_spectra_list_from_data(cat_by_survey, data):
    spectra_list = []
    for idx, catrow in enumerate(cat_by_survey):
        spectra_list.append(
            Spectrum(
                catrow, data['wave'], data['flux'],
                data['ivar'], data['mask'], data['reso'], idx
            )
        )

    return spectra_list


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
        Average SNR above Lya. Calculated in set_forest_region.
    mean_snr: dict(float)
        Mean signal-to-noise ratio in the forest.
    _f1, _f2: dict(int)
        Forest indices. Set up using `set_forest_region` method. Then use
        property functions to access forest wave, flux, ivar instead.
    cont_params: dict
        Continuum parameters. Initial estimates are constructed.
    """
    WAVE_LYA_A = 1215.67
    """float: Lya wavelength in A."""
    _wave = None
    """dict(:external+numpy:py:class:`ndarray <numpy.ndarray>`): Common
    wavelength grid for **all** Spectra."""
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
    def set_blinding(catalog, args):
        """Set the blinding strategy.

        'LASNIGHT' column, args.mock_analysis, args.forest_w1 decide the
        blinding strategy. Mock, side band and SV analyses are not blinded.

        Arguments
        ---------
        catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Entire quasar catalog.
        args: argparse.Namespace
            Should have ``mock_analysis (bool)`` and ``forest_w1 (floar)``.
        """
        # do not blind mocks or metal forests
        if args.mock_analysis or args.forest_w1 > Spectrum.WAVE_LYA_A:
            Spectrum._blinding = "none"
        # sv data, no blinding
        elif all(catalog['LASTNIGHT'] < 20210514):
            Spectrum._blinding = "none"
        elif all(catalog['LASTNIGHT'] < 20210801):
            Spectrum._blinding = "desi_m2"
        elif all(catalog['LASTNIGHT'] < 20220801):
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

    def __init__(self, catrow, wave, flux, ivar, mask, reso, idx):
        self.catrow = catrow
        Spectrum._set_wave(wave)

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
            self._f1[arm], self._f2[arm] = 0, wave_arm.size
            self.flux[arm] = flux[arm][idx]
            self.ivar[arm] = ivar[arm][idx]
            w = (mask[arm][idx] != 0) | np.isnan(self.flux[arm])\
                | np.isnan(self.ivar[arm])
            self.flux[arm][w] = 0
            self.ivar[arm][w] = 0

            if not reso:
                pass
            elif reso[arm].ndim == 2:
                self.reso[arm] = reso[arm].copy()
            else:
                self.reso[arm] = reso[arm][idx]

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

    def set_forest_region(self, w1, w2, lya1, lya2):
        """ Sets slices for the forest region. Also calculates the mean SNR in
        the forest and an initial guess for the continuum amplitude.

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

            # Does this create a view or copy array?
            self._forestwave[arm] = wave_arm[ii1:ii2]
            self._forestflux[arm] = self.flux[arm][ii1:ii2]
            self._forestivar[arm] = self.ivar[arm][ii1:ii2]
            if self.reso:
                self._forestreso[arm] = self.reso[arm][:, ii1:ii2]

        self._forestivar_sm = self._forestivar
        self._forestweight = self._forestivar
        self._set_forest_related_parameters()

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
            self._forestwave.pop(arm, None)
            self._forestflux.pop(arm, None)
            self._forestivar.pop(arm, None)
            self._forestreso.pop(arm, None)
            self._forestivar_sm.pop(arm, None)
            self._forestweight.pop(arm, None)
            self.cont_params['cont'].pop(arm, None)

    def remove_nonforest_pixels(self):
        """ Remove non-forest pixels from storage.

        This equates `flux` to `forestflux` etc, but `wave` is not modified,
        since it is a static variable. Good practive is to loop using, e.g.,
        `for arm, wave_arm in self.forestwave.items():`.
        """
        self.flux = self.forestflux
        self.ivar = self.forestivar
        self.reso = self.forestreso

        # Is this needed?
        self._forestflux = self.flux
        self._forestivar = self.ivar
        self._forestreso = self.reso

    def get_real_size(self):
        """int: Sum of number of pixels with `forestivar > 0` for all arms."""
        size = 0
        for ivar_arm in self.forestivar.values():
            size += np.sum(ivar_arm > 0)

        return size

    def is_long(self, dforest_wave, skip_ratio):
        npixels = (1 + self.z_qso) * dforest_wave / self.dwave
        return self.get_real_size() > skip_ratio * npixels

    def set_smooth_ivar(self, smoothing_size=16.):
        """ Set :attr:`forestivar_sm` to smoothed inverse variance. Before this
        call :attr:`forestivar_sm` points to :attr:`forestivar`. If
        ``smoothing_size <= 0``, smoothing is undone such that ivar_sm
        points to ivar.


        ``smoothing_size`` is saved to a private :attr:`_smoothing_scale`
        variable for future use.

        Arguments
        ---------
        smoothing_size: float, default: 16
            Gaussian smoothing spread in A.
        """
        self._forestivar_sm = {}
        if smoothing_size <= 0:
            self._smoothing_scale = 0
            self._forestivar_sm = self._forestivar
            return

        self._smoothing_scale = smoothing_size
        sigma_pix = smoothing_size / self.dwave
        for arm, ivar_arm in self.forestivar.items():
            self._forestivar_sm[arm] = get_smooth_ivar(ivar_arm, sigma_pix)

    def set_forest_weight(
            self,
            varlss_interp=_zero_function,
            eta_interp=_one_function
    ):
        """ Sets :attr:`forestweight` for a given var_lss and eta correction.
        Always uses :attr:`forestivar_sm`, which is not actually smoothed if
        :meth:`set_smooth_ivar` is not called.

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
        self._forestweight = {}
        if not self.cont_params['valid'] or not self.cont_params['cont']:
            return

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

    def _coadd_arms_reso(self, nwaves, idxes):
        """Coadd resolution matrix"""
        max_ndia = np.max([reso.shape[0] for reso in self.forestreso.values()])
        coadd_reso = np.zeros((max_ndia, nwaves))
        creso_norm = np.zeros(nwaves)

        for arm, reso_arm in self.forestreso.items():
            weight = self.forestweight[arm].copy()
            weight[weight == 0] = 1e-8

            reso_arm = self.forestreso[arm]
            ddia = max_ndia - reso_arm.shape[0]
            # Assumption ddia cannot be odd
            ddia = ddia // 2
            if ddia > 0:
                reso_arm = np.pad(reso_arm, ((ddia, ddia), (0, 0)))

            coadd_reso[:, idxes[arm]] += weight * reso_arm
            creso_norm[idxes[arm]] += weight

        coadd_reso /= creso_norm
        self._forestreso = {'brz': coadd_reso}

    def coadd_arms_forest(
            self,
            varlss_interp=_zero_function,
            eta_interp=_one_function
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
        if not self.cont_params['valid'] or not self.cont_params['cont']:
            raise QsonicException("Continuum needed for coadding.")

        min_wave = np.min([wave[0] for wave in self.forestwave.values()])
        max_wave = np.max([wave[-1] for wave in self.forestwave.values()])

        nwaves = int((max_wave - min_wave) / self.dwave + 0.1) + 1
        coadd_wave = np.arange(nwaves) * self.dwave + min_wave
        coadd_flux = np.zeros(nwaves)
        coadd_ivar = np.zeros(nwaves)
        coadd_cont = np.empty(nwaves)
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

            # continuum needs not weighting
            coadd_cont[idx] = self.cont_params['cont'][arm]

        w = coadd_norm > 0
        coadd_flux[w] /= coadd_norm[w]
        coadd_ivar[w] = coadd_norm[w]**2 / coadd_ivar[w]

        self._forestwave = {'brz': coadd_wave}
        self._forestflux = {'brz': coadd_flux}
        self._forestivar = {'brz': coadd_ivar}
        self.cont_params['cont'] = {'brz': coadd_cont}
        if self.forestreso:
            self._coadd_arms_reso(nwaves, idxes)

        self.set_smooth_ivar(self._smoothing_scale)
        self._forestweight = {}
        self.set_forest_weight(varlss_interp, eta_interp)

        mean_snr = np.dot(
            np.sqrt(coadd_ivar), coadd_flux) / np.sum(coadd_ivar > 0)
        self.mean_snr = {'brz': mean_snr}

    def mean_resolution(self, arm, weight=None):
        """ Returns the weighted mean Gaussian sigma of the spectrograph
        resolution.

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
        'RESOMAT' column if resolution matrix is present to extention name
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

        for arm, wave_arm in self.forestwave.items():
            if self.mean_snr[arm] == 0:
                continue

            hdr_dict['MEANSNR'] = self.mean_snr[arm]

            cont_est = self.cont_params['cont'][arm]
            delta = self.forestflux[arm] / cont_est - 1
            ivar = self.forestivar[arm] * cont_est**2
            weight = self.forestweight[arm] * cont_est**2

            cols = [wave_arm, delta, ivar, weight, cont_est]
            if self.forestreso:
                hdr_dict['MEANRESO'] = self.mean_resolution(arm)
                cols.append(self.forestreso[arm].T.astype('f8'))

            fts_file.write(
                cols, names=Spectrum._fits_colnames, header=hdr_dict,
                extname=f"{self.targetid}-{arm}")

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
        return Spectrum._wave

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
        :meth:`.set_smooth_ivar` is called."""
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
        self.wave = data[key]
        if key == "LOGLAM":
            self.wave = 10**data['LOGLAM']

        key = Delta._check_hdu(colnames, "delta")
        self.delta = data[key]
        self.ivar = data['IVAR']
        self.weight = data['WEIGHT']
        self.cont = data['CONT']
