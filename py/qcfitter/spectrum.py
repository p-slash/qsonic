import numpy as np

from qcfitter.mathtools import get_smooth_ivar


def add_wave_region_parser(parser):
    """ Adds wavelength analysis related arguments to parser. These
    arguments are grouped under 'Wavelength analysis region'. All of them
    come with defaults, none are required.

    Arguments
    ---------
    parser: argparse.ArgumentParser
    """
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
    return (spec for spec in spectra_list if spec.cont_params['valid'])


class Spectrum(object):
    """Represents one spectrum.

    Parameters
    ----------
    catrow: ndarray
        Catalog row.
    wave: dict of numpy array
        Dictionary of arrays specifying the wavelength grid. Static variable!
    flux: dict
        Dictionary of arrays specifying the flux.
    ivar: dict
        Dictionary of arrays specifying the inverse variance.
    mask: dict
        Dictionary of arrays specifying the bitmask. Not stored
    reso: dict
        Dictionary of 2D arrays specifying the resolution matrix.

    Attributes
    ----------
    arms: list
        List of characters to id spectrograph like 'B', 'R' and 'Z'.
        Static variable!
    z_qso: float
        Quasar redshift.
    rsnr: float
        Average SNR above Lya. Calculated in set_forest_region.
    targetid: int
        Unique TARGETID identifier.
    ra, dec: float
        RA and DEC
    _f1, _f2: dict of int
        Forest indices. Set up using set_forest method. Then use property
        functions to access forest wave, flux, ivar instead.
    cont_params: dict
        Initial estimates are constructed.

    Methods
    ----------
    set_forest_region
    remove_nonforest_pixels
    get_real_size
    coadd_arms_forest

    """
    WAVE_LYA_A = 1215.67
    _wave = None  # Dictionary of ndarrays
    _dwave = None  # Float

    @staticmethod
    def _set_wave(wave, check_consistency=False):
        if not Spectrum._wave:
            Spectrum._wave = wave.copy()
            arm = list(wave.keys())[0]
            wave_arm = wave[arm]
            Spectrum._dwave = wave_arm[1] - wave_arm[0]
        elif check_consistency:
            for arm, wave_arm in Spectrum._wave.items():
                assert (arm in wave.keys())
                assert (np.allclose(Spectrum._wave[arm], wave_arm))

    def __init__(self, catrow, wave, flux, ivar, mask, reso, idx):
        self.catrow = catrow
        Spectrum._set_wave(wave)

        self.flux = {}
        self.ivar = {}
        self.reso = {}

        self.rsnr = 0
        self._f1 = {}
        self._f2 = {}
        self._forestwave = {}
        self._forestflux = {}
        self._forestivar = {}
        self._forestivar_sm = {}
        self._forestreso = {}

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

    def set_forest_region(self, w1, w2, lya1, lya2):
        """ Sets slices for the forest region. Also calculates
        average SNR above Lya

        Arguments
        ---------
        w1, w2: floats
        Observed wavelength range

        lya1, lya2: floats
        Rest-frame wavelength for the forest
        """
        l1 = max(w1, (1 + self.z_qso) * lya1)
        l2 = min(w2, (1 + self.z_qso) * lya2)
        rsnr_weight = 1e-6

        a0 = 1e-6
        n0 = 1e-6
        for arm, wave_arm in self.wave.items():
            # Calculate SNR above Lya
            ii1 = np.searchsorted(
                wave_arm, (1 + self.z_qso) * Spectrum.WAVE_LYA_A)
            weight = np.sqrt(self.ivar[arm][ii1:])
            self.rsnr += np.dot(self.flux[arm][ii1:], weight)
            rsnr_weight += np.sum(weight > 0)

            # Slice to forest limits
            ii1, ii2 = np.searchsorted(wave_arm, [l1, l2])
            real_size_arm = np.sum(self.ivar[arm][ii1:ii2] > 0)
            if real_size_arm == 0:
                continue

            # if larger than skip ratio, add to dict
            self._f1[arm], self._f2[arm] = ii1, ii2

            # Does this create a view or copy array?
            self._forestwave[arm] = wave_arm[ii1:ii2]
            self._forestflux[arm] = self.flux[arm][ii1:ii2]
            self._forestivar[arm] = self.ivar[arm][ii1:ii2]
            if self.reso:
                self._forestreso[arm] = self.reso[arm][:, ii1:ii2]

            # np.shares_memory(self.forestflux, self.flux)
            w = self.forestflux[arm] > 0

            a0 += np.dot(self.forestflux[arm][w], self.forestivar[arm][w])
            n0 += np.sum(self.forestivar[arm][w])

        self.rsnr /= rsnr_weight
        self.cont_params['x'][0] = a0 / n0
        self._forestivar_sm = self._forestivar

    def drop_short_arms(self, lya1=0, lya2=0, skip_ratio=0):
        """Arms that have less than skip_ratio pixels are removed
        from forest dictionary.

        Arguments
        ---------
        lya1, lya2: floats
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
            self.cont_params['cont'].pop(arm, None)

    def remove_nonforest_pixels(self):
        self.flux = self.forestflux
        self.ivar = self.forestivar
        self.reso = self.forestreso

        # Is this needed?
        self._forestflux = self.flux
        self._forestivar = self.ivar
        self._forestreso = self.reso

    def get_real_size(self):
        size = 0
        for ivar_arm in self.forestivar.values():
            size += np.sum(ivar_arm > 0)

        return size

    def is_long(self, dforest_wave, skip_ratio):
        npixels = (1 + self.z_qso) * dforest_wave / self.dwave
        return self.get_real_size() > skip_ratio * npixels

    def set_smooth_ivar(self):
        self._forestivar_sm = {}
        for arm, ivar_arm in self.forestivar.items():
            self._forestivar_sm[arm] = get_smooth_ivar(ivar_arm)

    def _coadd_arms_reso(self, nwaves, idxes):
        max_ndia = np.max([reso.shape[0] for reso in self.forestreso.values()])
        coadd_reso = np.zeros((max_ndia, nwaves))
        creso_norm = np.zeros(nwaves)

        for arm, reso_arm in self.forestreso.items():
            reso_arm = self.forestreso[arm]
            ddia = max_ndia - reso_arm.shape[0]
            if ddia > 0:
                reso_arm = np.pad(reso_arm, ((ddia, ddia), (0, 0)))

            coadd_reso[:, idxes[arm]] += reso_arm
            creso_norm[idxes[arm]] += 1

        coadd_reso /= creso_norm
        self._forestreso = {'brz': coadd_reso}

    def coadd_arms_forest(self, varlss_interp):
        """ Coadds different arms using smoothed pipeline ivar and var_lss.
        Resolution matrix is equally weighted!
        """
        if not self.cont_params['valid'] or not self.cont_params['cont']:
            raise Exception("Continuum needed for coadding.")

        min_wave = np.min([wave[0] for wave in self.forestwave.values()])
        max_wave = np.max([wave[-1] for wave in self.forestwave.values()])

        nwaves = int((max_wave - min_wave) / self.dwave + 0.5) + 1
        coadd_wave = np.arange(nwaves) * self.dwave + min_wave
        coadd_flux = np.zeros(nwaves)
        coadd_ivar = np.zeros(nwaves)
        coadd_cont = np.empty(nwaves)
        coadd_norm = np.zeros(nwaves)

        idxes = {}
        for arm, wave_arm in self.forestwave.items():
            idx = ((wave_arm - min_wave) / self.dwave + 0.5).astype(int)
            idxes[arm] = idx

            var_lss = varlss_interp(wave_arm)
            var_lss *= self.cont_params['cont'][arm]**2
            ivar2 = self.forestivar_sm[arm]
            weight = ivar2 / (1 + ivar2 * var_lss)

            var = np.zeros_like(ivar2)
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

        if self.reso:
            self._coadd_arms_reso(nwaves, idxes)

    def mean_snr(self):
        snr = 0
        npix = 1e-6
        for arm, ivar_arm in self.forestivar.items():
            w = ivar_arm > 0
            armpix = np.sum(w)
            if armpix == 0:
                continue

            snr += np.dot(np.sqrt(ivar_arm), self.forestflux[arm])
            npix += armpix
        return snr / npix

    def write(self, fts_file, varlss_interp):
        hdr_dict = {
            'LOS_ID': self.targetid,
            'TARGETID': self.targetid,
            'RA': self.ra, 'DEC': self.dec,
            'Z': self.z_qso,
            'BLINDING': "none",
            'WAVE_SOLUTION': "lin",
            'MEANSNR': 0.,
            'RSNR': self.rsnr,
            'DLAMBDA': self.dwave,
        }

        for arm, wave_arm in self.forestwave.items():
            armpix = np.sum(self.forestivar[arm] > 0)
            if armpix == 0:
                continue

            hdr_dict['MEANSNR'] = np.dot(
                np.sqrt(self.forestivar[arm]),
                self.forestflux[arm]
            ) / armpix

            cont_est = self.cont_params['cont'][arm]
            delta = self.forestflux[arm] / cont_est - 1
            ivar = self.forestivar[arm] * cont_est**2
            var_lss = varlss_interp(wave_arm)
            weight = ivar / (1 + ivar * var_lss)

            cols = [wave_arm, delta, ivar, weight, cont_est]
            names = ['LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT']
            if self.forestreso:
                cols.append(self.forestreso[arm].T.astype('f8'))
                names.append('RESOMAT')

            fts_file.write(
                cols, names=names, header=hdr_dict,
                extname=f"{self.targetid}-{arm}")

    @property
    def z_qso(self):
        return self.catrow['Z']

    @property
    def targetid(self):
        return self.catrow['TARGETID']

    @property
    def hpix(self):
        return self.catrow['HPXPIXEL']

    @property
    def ra(self):
        return self.catrow['RA']

    @property
    def dec(self):
        return self.catrow['DEC']

    @property
    def wave(self):
        return Spectrum._wave

    @property
    def dwave(self):
        return Spectrum._dwave

    @property
    def forestwave(self):
        return self._forestwave

    @property
    def forestflux(self):
        return self._forestflux

    @property
    def forestivar(self):
        return self._forestivar

    @property
    def forestivar_sm(self):
        return self._forestivar_sm

    @property
    def forestreso(self):
        return self._forestreso
