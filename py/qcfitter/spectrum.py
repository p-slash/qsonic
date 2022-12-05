import numpy as np
import fitsio

def _read_onehealpix_file(cat_by_survey, fspec, arms_to_keep):
    """Common function to read a single fits file.

    Arguments
    ---------
    cat_by_survey: named np.array
    catalog. If data, split by survey and contains only one survey.

    fspec: str
    filename to open

    arms_to_keep: list of str
    must only contain B, R and Z

    Returns
    ---------
    data: dict
    only quasar spectra are read into keywords wave, flux etc. Resolution is read if present.

    quasar_indices: np.array of int
    indices of quasars in fits file.
    """
    cat_by_survey.sort(order='TARGETID')
    fitsfile = fitsio.FITS(fspec)

    fbrmap = fitsfile['FIBERMAP'].read()
    isin = np.isin(fbrmap['TARGETID'], cat_by_survey['TARGETID'])
    quasar_indices = np.nonzero(isin)[0]
    if (quasar_indices.size != cat_by_survey.size):
        logging.error(f"Error not all targets are in file {cat_by_survey.size} vs {quasar_indices.size}")

    fbrmap = fbrmap[isin]
    sort_idx = fbrmap.argsort(order='TARGETID')
    fbrmap = fbrmap[sort_idx]

    assert np.all(cat_by_survey['TARGETID'] == fbrmap['TARGETID'])

    data = {
        'wave': {},
        'flux': {},
        'ivar': {},
        'mask': {},
        'reso': {}
    }

    for arm in arms_to_keep:
        data['wave'][arm] = np.array(fitsfile[f'{arm}_WAVELENGTH'].read())
        data['flux'][arm] = np.array(fitsfile[f'{arm}_FLUX'].read(rows=quasar_indices)[sort_idx])
        data['ivar'][arm] = np.array(fitsfile[f'{arm}_IVAR'].read(rows=quasar_indices)[sort_idx])
        data['mask'][arm] = np.array(fitsfile[f'{arm}_MASK'].read(rows=quasar_indices)[sort_idx])
        if f'{arm}_RESOLUTION' in fitsfile:
            data['reso'][arm] = np.array(fitsfile[f'{arm}_RESOLUTION'].read(rows=quasar_indices)[sort_idx])

    fitsfile.close()

    return data, quasar_indices

def read_onehealpix_file_data(cat_by_survey, input_dir, pixnum, arms_to_keep, program="dark"):
    survey = cat_by_survey['SURVEY'][0]

    fspec = f"{input_dir}/{survey}/{program}/{pixnum//100}/{pixnum}/coadd-{survey}-{program}-{pixnum}.fits"
    data, quasar_indices = _read_onehealpix_file(cat_by_survey, fspec, arms_to_keep)

    return data, quasar_indices.size

def read_onehealpix_file_mock(cat, input_dir, pixnum, arms_to_keep, nside=16):
    fspec = f"{input_dir}/{pixnum//100}/{pixnum}/spectra-{nside}-{pixnum}.fits"
    data, quasar_indices = _read_onehealpix_file(cat, fspec, arms_to_keep)

    fspec = f"{input_dir}/{pixnum//100}/{pixnum}/truth-{nside}-{pixnum}.fits"
    fitsfile = fitsio.FITS(fspec)
    for arm in arms_to_keep:
        data['reso'][arm] = np.array(fitsfile[f'{arm}_RESOLUTION'].read())
    fitsfile.close()

    return data, quasar_indices.size

def generate_spectra_list_from_data(cat_by_survey, data, nquasars):
    spectra_list = []
    for idx in range(nquasars):
        row = cat_by_survey[idx]
        z_qso = row['Z']
        targetid = row['TARGETID']

        spectra_list.append(
            Spectrum(z_qso, targetid, data['wave'], data['flux'],
                data['ivar'], data['mask'], data['reso'], idx)
        )

    return spectra_list

def read_spectra(cat, input_dir, arms_to_keep, mock_analysis, program="dark"):
    """ Returns a list of Spectrum objects for a given catalog.

    Arguments
    ---------
    cat: named np.array
    catalog of quasars in single healpix.

    input_dir: str
    input directory

    arms_to_keep: list of str
    must only contain B, R and Z

    mock_analysis: bool
    reads for mock data if true.

    program: str
    always use dark program.

    Returns
    ---------
    spectra_list: list of Spectrum
    """
    spectra_list = []
    pixnum = cat['PIXNUM'][0]

    if not mock_analysis:
        cat.sort(order='SURVEY')
        unique_surveys, s2 = np.unique(cat['SURVEY'], return_index=True)
        survey_split_cat = np.split(cat, s2[1:])

        for cat_by_survey in survey_split_cat:
            data, nquasars = read_onehealpix_file_data(cat_by_survey, input_dir, pixnum, arms_to_keep, program)
            spectra_list.extend(
                generate_spectra_list_from_data(cat_by_survey, data, nquasars)
            )
    else:
        data, nquasars = read_onehealpix_file_mock(cat, input_dir, pixnum, arms_to_keep)
        spectra_list.extend(
            generate_spectra_list_from_data(cat, data, nquasars)
        )
    
    return spectra_list

class Spectrum(object):
    """Represents one spectrum.

    Parameters
    ----------
    z_qso: float
        Quasar redshift.
    targetid: int
        Unique TARGETID identifier.
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
        List of characters to id spectrograph like 'B', 'R' and 'Z'. Static variable!
    _f1, _f2: dict of int
        Forest indices. Set up using set_forest method. Then use property functions
        to access forest wave, flux, ivar instead.
    cont_params: dict
        Initial estimates are constructed.

    Methods
    ----------
    removePixels(idx_to_remove)

    """
    _wave = None
    _arms = None
    @staticmethod
    def _set_wave(wave):
        if not Spectrum._wave:
            Spectrum._arms = wave.keys()
            Spectrum._wave = wave.copy()
        else:
            for arm in Spectrum._arms:
                assert (arm in wave.keys())
                assert (np.allclose(Spectrum._wave[arm], wave[arm]))

    def __init__(self, z_qso, targetid, wave, flux, ivar, mask, reso, idx):
        self.z_qso = z_qso
        self.targetid = targetid
        Spectrum._set_wave(wave)

        self.flux = {}
        self.ivar = {}
        self.reso = {}

        self._f1 = {}
        self._f2 = {}
        self._forestwave = {}
        self._forestflux = {}
        self._forestivar = {}
        self._forestreso = {}

        for arm in self.arms:
            self._f1[arm] = 0
            self._f2[arm] = self.wave[arm].size
            self.flux[arm] = flux[arm][idx]
            self.ivar[arm] = ivar[arm][idx]
            _mask = mask[arm][idx]
            self.flux[arm][_mask] = 0
            self.ivar[arm][_mask] = 0

            if reso[arm].ndim == 2:
                self.reso[arm] = reso[arm].copy()
            else:
                self.reso[arm] = reso[arm][idx]

        self.cont_params = {}
        self.cont_params['method'] = ''
        self.cont_params['valid'] = False
        self.cont_params['x'] = np.array([1., 0.])

    # def set_continuum(self, wave_p, cont_p):
    #     for arm in self.arms:
    #         self.cont[arm] = np.interp(self.forestwave[arm], wave_p, cont_p)

    def set_forest_region(self, w1, w2, lya1, lya2):
        l1 = max(w1, (1+self.z_qso)*lya1)
        l2 = min(w2, (1+self.z_qso)*lya2)

        a0 = 1e-6
        n0 = 1e-6
        for arm in self.arms:
            self._f1[arm], self._f2[arm] = np.searchsorted(self.wave[arm], [l1, l2])

            # Does this create a view or copy array?
            self._forestwave[arm] = self.wave[arm][self._f1[arm]:self._f2[arm]]
            self._forestflux[arm] = self.flux[arm][self._f1[arm]:self._f2[arm]]
            self._forestivar[arm] = self.ivar[arm][self._f1[arm]:self._f2[arm]]
            self._forestreso[arm] = self.reso[arm][:, self._f1[arm]:self._f2[arm]]

            # np.shares_memory(self.forestflux, self.flux)
            w = self.forestflux[arm]>0

            a0 += np.sum(self.forestflux[arm][w]*self.forestivar[arm][w])
            n0 += np.sum(self.forestivar[arm][w])

        self.cont_params['x'][0] = a0/n0

    @property
    def wave(self):
        return Spectrum._wave

    @property
    def arms(self):
        return Spectrum._arms

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
    def forestreso(self):
        return self._forestreso

    def remove_nonforest_pixels(self):
        self.flux = self.forestflux
        self.ivar = self.forestivar
        self.reso = self.forestreso

        # Is this needed?
        self._forestflux = self.flux
        self._forestivar = self.ivar
        self._forestreso = self.reso



