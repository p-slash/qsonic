import numpy as np
from healpy import ang2pix
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
            Spectrum(z_qso, targetid, row['RA'], row['DEC'], data['wave'], data['flux'],
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
    pixnum = cat['HPXPIXEL'][0]

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

def save_deltas(spectra_list, outdir, out_nside, varlss_interp):
    """ Saves given list of spectra as deltas. NO coaddition of arms.
    Each arm is saved separately

    Arguments
    ---------
    spectra_list: list of Spectrum
    continuum fitted spectra objects

    outdir: str
    output directory

    out_nside: int
    output healpix nside

    varlss_interp: Interpolator
    interpolator for LSS variance

    """
    pixnos = np.array([ang2pix(out_nside, spec.ra, spec.dec, lonlat=True, nest=True)
        for spec in spectra_list])
    sort_idx = np.argsort(pixnos)
    pixnos = pixnos[sort_idx]
    unique_pix, s = np.unique(pixnos, return_index=True)
    split_spectra = np.split(np.array(spectra_list)[sort_idx], s[1:])

    for healpix, hp_specs in zip(unique_pix, split_spectra):
        results = fitsio.FITS(f"{outdir}/deltas-{healpix}.fits.gz",'rw', clobber=True)

        for spec in hp_specs:
            if not spec.cont_params['valid']:
                continue

            hdr_dict = {
                'LOS_ID': spec.targetid,
                'TARGETID': spec.targetid,
                'RA': spec.ra, 'DEC': spec.dec,
                'Z': spec.z_qso,
                'BLINDING': "none",
                'WAVE_SOLUTION': "lin",
                'MEANSNR': 0.,
                'DLAMBDA': spec.dwave
            }

            for arm, wave_arm in spec.forestwave.items():
                _cont = spec.cont_params['cont'][arm]
                delta = spec.forestflux[arm]/_cont-1
                ivar  = spec.forestivar[arm]*_cont**2
                var_lss = varlss_interp(wave_arm)
                weight = ivar / (1+ivar*var_lss)

                hdr_dict['MEANSNR'] = np.mean(np.sqrt(ivar[ivar>0]))

                cols = [wave_arm, delta, ivar, weight, _cont, spec.forestreso[arm].T]
                names = ['LAMBDA', 'DELTA', 'IVAR', 'WEIGHT', 'CONT', 'RESOMAT']

                results.write(cols, names=names, header=hdr_dict,
                    extname=f"{spec.targetid}-{arm}")

        results.close()

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
    _dwave = None

    @staticmethod
    def _set_wave(wave):
        if not Spectrum._wave:
            Spectrum._arms = wave.keys()
            Spectrum._wave = wave.copy()
        else:
            for arm in Spectrum._arms:
                assert (arm in wave.keys())
                assert (np.allclose(Spectrum._wave[arm], wave[arm]))

                if Spectrum._dwave is None:
                    Spectrum._dwave = wave[arm][1] - wave[arm][0]

    def __init__(self, z_qso, targetid, ra, dec, wave, flux, ivar, mask, reso, idx):
        self.z_qso = z_qso
        self.targetid = targetid
        self.ra = ra
        self.dec = dec
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
            _mask = mask[arm][idx] | np.isnan(self.flux[arm]) | np.isnan(self.ivar[arm])
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
        self.cont_params['cont'] = None

    # def set_continuum(self, wave_p, cont_p):
    #     for arm in self.arms:
    #         self.cont[arm] = np.interp(self.forestwave[arm], wave_p, cont_p)

    def set_forest_region(self, w1, w2, lya1, lya2, skip_ratio=0):
        """ Sets slices for the forest region. Arms that have less than
        skip_ratio pixels will not be added to forest dictionary.

        Arguments
        ---------
        w1, w2: floats
        Observed wavelength range

        lya1, lya2: floats
        Rest-frame wavelength for the forest

        skip_ratio: float
        Remove arms if they have less than this ratio of pixels

        """
        _npixels_expected = int(skip_ratio*(1+self.z_qso)*(lya2 - lya1)/self.dwave)+1

        l1 = max(w1, (1+self.z_qso)*lya1)
        l2 = min(w2, (1+self.z_qso)*lya2)

        a0 = 1e-6
        n0 = 1e-6
        for arm in self.arms:
            ii1, ii2 = np.searchsorted(self.wave[arm], [l1, l2])
            real_size_arm = ii2-ii1 - np.sum(self.ivar[arm][ii1:ii2] == 0)
            if real_size_arm < _npixels_expected:
                continue

            # if larger than skip ratio, add to dict
            self._f1[arm], self._f2[arm] = ii1, ii2

            # Does this create a view or copy array?
            self._forestwave[arm] = self.wave[arm][ii1:ii2]
            self._forestflux[arm] = self.flux[arm][ii1:ii2]
            self._forestivar[arm] = self.ivar[arm][ii1:ii2]
            self._forestreso[arm] = self.reso[arm][:, ii1:ii2]

            # np.shares_memory(self.forestflux, self.flux)
            w = self.forestflux[arm]>0

            a0 += np.sum(self.forestflux[arm][w]*self.forestivar[arm][w])
            n0 += np.sum(self.forestivar[arm][w])

        self.cont_params['x'][0] = a0/n0

    def get_real_size(self):
        size = 0
        for ivar_arm in self.forestivar.values():
            size += ivar_arm.size - np.sum(ivar_arm == 0)

        return size

    @property
    def wave(self):
        return Spectrum._wave
    @property
    def dwave(self):
        return Spectrum._dwave

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



