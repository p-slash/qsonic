import numpy as np
import fitsio

def _read_onehealpix_file(cat_by_survey, fspec, arms_to_keep):
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
    cat_by_survey.sort(order='TARGETID')
    survey = cat_by_survey['SURVEY'][0]

    fspec = f"{input_dir}/{survey}/{program}/{pixnum//100}/{pixnum}/coadd-{survey}-{program}-{pixnum}.fits"
    
    data, quasar_indices = _read_onehealpix_file(cat_by_survey, fspec, arms_to_keep)

    return data, quasar_indices.size

def read_onehealpix_file_mock(cat, input_dir, pixnum, arms_to_keep, nside=16):
    cat.sort(order='TARGETID')

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
    arms: list
        List of characters to id spectrograph like 'B', 'R' and 'Z'.
    wave: dict of numpy array
        Dictionary of arrays specifying the wavelength grid.
    flux: dict
        Dictionary of arrays specifying the flux.
    ivar: dict
        Dictionary of arrays specifying the inverse variance.
    mask: dict
        Dictionary of arrays specifying the bitmask.
    reso: dict
        Dictionary of 2D arrays specifying the resolution matrix.

    Methods
    ----------
    removePixels(idx_to_remove)

    """
    def __init__(self, z_qso, targetid, wave, flux, ivar, mask, reso, idx):
        self.z_qso = z_qso
        self.targetid = targetid
        self.arms = wave.keys()
        self.wave = wave.copy()
        self.flux = {}
        self.ivar = {}
        self.mask = {}
        self.reso = {}

        for arm in self.arms:
            self.flux[arm] = flux[arm][idx]
            self.ivar[arm] = ivar[arm][idx]
            self.mask[arm] = mask[arm][idx]
            if reso[arm].ndim == 2:
                self.reso[arm] = reso[arm].copy()
            else:
                self.reso[arm] = reso[arm][idx]


    # def removePixels(self, idx_to_remove):



