""" This module contains functions to read spectra and save deltas. Can be
imported without a need for MPI."""

import argparse
import warnings

import fitsio
import numpy as np

from qsonic import QsonicException
import qsonic.spectrum


def add_io_parser(parser=None):
    """ Adds IO related arguments to parser. These arguments are grouped under
    'Input/output parameters and selections'. ``--input-dir`` and ``--catalog``
    are required.

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

    ingroup = parser.add_argument_group('Input options')
    ingroup.add_argument(
        "--input-dir", '-i', required=True,
        help="Input directory.")
    ingroup.add_argument(
        "--catalog", required=True,
        help="Catalog filename")
    ingroup.add_argument(
        "--mock-analysis", action="store_true",
        help="Input folder is mock. Uses nside=16")
    ingroup.add_argument(
        "--keep-surveys", nargs='+', default=['sv3', 'main'],
        help="Surveys to keep.")
    ingroup.add_argument(
        "--skip-resomat", action="store_true",
        help="Skip reading resolution matrix for 3D.")
    ingroup.add_argument(
        "--arms", default=['B', 'R'], choices=['B', 'R', 'Z'], nargs='+',
        help="Arms to read.")

    outgroup = parser.add_argument_group('Output options')
    outgroup.add_argument(
        "--outdir", '-o',
        help="Output directory to save deltas.")
    outgroup.add_argument(
        "--coadd-arms", action="store_true",
        help="Coadds arms when saving.")
    outgroup.add_argument(
        "--save-by-hpx", action="store_true",
        help="Save by healpix. If not, saves by MPI rank.")
    return parser


def read_spectra_onehealpix(
        catalog_hpx, input_dir, arms_to_keep, mock_analysis, skip_resomat,
        program="dark"
):
    """ Returns a list of Spectrum objects for a given catalog of a single
    healpix.

    Arguments
    ---------
    catalog_hpx: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog of quasars in a single healpix. If for data 'SURVEY' column
        must be present and sorted.
    input_dir: str
        Input directory
    arms_to_keep: list(str)
        Must only contain B, R and Z
    mock_analysis: bool
        Reads for mock data if true.
    skip_resomat: bool
        If true, do not read resomat.
    program: str, default: "dark"
        Always use dark program.

    Returns
    ---------
    spectra_list: list(Spectrum)

    Raises
    ---------
    RuntimeWarning
        If number of quasars in the healpix file does not match the catalog.
    """
    spectra_list = []

    if mock_analysis:
        data, idx_cat = read_onehealpix_file_mock(
            catalog_hpx, input_dir, arms_to_keep, skip_resomat
        )

        if idx_cat.size != catalog_hpx.size:
            catalog_hpx = catalog_hpx[idx_cat]

        spectra_list.extend(
            qsonic.spectrum.generate_spectra_list_from_data(
                catalog_hpx, data)
        )
    else:
        # Assume sorted by survey
        # cat.sort(order='SURVEY')
        unique_surveys, s2 = np.unique(
            catalog_hpx['SURVEY'], return_index=True)
        survey_split_cat = np.split(catalog_hpx, s2[1:])

        for cat_by_survey in survey_split_cat:
            data, idx_cat = read_onehealpix_file_data(
                cat_by_survey, input_dir, arms_to_keep,
                skip_resomat, program
            )

            if idx_cat.size != cat_by_survey.size:
                cat_by_survey = cat_by_survey[idx_cat]

            spectra_list.extend(
                qsonic.spectrum.generate_spectra_list_from_data(
                    cat_by_survey, data
                )
            )

    return spectra_list


def read_resolution_matrices_onehealpix_data(
        catalog_hpx, input_dir, spectra_list, program="dark"
):
    """ Returns a list of Spectrum objects for a given catalog of a single
    healpix.

    Arguments
    ---------
    catalog_hpx: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog of quasars in a single healpix. If for data 'SURVEY' column
        must be present and sorted.
    input_dir: str
        Input directory
    spectra_list: list(Spectrum)
        List of spectra to read resolution matrix
    program: str, default: "dark"
        Always use dark program.

    Returns
    ---------
    spectra_list: list(Spectrum)

    Raises
    ---------
    RuntimeWarning
        If number of quasars in the healpix file does not match the catalog.
    """
    # assert (catalog_hpx.size == len(spectra_list))

    unique_surveys, s2 = np.unique(
        catalog_hpx['SURVEY'], return_index=True)
    survey_split_cat = np.split(catalog_hpx, s2[1:])
    s2 = np.append(s2, len(spectra_list))

    arms_to_keep = spectra_list[0].wave.keys()

    for ii, cat_by_survey in enumerate(survey_split_cat):
        survey = cat_by_survey['SURVEY'][0]
        pixnum = cat_by_survey['HPXPIXEL'][0]
        targetids_by_survey = cat_by_survey['TARGETID']

        fspec = (f"{input_dir}/{survey}/{program}/{pixnum//100}/"
                 f"{pixnum}/coadd-{survey}-{program}-{pixnum}.fits")

        i1, i2 = s2[ii], s2[ii + 1]
        spectra_list[i1:i2] = _read_onehealpix_file_onlyreso(
            targetids_by_survey, fspec, arms_to_keep, spectra_list[i1:i2])

    return spectra_list


def read_deltas(fname):
    """ Returns a list of all Delta objects in a file.

    Arguments
    ---------
    fname: str
        FITS file name

    Returns
    ---------
    deltas_list: list(Delta)

    Raises
    ---------
    RuntimeError
        If the file is missing certain columns and keys. See :class:`Delta`.
    """
    deltas_list = []
    fitsfile = fitsio.FITS(fname)

    for hdu in fitsfile[1:]:
        deltas_list.append(qsonic.spectrum.Delta(hdu))

    fitsfile.close()

    return deltas_list


def save_deltas(
        spectra_list, outdir, save_by_hpx=False, mpi_rank=None
):
    """ Saves given list of spectra as deltas. NO coaddition of arms.
    Each arm is saved separately. Only valid spectra are saved.

    Arguments
    ---------
    spectra_list: list(Spectrum)
        Continuum fitted spectra objects. All must be valid!
    outdir: str
        Output directory. Does not save if empty of None
    save_by_hpx: bool, default: False
        Saves by healpix if True. Has priority over mpi_rank
    mpi_rank: int, default: None
        Rank of the MPI process. Save by `mpi_rank` if passed.

    Raises
    ---------
    QsonicException
        If both `mpi_rank` and `save_by_hpx` is None/False.
    QsonicException
        If blinding is not set.
    """
    if not outdir:
        return

    if save_by_hpx:
        pixnos = np.array([spec.hpix for spec in spectra_list])
        sort_idx = np.argsort(pixnos)
        pixnos = pixnos[sort_idx]
        unique_pix, s = np.unique(pixnos, return_index=True)
        split_spectra = np.split(np.array(spectra_list)[sort_idx], s[1:])
    elif mpi_rank is not None:
        unique_pix = [mpi_rank]
        split_spectra = [spectra_list]
    else:
        raise QsonicException("save_by_hpx and mpi_rank can't both be None.")

    if qsonic.spectrum.Spectrum.blinding_not_set():
        raise QsonicException("Blinding is not set. Cannot save delta.")

    for healpix, hp_specs in zip(unique_pix, split_spectra):
        results = fitsio.FITS(
            f"{outdir}/delta-{healpix}.fits", 'rw', clobber=True)

        for spec in qsonic.spectrum.valid_spectra(hp_specs):
            spec.write(results)

        results.close()


def _read_resoimage(imhdu, quasar_indices, nwave):
    # Reading into sorted order is faster
    # Since the output catalog is already sorted, we place them in order
    ndiags = imhdu.get_dims()[1]
    dtype = imhdu._get_image_numpy_dtype()
    data = np.empty((quasar_indices.size, ndiags, nwave), dtype=dtype)
    for idata, iqso in enumerate(quasar_indices):
        data[idata, :, :] = imhdu[int(iqso), :, :]

    return data


def _read_imagehdu(imhdu, quasar_indices, nwave):
    dtype = imhdu._get_image_numpy_dtype()
    data = np.empty((quasar_indices.size, nwave), dtype=dtype)
    for idata, iqso in enumerate(quasar_indices):
        data[idata, :] = imhdu[int(iqso), :]

    return data

# Timing showed this is slower
# def _read_imagehdu_2(imhdu, quasar_indices):
#     ndims = imhdu.get_info()['ndims']
#     if ndims == 2:
#         return np.vstack([imhdu[int(idx), :] for idx in quasar_indices])
#     return np.vstack([imhdu[int(idx), :, :] for idx in quasar_indices])


def _read_onehealpix_file(
        targetids_by_survey, fspec, arms_to_keep, skip_resomat
):
    """ Common function to read a single fits file.

    Arguments
    ---------
    targetids_by_survey: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Targetids_by_survey (used to be catalog). If data, split by survey and
        contains only one survey.
    fspec: str
        Filename to open.
    arms_to_keep: list(str)
        Must only contain B, R and Z.
    skip_resomat: bool
        If true, do not read resomat.

    Returns
    ---------
    data: dict( :external+numpy:py:class:`ndarray <numpy.ndarray>` )
        Only quasar spectra are read into keywords wave, flux etc. Resolution
        is read if present.
    idx_cat: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Indices in `targetids_by_survey` there were succesfully read.

    Raises
    ---------
    RuntimeWarning
        If number of quasars in the healpix file does not match the catalog.
    """
    # Assume it is sorted
    # cat_by_survey.sort(order='TARGETID')
    fitsfile = fitsio.FITS(fspec)

    fbrmap = fitsfile['FIBERMAP'].read(columns='TARGETID')
    common_targetids, idx_fbr, idx_cat = np.intersect1d(
        fbrmap, targetids_by_survey, assume_unique=True, return_indices=True)
    if (common_targetids.size != targetids_by_survey.size):
        warnings.warn(
            f"Error reading {fspec}. "
            "Number of quasars in healpix does not match the catalog "
            f"catalog:{targetids_by_survey.size} vs "
            f"healpix:{common_targetids.size}!", RuntimeWarning)

    data = {
        'wave': {},
        'flux': {},
        'ivar': {},
        'mask': {},
        'reso': {}
    }

    for arm in arms_to_keep:
        # Cannot read by rows= argument.
        data['wave'][arm] = fitsfile[f'{arm}_WAVELENGTH'].read()
        nwave = data['wave'][arm].size

        data['flux'][arm] = _read_imagehdu(
            fitsfile[f'{arm}_FLUX'], idx_fbr, nwave)
        data['ivar'][arm] = _read_imagehdu(
            fitsfile[f'{arm}_IVAR'], idx_fbr, nwave)
        data['mask'][arm] = _read_imagehdu(
            fitsfile[f'{arm}_MASK'], idx_fbr, nwave)

        if skip_resomat or f'{arm}_RESOLUTION' not in fitsfile:
            continue

        data['reso'][arm] = _read_resoimage(
            fitsfile[f'{arm}_RESOLUTION'], idx_fbr, nwave)

    fitsfile.close()

    return data, idx_cat


def _read_onehealpix_file_onlyreso(
        targetids_by_survey, fspec, arms_to_keep, spectra_list
):
    """ Function to read forest region resolution matrix for data. Note reading
    resolution matrix is already fast for mocks.

    Arguments
    ---------
    targetids_by_survey: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Targetids_by_survey (used to be catalog). If data, split by survey and
        contains only one survey.
    fspec: str
        Filename to open.
    arms_to_keep: list(str)
        Must only contain B, R and Z.
    spectra_list: list(Spectrum)
        List of spectra that forest region is set. Modified in place and also
        returned.

    Returns
    ---------
    spectra_list: list(Spectrum)
        List of spectra where forestreso is set.

    Raises
    ---------
    RuntimeWarning
        If number of quasars in the healpix file does not match the catalog.
    """
    fitsfile = fitsio.FITS(fspec)

    fbrmap = fitsfile['FIBERMAP'].read(columns='TARGETID')
    common_targetids, idx_fbr, idx_cat = np.intersect1d(
        fbrmap, targetids_by_survey, assume_unique=True, return_indices=True)
    if (common_targetids.size != targetids_by_survey.size):
        warnings.warn(
            f"Error reading {fspec}. "
            "Number of quasars in healpix does not match the catalog "
            f"catalog:{targetids_by_survey.size} vs "
            f"healpix:{common_targetids.size}!", RuntimeWarning)

    for arm in arms_to_keep:
        imhdu = fitsfile[f'{arm}_RESOLUTION']

        for jj, spec in zip(idx_fbr, spectra_list):
            # assert (common_targetids[jj] == spec.targetid)

            if arm not in spec.forestwave.keys():
                continue

            f1 = spec._f1[arm]
            f2 = spec._f2[arm]
            spec._forestreso[arm] = imhdu[int(jj), :, f1:f2][0]

    fitsfile.close()

    return spectra_list


def read_onehealpix_file_data(
        cat_by_survey, input_dir, arms_to_keep, skip_resomat, program="dark"
):
    """ Read a single fits file for data.

    Arguments
    ---------
    cat_by_survey: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog for a single survey and healpix. Ordered by TARGETID.
    input_dir: str
        Input directory.
    arms_to_keep: list(str)
        Must only contain B, R and Z.
    skip_resomat: bool
        If true, do not read resomat.

    Returns
    ---------
    data: dict
        Only quasar spectra are read into keywords wave, flux etc. Resolution
        is read if present.
    idx_cat: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Indices in `cat_by_survey` there were succesfully read.

    Raises
    ---------
    RuntimeWarning
        If number of quasars in the healpix file does not match the catalog.
    """
    survey = cat_by_survey['SURVEY'][0]
    pixnum = cat_by_survey['HPXPIXEL'][0]

    fspec = (f"{input_dir}/{survey}/{program}/{pixnum//100}/"
             f"{pixnum}/coadd-{survey}-{program}-{pixnum}.fits")
    data, idx_cat = _read_onehealpix_file(
        cat_by_survey['TARGETID'], fspec, arms_to_keep, skip_resomat)

    return data, idx_cat


def read_onehealpix_file_mock(
        catalog_hpx, input_dir, arms_to_keep, skip_resomat, nside=16
):
    """ Read a single FITS file for mocks.

    Arguments
    ---------
    catalog_hpx: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Catalog for a single healpix. Ordered by TARGETID.
    input_dir: str
        Input directory.
    arms_to_keep: list(str)
        Must only contain B, R and Z.
    skip_resomat: bool
        If true, do not read resomat.

    Returns
    ---------
    data: dict
        Only quasar spectra are read into keywords wave, flux etc. Resolution
        is read if present.
    idx_cat: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        Indices in `catalog_hpx` there were succesfully read.

    Raises
    ---------
    RuntimeWarning
        If number of quasars in the healpix file does not match the catalog.
    """
    pixnum = catalog_hpx['HPXPIXEL'][0]
    fspec = f"{input_dir}/{pixnum//100}/{pixnum}/spectra-{nside}-{pixnum}.fits"
    data, idx_cat = _read_onehealpix_file(
        catalog_hpx['TARGETID'], fspec, arms_to_keep, skip_resomat)

    if skip_resomat:
        return data, idx_cat

    fspec = f"{input_dir}/{pixnum//100}/{pixnum}/truth-{nside}-{pixnum}.fits"
    fitsfile = fitsio.FITS(fspec)
    for arm in arms_to_keep:
        data['reso'][arm] = np.array(fitsfile[f'{arm}_RESOLUTION'].read())
    fitsfile.close()

    return data, idx_cat


def _float_range(f1, f2):
    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range.
        """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < f1 or f > f2:
            raise argparse.ArgumentTypeError(f"must be in range [{f1}--{f2}]")
        return f

    # Return function handle to checking function
    return float_range_checker
