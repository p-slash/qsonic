import logging
import argparse
import sys

import numpy as np
from mpi4py import MPI

from qcfit.catalog import Catalog
from qcfit.spectrum import Spectrum


def balance_load(split_catalog, mpi_size, mpi_rank):
    number_of_spectra = np.zeros(mpi_size, dtype=int)
    local_queue = []
    for cat in split_catalog:
        min_idx = np.argmin(number_of_spectra)
        number_of_spectra[min_idx] += cat.size

        if min_idx == mpi_rank:
            local_queue.append(cat)

    return local_queue

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    # Arguments passed to run the script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", help="Input directory to healpix", required=True)
    parser.add_arguemnt("--catalog", help="Catalog filename", required=True)
    parser.add_argument("--mock-analysis", help="Input folder is mock.", action="store_true")
    parser.add_argument("--arms", help="Arms to read.", default=['B', 'R'], nargs='+')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    if any(arm in args.arms not in ['B', 'R', 'Z']):
        logging.error("Arms should be 'B', 'R' or 'Z'.")
        comm.Abort()

    # read catalog
    if mpi_rank == 0:
        logging.info("Reading catalog on master node.")
        n_side = 16 if args.mock_analysis else 64
        qso_cat = Catalog(args.catalog, n_side)
        logging.info("Broadcasting.")

    comm.Bcast(qso_cat)
    # We decide forest filename list
    # Group into unique pixels
    unique_pix, s = np.unique(qso_cat.catalog['PIXNUM'], return_index=True)
    split_catalog = np.split(qso_cat.catalog, s[1:])

    # Roughly equal number of spectra
    if mpi_rank == 0:
        logging.info("Load balancing.")

    local_queue = balance_load(split_catalog, mpi_size, mpi_rank)

    if mpi_rank == 0:
        logging.info("Reading spectra.")

    spectra_list = []

    # Each process reads its own list
    for cat in local_queue:
        pixnum = cat['PIXNUM'][0]
        program = "dark"

        if not args.mock_analysis:
            cat.sort(order='SURVEY')
            unique_surveys, s2 = np.unique(cat['SURVEY'], return_index=True)
            survey_split_cat = np.split(cat, s2[1:])

            for cat_by_survey in survey_split_cat:
                cat_by_survey.sort(order='TARGETID')
                survey = cat_by_survey['SURVEY'][0]
                fspec = f"{args.input_dir}/{survey}/{program}/{pixnum//100}/{pixnum}/coadd-{survey}-{program}-{pixnum}.fits"
                fitsfile = fitsio.FITS(fspec)
                fbrmap = fits_spec['FIBERMAP'].read()
                isin = np.isin(fibermap['TARGETID'], cat_by_survey['TARGETID'])
                quasar_indices = np.nonzero(isin)[0]
                if (quasar_indices.size != cat_by_survey.size):
                    logging.error(f"Error not all targets are in file {cat_by_survey.size} vs {quasar_indices.size}")

                fbrmap = fbrmap[isin]
                sort_idx = fbrmap.argsort(order='TARGETID')
                fbrmap = fbrmap[sort_idx]

                assert np.all(cat_by_survey['TARGETID'] == fbrmap['TARGETID'])

                wave = {}
                flux = {}
                ivar = {}
                mask = {}
                reso = {}
                for arm in args.arms:
                    wave[arm] = fitsfile[f'{arm}_WAVELENGTH'].read()
                    flux[arm] = fitsfile[f'{arm}_FLUX'].read(rows=quasar_indices)[sort_idx]
                    ivar[arm] = fitsfile[f'{arm}_IVAR'].read(rows=quasar_indices)[sort_idx]
                    mask[arm] = fitsfile[f'{arm}_MASK'].read(rows=quasar_indices)[sort_idx]
                    reso[arm] = fitsfile[f'{arm}_RESOLUTION'].read(rows=quasar_indices)[sort_idx]

                fitsfile.close()

                for idx in range(quasar_indices.size):
                    z_qso = cat_by_survey['Z'] # This is wrong
                    targetid = cat_by_survey['TARGETID']

                    spectra_list.append(Spectrum(z_qso, targetid, wave, flux, ivar, mask, reso, idx))



    # Continuum fitting
        # Initialize global functions
        # For each forest fit continuum
        # Stack all spectra in each process
        # Broadcast and recalculate global functions
        # Iterate

    # Save deltas

