import logging
import argparse

import numpy as np
from mpi4py import MPI

from qcfit.catalog import Catalog
from qcfit.spectrum import Spectrum, read_spectra
from qcfit.mpi_utils import balance_load

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

    logging.basicConfig(level=logging.DEBUG)

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
        spectra_list.extend(read_spectra(cat, args.input_dir, args.arms))



    # Continuum fitting
        # Initialize global functions
        # For each forest fit continuum
        # Stack all spectra in each process
        # Broadcast and recalculate global functions
        # Iterate

    # Save deltas

