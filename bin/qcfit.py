import logging
import argparse

import numpy as np
from mpi4py import MPI

from qcfitter.catalog import Catalog
from qcfitter.spectrum import Spectrum, read_spectra
from qcfitter.mpi_utils import balance_load, logging_mpi

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    # Arguments passed to run the script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", help="Input directory to healpix", required=True)
    parser.add_argument("--catalog", help="Catalog filename", required=True)
    parser.add_argument("--mock-analysis", help="Input folder is mock.", action="store_true")
    parser.add_argument("--arms", help="Arms to read.", default=['B', 'R'], nargs='+')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if any(arm not in ['B', 'R', 'Z'] for arm in args.arms):
        logging.error("Arms should be 'B', 'R' or 'Z'.")
        comm.Abort()

    # read catalog
    logging_mpi("Reading catalog", mpi_rank)
    n_side = 16 if args.mock_analysis else 64
    qso_cat = Catalog(args.catalog, n_side, mpi_rank=mpi_rank)

    # We decide forest filename list
    # Group into unique pixels
    unique_pix, s = np.unique(qso_cat.catalog['PIXNUM'], return_index=True)
    split_catalog = np.split(qso_cat.catalog, s[1:])

    # Roughly equal number of spectra
    logging_mpi("Load balancing.", mpi_rank)

    local_queue = balance_load(split_catalog, mpi_size, mpi_rank)

    logging_mpi("Reading spectra.", mpi_rank)

    spectra_list = []
    # Each process reads its own list
    for cat in local_queue:
        spectra_list.extend(read_spectra(cat, args.input_dir, args.arms, args.mock_analysis))

    logging_mpi("All spectra are read.", mpi_rank)

    # Continuum fitting
        # Initialize global functions
        # For each forest fit continuum
        # Stack all spectra in each process
        # Broadcast and recalculate global functions
        # Iterate

    # Save deltas

