import logging
import argparse

import numpy as np
from mpi4py import MPI

from qcfitter.catalog import Catalog
from qcfitter.spectrum import Spectrum, read_spectra
from qcfitter.mpi_utils import balance_load, logging_mpi
from qcfitter.continuum import ContinuumFitter

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

    parser.add_argument("--wave1", help="First analysis wavelength", type=float,
        default=3600.)
    parser.add_argument("--wave2", help="Last analysis wavelength", type=float,
        default=6000.)
    parser.add_argument("--forest-w1", help="First forest wavelength.", type=float,
        default=1040.)
    parser.add_argument("--forest-w2", help="Lasr forest wavelength.", type=float,
        default=1200.)

    parser.add_argument("--rfdwave", help="Rest-frame wave steps", type=float,
        default=1.)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if any(arm not in ['B', 'R', 'Z'] for arm in args.arms):
        logging_mpi("Arms should be 'B', 'R' or 'Z'.", mpi_rank, "error")
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

    nspec_all = comm.reduce(len(spectra_list), op=MPI.SUM, root=0)
    logging_mpi(f"All {nspec_all} spectra are read.", mpi_rank)

    logging_mpi("Setting forest region.", mpi_rank)
    for spec in spectra_list:
        spec.set_forest_region(args.wave1, args.wave2, args.forest_w1, args.forest_w2)

    # Continuum fitting
    # -------------------
    # Initialize global functions
    qcfit = ContinuumFitter(args.forest_w1, args.forest_w2, args.rfdwave)
    logging_mpi("Fitting continuum.", mpi_rank)

    no_valid_fits=0
    no_invalid_fits=0
    # For each forest fit continuum
    for spec in spectra_list:
        qcfit.fit_continuum(spec)
        if not spec.cont_params['valid']:
            no_invalid_fits+=1
            logging.error(f"mpi:{mpi_rank}:Invalid continuum TARGETID: {spec.targetid}.")
        else:
            no_valid_fits += 1

    no_valid_fits = comm.reduce(no_valid_fits, op=MPI.SUM, root=0)
    no_invalid_fits = comm.reduce(no_invalid_fits, op=MPI.SUM, root=0)
    logging_mpi(f"Number of valid fits: {no_valid_fits}", mpi_rank)
    logging_mpi(f"Number of invalid fits: {no_invalid_fits}", mpi_rank)
    # Stack all spectra in each process
    # Broadcast and recalculate global functions
    # Iterate

    logging_mpi("All continua are fit.", mpi_rank)
    # Save deltas

