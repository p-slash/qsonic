import logging
import argparse

from os import makedirs as os_makedirs

import numpy as np
from mpi4py import MPI

from qcfitter.catalog import Catalog
from qcfitter.spectrum import Spectrum, read_spectra, save_deltas
from qcfitter.mpi_utils import balance_load, logging_mpi
from qcfitter.picca_continuum import PiccaContinuumFitter

def parse():
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", help="Input directory to healpix", required=True)
    parser.add_argument("--catalog", help="Catalog filename", required=True)
    # parser.add_argument("--keep-surveys", help="Surveys to keep.")
    parser.add_argument("--outdir", '-o', help="Output directory to save deltas.")

    parser.add_argument("--mock-analysis", help="Input folder is mock. Uses nside=16",
        action="store_true")
    parser.add_argument("--skip-resomat", help="Skip reading resolution matrix for 3D.",
        action="store_true")
    parser.add_argument("--arms", help="Arms to read.", default=['B', 'R'], nargs='+')

    parser.add_argument("--wave1", help="First analysis wavelength", type=float,
        default=3600.)
    parser.add_argument("--wave2", help="Last analysis wavelength", type=float,
        default=6000.)
    parser.add_argument("--forest-w1", help="First forest wavelength.", type=float,
        default=1040.)
    parser.add_argument("--forest-w2", help="Last forest wavelength.", type=float,
        default=1200.)
    parser.add_argument("--fiducials", help="Fiducial mean flux and var_lss fits file.")

    parser.add_argument("--skip", help="Skip short spectra lower than given ratio.",
        type=float, default=0.)
    parser.add_argument("--rfdwave", help="Rest-frame wave steps", type=float,
        default=1.)
    parser.add_argument("--no-iterations", help="Number of iterations to perform for continuum fitting.",
        type=int, default=5)
    parser.add_argument("--keep-nonforest-pixels", help="Keeps non forest wavelengths. Memory intensive!",
        action="store_true")

    parser.add_argument("--coadd-arms", help="Coadds arms when saving.",
        action="store_true")
    parser.add_argument("--save-by-hpx", help="Save by healpix. If not, saves by MPI rank.",
        action="store_true")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    logging.basicConfig(level=logging.DEBUG)

    if mpi_rank == 0:
        try:
            args = parse()
        except:
            comm.Abort()

        if any(arm not in ['B', 'R', 'Z'] for arm in args.arms):
            logging.error("Arms should be 'B', 'R' or 'Z'.")
            comm.Abort()

        if args.skip < 0 or args.skip > 1:
            logging.error("Skip ratio should be between 0 and 1")
            comm.Abort()

    args = comm.bcast(args)

    # read catalog
    n_side = 16 if args.mock_analysis else 64
    qso_cat = Catalog(args.catalog, comm, n_side=n_side, mpi_rank=mpi_rank)

    # We decide forest filename list
    # Group into unique pixels
    unique_pix, s = np.unique(qso_cat.catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(qso_cat.catalog, s[1:])
    logging_mpi(f"There are {unique_pix.size} healpixels. Using more MPI processes will be unused.", mpi_rank)

    # Roughly equal number of spectra
    logging_mpi("Load balancing.", mpi_rank)
    local_queue = balance_load(split_catalog, mpi_size, mpi_rank)

    logging_mpi("Reading spectra.", mpi_rank)
    spectra_list = []
    # Each process reads its own list
    for cat in local_queue:
        local_specs = read_spectra(
            cat, args.input_dir, args.arms,
            args.mock_analysis, args.skip_resomat
        )
        for spec in local_specs:
            spec.set_forest_region(
                args.wave1, args.wave2,
                args.forest_w1, args.forest_w2,
                args.skip/2
            )

            spec.set_smooth_ivar()

            if not args.keep_nonforest_pixels:
                spec.remove_nonforest_pixels()

        spectra_list.extend(local_specs)

    nspec_all = comm.reduce(len(spectra_list), op=MPI.SUM, root=0)
    logging_mpi(f"All {nspec_all} spectra are read.", mpi_rank)

    # Mask

    # remove from sample if no pixels is small
    if args.skip > 0:
        dforest_wave = args.forest_w2 - args.forest_w1
        _npixels = lambda spec: (1+spec.z_qso)*dforest_wave/spec.dwave
        spectra_list = [spec for spec in spectra_list if spec.get_real_size() > args.skip*_npixels(spec)]

    # Continuum fitting
    # -------------------
    # Initialize continuum fitter & global functions
    try:
        qcfit = PiccaContinuumFitter(
            args.forest_w1, args.forest_w2, args.rfdwave,
            comm, mpi_rank, fiducial_fits=args.fiducials
        )
    except Exception as e:
        logging_mpi(f"{e}", 0, "error")
        comm.Abort()

    logging_mpi("Fitting continuum.", mpi_rank)

    # Fit continua
    # Stack all spectra in each process
    # Broadcast and recalculate global functions
    # Iterate
    qcfit.iterate(spectra_list, args.no_iterations)
    # Keep only valid spectra
    spectra_list = [spec for spec in spectra_list if spec.cont_params['valid']]
    logging_mpi("All continua are fit.", mpi_rank)

    if args.coadd_arms:
        logging_mpi("Coadding arms.", mpi_rank)
        for spec in spectra_list:
            spec.coadd_arms_forest(qcfit.varlss_interp)

    # Save deltas
    if args.outdir:
        logging_mpi("Saving deltas.", mpi_rank)
        os_makedirs(args.outdir, exist_ok=True)
        if args.save_by_hpx:
            out_nside = nside
            out_by_mpi= None
        else:
            out_nside = None
            out_by_mpi= mpi_rank

        save_deltas(spectra_list, args.outdir, qcfit.varlss_interp,
            out_nside=out_nside, mpi_rank=out_by_mpi)





