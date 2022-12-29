import logging
import argparse
import time

from os import makedirs as os_makedirs

import numpy as np
from mpi4py import MPI

import qcfitter.catalog
import qcfitter.spectrum
import qcfitter.masks
from qcfitter.mpi_utils import balance_load, logging_mpi
from qcfitter.picca_continuum import PiccaContinuumFitter


def parse():
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir", required=True,
        help="Input directory to healpix")
    parser.add_argument(
        "--catalog", required=True,
        help="Catalog filename")
    parser.add_argument(
        "--keep-surveys", nargs='+', default=['sv3', 'main'],
        help="Surveys to keep.")
    parser.add_argument(
        "--min-rsnr", type=float, default=0.,
        help="Minium SNR <F/sigma> above Lya.")
    parser.add_argument(
        "--outdir", '-o',
        help="Output directory to save deltas.")

    parser.add_argument(
        "--mock-analysis", action="store_true",
        help="Input folder is mock. Uses nside=16")
    parser.add_argument(
        "--skip-resomat", action="store_true",
        help="Skip reading resolution matrix for 3D.")
    parser.add_argument(
        "--arms", default=['B', 'R'], nargs='+',
        help="Arms to read.")

    parser.add_argument(
        "--wave1", type=float, default=3600.,
        help="First observed wavelength edge.")
    parser.add_argument(
        "--wave2", type=float, default=6000.,
        help="Last observed wavelength edge.")
    parser.add_argument(
        "--forest-w1", type=float, default=1040.,
        help="First forest wavelength edge.")
    parser.add_argument(
        "--forest-w2", type=float, default=1200.,
        help="Last forest wavelength edge.")
    parser.add_argument(
        "--rfdwave", type=float, default=0.8,
        help="Rest-frame wave steps. Complies with forest limits")
    parser.add_argument(
        "--fiducials", help="Fiducial mean flux and var_lss fits file.")

    parser.add_argument(
        "--sky-mask",
        help="Sky mask file.")
    parser.add_argument(
        "--bal-mask", action="store_true",
        help="Mask BALs (assumes it is in catalog).")
    parser.add_argument(
        "--dla-mask",
        help="DLA catalog to mask.")

    parser.add_argument(
        "--skip", type=float, default=0.,
        help="Skip short spectra lower than given ratio.")
    parser.add_argument(
        "--no-iterations", type=int, default=5,
        help="Number of iterations for continuum fitting.")
    parser.add_argument(
        "--keep-nonforest-pixels", action="store_true",
        help="Keeps non forest wavelengths. Memory intensive!")

    parser.add_argument(
        "--coadd-arms", action="store_true",
        help="Coadds arms when saving.")
    parser.add_argument(
        "--save-by-hpx", action="store_true",
        help="Save by healpix. If not, saves by MPI rank.")

    args = parser.parse_args()

    return args


def mpi_parse(comm, mpi_rank):
    if mpi_rank == 0:
        try:
            args = parse()

            if any(arm not in ['B', 'R', 'Z'] for arm in args.arms):
                logging.error("Arms should be 'B', 'R' or 'Z'.")
                exit(0)

            if args.skip < 0 or args.skip > 1:
                logging.error("Skip ratio should be between 0 and 1")
                exit(0)

        except SystemExit:
            args = -1
    else:
        args = -1

    args = comm.bcast(args)
    if args == -1:
        exit(0)

    return args


def read_masks(comm, local_queue, args, mpi_rank):
    maskers = []
    if args.sky_mask:
        logging_mpi("Reading sky mask.", mpi_rank)
        try:
            skymasker = qcfitter.masks.SkyMask(args.sky_mask)
        except Exception as e:
            logging_mpi(f"{e}", mpi_rank, "error")
            comm.Abort()

        maskers.append(skymasker)

    # BAL mask
    if args.bal_mask:
        logging_mpi("Checking BAL mask.", mpi_rank)
        try:
            qcfitter.masks.BALMask.check_catalog(local_queue[0])
        except Exception as e:
            logging_mpi(f"{e}", mpi_rank, "error")
            comm.Abort()

        maskers.append(qcfitter.masks.BALMask)

    # DLA mask
    if args.dla_mask:
        logging_mpi("Reading DLA mask.", mpi_rank)
        local_targetids = np.concatenate(
            [cat['TARGETID'] for cat in local_queue])
        # Read catalog
        try:
            dlamasker = qcfitter.masks.DLAMask(
                args.dla_mask, local_targetids, comm, mpi_rank,
                dla_mask_limit=0.8)
        except Exception as e:
            logging_mpi(f"{e}", mpi_rank, "error")
            comm.Abort()

        maskers.append(dlamasker)

    return maskers


def apply_masks(maskers, spectra_list, mpi_rank):
    if maskers:
        start_time = time.time()
        logging_mpi("Applying masks.", mpi_rank)
        for spec in spectra_list:
            for masker in maskers:
                masker.apply(spec)
            spec.drop_short_arms()
        etime = (time.time() - start_time) / 60   # min
        logging_mpi(f"Masks are applied in {etime:.1f} mins.", mpi_rank)


def remove_short_spectra(spectra_list, lya1, lya2, skip_ratio, mpi_rank):
    if skip_ratio > 0:
        logging_mpi("Removing short spectra.", mpi_rank)
        dforest_wave = lya2 - lya1
        spectra_list = [spec for spec in spectra_list
                        if spec.is_long(dforest_wave, skip_ratio)]

    return spectra_list


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    logging.basicConfig(level=logging.DEBUG)

    args = mpi_parse(comm, mpi_rank)

    # read catalog
    n_side = 16 if args.mock_analysis else 64
    local_queue = qcfitter.catalog.read_qso_catalog(
        args.catalog, comm, mpi_rank, mpi_size, n_side, args.keep_surveys)

    # Read masks before data
    maskers = read_masks(comm, local_queue, args, mpi_rank)

    start_time = time.time()
    logging_mpi("Reading spectra.", mpi_rank)
    spectra_list = []
    # Each process reads its own list
    for cat in local_queue:
        local_specs = qcfitter.spectrum.read_spectra(
            cat, args.input_dir, args.arms,
            args.mock_analysis, args.skip_resomat
        )
        for spec in local_specs:
            spec.set_forest_region(
                args.wave1, args.wave2,
                args.forest_w1, args.forest_w2
            )

            if not args.keep_nonforest_pixels:
                spec.remove_nonforest_pixels()

        spectra_list.extend(
            [spec for spec in local_specs if spec.rsnr > args.min_rsnr])

    nspec_all = comm.reduce(len(spectra_list), op=MPI.SUM, root=0)
    etime = (time.time() - start_time) / 60  # min
    logging_mpi(
        f"All {nspec_all} spectra are read in {etime:.1f} mins.", mpi_rank)

    apply_masks(maskers, spectra_list, mpi_rank)

    # remove from sample if no pixels is small
    spectra_list = remove_short_spectra(
        spectra_list, args.forest_w1, args.forest_w2, args.skip, mpi_rank)

    # Create smoothed ivar as intermediate variable
    for spec in spectra_list:
        spec.set_smooth_ivar()

    # Continuum fitting
    # -------------------
    # Initialize continuum fitter & global functions
    logging_mpi("Initializing continuum fitter.", mpi_rank)
    start_time = time.time()
    try:
        qcfit = PiccaContinuumFitter(
            args.forest_w1, args.forest_w2, args.rfdwave,
            args.wave1, args.wave2,
            fiducial_fits=args.fiducials
        )
    except Exception as e:
        logging_mpi(f"{e}", 0, "error")
        comm.Abort()

    logging_mpi("Fitting continuum.", mpi_rank)

    # Fit continua
    # Stack all spectra in each process
    # Broadcast and recalculate global functions
    # Iterate
    qcfit.iterate(spectra_list, args.no_iterations, args.outdir)
    logging_mpi("All continua are fit.", mpi_rank)

    if args.outdir:
        logging_mpi("Saving continuum chi2 catalog.", mpi_rank)
        qcfitter.spectrum.save_contchi2_catalog(spectra_list, args.outdir,
                                                comm, mpi_rank)

    # Keep only valid spectra
    spectra_list = list(qcfitter.spectrum.valid_spectra(spectra_list))
    if args.coadd_arms:
        logging_mpi("Coadding arms.", mpi_rank)
        for spec in spectra_list:
            spec.coadd_arms_forest(qcfit.varlss_interp)

    # Final cleaning. Especially important if not coadding arms.
    for spec in spectra_list:
        spec.drop_short_arms(args.forest_w1, args.forest_w2, args.skip)

    etime = (time.time() - start_time) / 60  # min
    logging_mpi(f"Continuum fitting and tweaking took {etime:.1f} mins.",
                mpi_rank)

    # Save deltas
    if args.outdir:
        logging_mpi("Saving deltas.", mpi_rank)
        os_makedirs(args.outdir, exist_ok=True)
        if args.save_by_hpx:
            out_nside = n_side
            out_by_mpi = None
        else:
            out_nside = None
            out_by_mpi = mpi_rank

        qcfitter.spectrum.save_deltas(
            spectra_list, args.outdir, qcfit.varlss_interp,
            out_nside=out_nside, mpi_rank=out_by_mpi)
