import argparse
import logging
import time

from os import makedirs as os_makedirs

import numpy as np
from mpi4py import MPI

import qcfitter.catalog
import qcfitter.io
import qcfitter.spectrum
import qcfitter.masks
from qcfitter.mpi_utils import logging_mpi, mpi_parse
from qcfitter.picca_continuum import PiccaContinuumFitter


def get_parser(add_help=True):
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(
        add_help=add_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    qcfitter.io.add_io_parser(parser)
    qcfitter.spectrum.add_wave_region_parser(parser)
    qcfitter.masks.add_mask_parser(parser)
    PiccaContinuumFitter.add_parser(parser)

    return parser


def read_spectra_local_queue(local_queue, args):
    """ Read local spectra for the MPI rank. Set forest and observed wavelength
    range.

    Arguments
    ---------
    local_queue: list of ndarray
        Catalog from `qcfitter.spectrum.read_local_qso_catalog`.
    args: Namespace obtained by argparse
        Options passed to script.

    Returns
    ---------
    spectra_list: list of Spectrum
        Spectrum objects for the local MPI rank.
    """
    start_time = time.time()
    logging_mpi("Reading spectra.", mpi_rank)

    spectra_list = []
    # Each process reads its own list
    for cat in local_queue:
        local_specs = qcfitter.io.read_spectra(
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

    return spectra_list


def read_masks(comm, local_queue, args, mpi_rank):
    """ Read and set masking objects. Broadcast from the master process if
    necessary. See `qcfitter.masks` for `SkyMask`, `BALMask` and `DLAMask`.

    Arguments
    ---------
    comm: MPI.COMM_WORLD
        Communication object for broadcasting data.
    local_queue: list of ndarray
        Catalog from `qcfitter.spectrum.read_local_qso_catalog`.
    args: Namespace obtained by argparse
        Options passed to script.
    mpi_rank: int
        Rank of the MPI process

    Returns
    ---------
    maskers: list of Masks
        Mask objects from `qcfitter.masks`.
    """
    maskers = []

    try:
        if args.sky_mask:
            logging_mpi("Reading sky mask.", mpi_rank)
            skymasker = qcfitter.masks.SkyMask(args.sky_mask)

            maskers.append(skymasker)

        # BAL mask
        if args.bal_mask:
            logging_mpi("Checking BAL mask.", mpi_rank)
            qcfitter.masks.BALMask.check_catalog(local_queue[0])

            maskers.append(qcfitter.masks.BALMask)

        # DLA mask
        if args.dla_mask:
            logging_mpi("Reading DLA mask.", mpi_rank)
            local_targetids = np.concatenate(
                [cat['TARGETID'] for cat in local_queue])

            # Read catalog
            dlamasker = qcfitter.masks.DLAMask(
                args.dla_mask, local_targetids, comm, mpi_rank,
                dla_mask_limit=0.8)

            maskers.append(dlamasker)

    except Exception as e:
        logging_mpi(f"{e}", mpi_rank, "error")
        comm.Abort()

    return maskers


def apply_masks(maskers, spectra_list, mpi_rank):
    """ Apply masks in `maskers` to the local `spectra_list`. See
    `qcfitter.masks` for `SkyMask`, `BALMask` and `DLAMask`. Masking is set by
    setting `forestivar=0`. `DLAMask` further corrects for Lya and Lyb damping
    wings. Empty arms are removed after masking.

    Arguments
    ---------
    maskers: list of Masks
        Mask objects from `qcfitter.masks`.
    spectra_list: list of Spectrum
        Spectrum objects for the local MPI rank.
    mpi_rank: int
        Rank of the MPI process
    """
    if not maskers:
        return

    start_time = time.time()
    logging_mpi("Applying masks.", mpi_rank)
    for spec in spectra_list:
        for masker in maskers:
            masker.apply(spec)
        spec.drop_short_arms()
    etime = (time.time() - start_time) / 60   # min
    logging_mpi(f"Masks are applied in {etime:.1f} mins.", mpi_rank)


def remove_short_spectra(spectra_list, lya1, lya2, skip_ratio, mpi_rank):
    if not skip_ratio:
        return spectra_list

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

    args = mpi_parse(get_parser(), comm, mpi_rank)
    if mpi_rank == 0 and args.outdir:
        os_makedirs(args.outdir, exist_ok=True)

    # read catalog
    n_side = 16 if args.mock_analysis else 64
    local_queue = qcfitter.catalog.read_local_qso_catalog(
        args.catalog, comm, mpi_rank, mpi_size, is_mock=args.mock_analysis,
        n_side=n_side, keep_surveys=args.keep_surveys)

    # Read masks before data
    maskers = read_masks(comm, local_queue, args, mpi_rank)

    try:
        spectra_list = read_spectra_local_queue(local_queue, args)
    except Exception as e:
        logging_mpi(f"{e}", 0, "error")
        comm.Abort()

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
        qcfit = PiccaContinuumFitter(args)
    except Exception as e:
        logging_mpi(f"{e}", 0, "error")
        comm.Abort()

    logging_mpi("Fitting continuum.", mpi_rank)

    # Fit continua
    # Stack all spectra in each process
    # Broadcast and recalculate global functions
    # Iterate
    qcfit.iterate(spectra_list)

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
    logging_mpi("Saving deltas.", mpi_rank)
    qcfitter.io.save_deltas(
        spectra_list, args.outdir, qcfit.varlss_interp,
        save_by_hpx=args.save_by_hpx, mpi_rank=mpi_rank)
