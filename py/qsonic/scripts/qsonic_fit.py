import argparse
import logging
import time

from os import makedirs as os_makedirs

import numpy as np

from qsonic import QsonicException
import qsonic.calibration
import qsonic.catalog
import qsonic.io
import qsonic.spectrum
import qsonic.masks
from qsonic.mpi_utils import logging_mpi, mpi_parse
from qsonic.picca_continuum import (
    PiccaContinuumFitter, add_picca_continuum_parser)


def get_parser(add_help=True):
    """Constructs the parser needed for the script.

    Arguments
    ---------
    add_help: bool, default: True
        Add help to parser.

    Returns
    -------
    parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        add_help=add_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = qsonic.io.add_io_parser(parser)
    parser = qsonic.calibration.add_calibration_parser(parser)

    analysis_group = parser.add_argument_group('Analysis options')
    analysis_group.add_argument(
        "--smoothing-scale", default=16., type=float,
        help="Smoothing scale for pipeline noise in A.")
    analysis_group.add_argument(
        "--min-rsnr", type=float, default=0.,
        help="Minium SNR <F/sigma> above Lya.")
    analysis_group.add_argument(
        "--skip", type=qsonic.io._float_range(0, 1), default=0.,
        help="Skip short spectra lower than given ratio.")
    analysis_group.add_argument(
        "--keep-nonforest-pixels", action="store_true",
        help="Keeps non forest wavelengths. Memory intensive!")

    parser = qsonic.spectrum.add_wave_region_parser(parser)
    parser = qsonic.masks.add_mask_parser(parser)
    parser = add_picca_continuum_parser(parser)

    return parser


def mpi_read_spectra_local_queue(local_queue, args, comm, mpi_rank):
    """ Read local spectra for the MPI rank. Set forest and observed wavelength
    range.

    Arguments
    ---------
    local_queue: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Catalog from :func:`qsonic.catalog.mpi_get_local_queue`. Each
        element is a catalog for one healpix.
    args: argparse.Namespace
        Options passed to script.
    comm: MPI.COMM_WORLD
        Communication object for reducing data.
    mpi_rank: int
        Rank of the MPI process

    Returns
    ---------
    spectra_list: list(Spectrum)
        Spectrum objects for the local MPI rank.
    """
    start_time = time.time()
    logging_mpi("Reading spectra.", mpi_rank)

    # skip_resomat = args.skip_resomat or not args.mock_analysis
    skip_resomat = args.skip_resomat

    spectra_list = []
    # Each process reads its own list
    for cat in local_queue:
        local_specs = qsonic.io.read_spectra_onehealpix(
            cat, args.input_dir, args.arms, args.mock_analysis, skip_resomat)

        for spec in local_specs:
            spec.set_forest_region(
                args.wave1, args.wave2, args.forest_w1, args.forest_w2)

            if not args.keep_nonforest_pixels:
                spec.remove_nonforest_pixels()

        spectra_list.extend(
            [spec for spec in local_specs if spec.rsnr > args.min_rsnr])

        # if args.skip_resomat:
        #     continue

        # w = np.array(
        #     [spec.rsnr > args.min_rsnr for spec in local_specs], dtype=bool)
        # nspec = w.sum()

        # # Read resolution matrix hopefully faster
        # spectra_list[-nspec:] = \
        #     qsonic.io.read_resolution_matrices_onehealpix_data(
        #         cat[w], args.input_dir, spectra_list[-nspec:])

    nspec_all = comm.reduce(len(spectra_list))
    etime = (time.time() - start_time) / 60  # min
    logging_mpi(
        f"All {nspec_all} spectra are read in {etime:.1f} mins.", mpi_rank)

    return spectra_list


def mpi_noise_flux_calibrate(spectra_list, args, comm, mpi_rank):
    if args.noise_calibration:
        logging_mpi("Applying noise calibration.", mpi_rank)
        ncal = qsonic.calibration.NoiseCalibrator(
            args.noise_calibration, comm, mpi_rank)
        ncal.apply(spectra_list)

    if args.flux_calibration:
        logging_mpi("Applying flux calibration.", mpi_rank)
        fcal = qsonic.calibration.FluxCalibrator(
            args.flux_calibration, comm, mpi_rank)
        fcal.apply(spectra_list)


def mpi_read_masks(local_queue, args, comm, mpi_rank):
    """ Read and set masking objects. Broadcast from the master process if
    necessary. See :mod:`qsonic.masks` for
    :class:`SkyMask <qsonic.masks.SkyMask>`,
    :class:`BALMask <qsonic.masks.BALMask>` and
    :class:`DLAMask <qsonic.masks.DLAMask>`.

    Arguments
    ---------
    local_queue: list(:external+numpy:py:class:`ndarray <numpy.ndarray>`)
        Catalog from :func:`qsonic.catalog.mpi_get_local_queue`.
    args: argparse.Namespace
        Options passed to script.
    comm: MPI.COMM_WORLD
        Communication object for broadcasting data.
    mpi_rank: int
        Rank of the MPI process

    Returns
    ---------
    maskers: list(Masks)
        Mask objects from `qsonic.masks`.
    """
    maskers = []

    if args.sky_mask:
        logging_mpi("Reading sky mask.", mpi_rank)
        skymasker = qsonic.masks.SkyMask(args.sky_mask)

        maskers.append(skymasker)

    # BAL mask
    if args.bal_mask:
        logging_mpi("Checking BAL mask.", mpi_rank)
        qsonic.masks.BALMask.check_catalog(local_queue[0])

        maskers.append(qsonic.masks.BALMask)

    # DLA mask
    if args.dla_mask:
        logging_mpi("Reading DLA mask.", mpi_rank)
        local_targetids = np.concatenate(
            [cat['TARGETID'] for cat in local_queue])

        # Read catalog
        dlamasker = qsonic.masks.DLAMask(
            args.dla_mask, local_targetids, comm, mpi_rank,
            dla_mask_limit=0.8)

        maskers.append(dlamasker)

    return maskers


def apply_masks(maskers, spectra_list, mpi_rank=0):
    """ Apply masks in ``maskers`` to the local ``spectra_list``.

    See :mod:`qsonic.masks` for
    :class:`SkyMask <qsonic.masks.SkyMask>`,
    :class:`BALMask <qsonic.masks.BALMask>` and
    :class:`DLAMask <qsonic.masks.DLAMask>`. Masking is set by setting
    ``forestivar=0``. :class:`DLAMask <qsonic.masks.DLAMask>` further corrects
    for Lya and Lyb damping wings. Empty arms are removed after masking.

    Arguments
    ---------
    maskers: list(Masks)
        Mask objects from `qsonic.masks`.
    spectra_list: list(Spectrum)
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


def remove_short_spectra(spectra_list, lya1, lya2, skip_ratio, mpi_rank=0):
    if not skip_ratio:
        return spectra_list

    logging_mpi("Removing short spectra.", mpi_rank)
    dforest_wave = lya2 - lya1
    spectra_list = [spec for spec in spectra_list
                    if spec.is_long(dforest_wave, skip_ratio)]

    return spectra_list


def mpi_run_all(comm, mpi_rank, mpi_size):
    args = mpi_parse(get_parser(), comm, mpi_rank)
    if mpi_rank == 0 and args.outdir:
        os_makedirs(args.outdir, exist_ok=True)

    tol = (args.forest_w2 - args.forest_w1) * args.skip
    zmin_qso = args.wave1 / (args.forest_w2 - tol) - 1
    zmax_qso = args.wave2 / (args.forest_w1 + tol) - 1

    # read catalog
    full_catalog = qsonic.catalog.mpi_read_quasar_catalog(
        args.catalog, comm, mpi_rank, is_mock=args.mock_analysis,
        keep_surveys=args.keep_surveys,
        zmin=zmin_qso, zmax=zmax_qso)

    local_queue = qsonic.catalog.mpi_get_local_queue(
        full_catalog, mpi_rank, mpi_size)

    # Blinding
    qsonic.spectrum.Spectrum.set_blinding(full_catalog, args)

    # Read masks before data
    maskers = mpi_read_masks(local_queue, args, comm, mpi_rank)

    spectra_list = mpi_read_spectra_local_queue(
        local_queue, args, comm, mpi_rank)

    mpi_noise_flux_calibrate(spectra_list, args, comm, mpi_rank)

    apply_masks(maskers, spectra_list, mpi_rank)

    # remove from sample if no pixels is small
    spectra_list = remove_short_spectra(
        spectra_list, args.forest_w1, args.forest_w2, args.skip, mpi_rank)

    # Create smoothed ivar as intermediate variable
    if args.smoothing_scale > 0:
        for spec in spectra_list:
            spec.set_smooth_ivar(args.smoothing_scale)

    # Continuum fitting
    # -------------------
    # Initialize continuum fitter & global functions
    logging_mpi("Initializing continuum fitter.", mpi_rank)
    start_time = time.time()
    qcfit = PiccaContinuumFitter(args)
    logging_mpi("Fitting continuum.", mpi_rank)

    # Fit continua
    # Stack all spectra in each process
    # Broadcast and recalculate global functions
    # Iterate
    qcfit.iterate(spectra_list)

    if args.coadd_arms:
        logging_mpi("Coadding arms.", mpi_rank)
        for spec in qsonic.spectrum.valid_spectra(spectra_list):
            spec.coadd_arms_forest(qcfit.varlss_interp, qcfit.eta_interp)

    qcfit.save_contchi2_catalog(spectra_list)

    # Keep only valid spectra
    spectra_list = list(qsonic.spectrum.valid_spectra(spectra_list))

    # Final cleaning. Especially important if not coadding arms.
    for spec in spectra_list:
        spec.drop_short_arms(args.forest_w1, args.forest_w2, args.skip)

    etime = (time.time() - start_time) / 60  # min
    logging_mpi(f"Continuum fitting and tweaking took {etime:.1f} mins.",
                mpi_rank)

    # Save deltas
    logging_mpi("Saving deltas.", mpi_rank)
    qsonic.io.save_deltas(
        spectra_list, args.outdir,
        save_by_hpx=args.save_by_hpx, mpi_rank=mpi_rank)


def main():
    start_time = time.time()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)

    try:
        mpi_run_all(comm, mpi_rank, mpi_size)
    except QsonicException as e:
        logging_mpi(e, mpi_rank, "exception")
    except Exception as e:
        logging.error(f"Unexpected error on Rank{mpi_rank}. Abort.")
        logging.exception(e)
        comm.Abort()

    etime = (time.time() - start_time) / 60  # min
    logging_mpi(f"Total time spent is {etime:.1f} mins.", mpi_rank)
