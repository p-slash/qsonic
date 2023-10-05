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
from qsonic.mpi_utils import mpi_parse
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
        "--min-rsnr", type=float, default=0,
        help="Minium SNR <F/sigma> above Lya.")
    analysis_group.add_argument(
        "--min-forestsnr", type=float, default=0,
        help="Minium SNR <F/sigma> within the forest.")
    analysis_group.add_argument(
        "--skip", type=qsonic.io._float_range(0, 1), default=0.2,
        help="Skip short spectra lower than given ratio.")

    parser = qsonic.spectrum.add_wave_region_parser(parser)
    parser = qsonic.masks.add_mask_parser(parser)
    parser = add_picca_continuum_parser(parser)

    return parser


def args_logic_fnc_qsonic_fit(args):
    args_pass = True

    args.arms = list(set(args.arms))

    if args.true_continuum:
        if not args.mock_analysis:
            logging.error(
                "True continuum is only applicable to mock analysis.")
            args_pass = False

        if not args.fiducial_meanflux:
            logging.error(
                "True continuum analysis requires fiducial mean flux.")
            args_pass = False

        if not args.fiducial_varlss:
            logging.error("True continuum analysis requires fiducial var_lss.")
            args_pass = False

    if args.wave2 <= args.wave1:
        logging.error("wave2 must be greater than wave1.")
        args_pass = False

    if args.forest_w2 <= args.forest_w1:
        logging.error("forest_w2 must be greater than forest_w1.")
        args_pass = False

    if args.mock_analysis and args.tile_format:
        logging.error(
            "Mock analysis in tile format is not supported.")
        args_pass = False

    return args_pass


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
    logging.info("Reading spectra.")

    # skip_resomat = args.skip_resomat or not args.mock_analysis

    readerFunction = qsonic.io.get_spectra_reader_function(
        args.input_dir, args.arms, args.mock_analysis, args.skip_resomat,
        args.true_continuum, args.tile_format)

    spectra_list = []
    # Each process reads its own list
    for cat in local_queue:
        local_specs = readerFunction(cat)

        for spec in local_specs:
            spec.set_forest_region(
                args.wave1, args.wave2, args.forest_w1, args.forest_w2)
            spec.remove_nonforest_pixels()

            if not spec.forestwave or spec.rsnr < args.min_rsnr:
                continue

            spectra_list.append(spec)

    if args.coadd_arms == "before":
        logging.info("Coadding arms.")
        for spec in spectra_list:
            spec.coadd_arms_forest()

    nspec_all = comm.reduce(len(spectra_list))
    etime = (time.time() - start_time) / 60  # min
    logging.info(f"All {nspec_all} spectra are read in {etime:.1f} mins.")

    return spectra_list


def mpi_noise_flux_calibrate(spectra_list, args, comm, mpi_rank):
    if args.noise_calibration:
        logging.info("Applying noise calibration.")
        ncal = qsonic.calibration.NoiseCalibrator(
            args.noise_calibration, comm, mpi_rank)
        ncal.apply(spectra_list)

    if args.flux_calibration:
        logging.info("Applying flux calibration.")
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
        logging.info("Reading sky mask.")
        skymasker = qsonic.masks.SkyMask(args.sky_mask)

        maskers.append(skymasker)

    # BAL mask
    if args.bal_mask:
        logging.info("Checking BAL mask.")
        qsonic.masks.BALMask.check_catalog(local_queue[0])

        maskers.append(qsonic.masks.BALMask)

    # DLA mask
    if args.dla_mask:
        logging.info("Reading DLA mask.")
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
    logging.info("Applying masks.")
    for spec in spectra_list:
        for masker in maskers:
            masker.apply(spec)
        spec.drop_short_arms()
    etime = (time.time() - start_time) / 60   # min
    logging.info(f"Masks are applied in {etime:.1f} mins.")


def remove_short_spectra(spectra_list, lya1, lya2, skip_ratio, mpi_rank=0):
    if not skip_ratio:
        return spectra_list

    logging.info("Removing short spectra.")
    dforest_wave = lya2 - lya1
    spectra_list = [spec for spec in spectra_list
                    if spec.is_long(dforest_wave, skip_ratio)]

    return spectra_list


def mpi_continuum_fitting(spectra_list, args, comm, mpi_rank):
    # Initialize continuum fitter & global functions
    logging.info("Initializing continuum fitter.")
    start_time = time.time()
    qcfit = PiccaContinuumFitter(args)

    if not args.true_continuum:
        # Fit continua
        # Stack all spectra in each process
        # Broadcast and recalculate global functions
        # Iterate
        logging.info("Fitting continuum.")
        qcfit.iterate(spectra_list)
    else:
        logging.info("True continuum.")
        qcfit.true_continuum(spectra_list)

    if args.coadd_arms == "after":
        logging.info("Coadding arms.")
        for spec in qsonic.spectrum.valid_spectra(spectra_list):
            spec.coadd_arms_forest(qcfit.varlss_interp, qcfit.eta_interp)
            spec.calc_continuum_chi2()

    qcfit.save_contchi2_catalog(spectra_list)

    # Keep only valid spectra
    spectra_list = list(qsonic.spectrum.valid_spectra(spectra_list))

    # Final cleaning. Especially important if not coadding arms.
    for spec in spectra_list:
        spec.drop_short_arms(args.forest_w1, args.forest_w2, args.skip)

    etime = (time.time() - start_time) / 60  # min
    logging.info(f"Continuum fitting and tweaking took {etime:.1f} mins.")

    return spectra_list


def mpi_run_all(comm, mpi_rank, mpi_size):
    args = mpi_parse(get_parser(), comm, mpi_rank,
                     args_logic_fnc=args_logic_fnc_qsonic_fit)

    if mpi_rank == 0 and args.outdir:
        os_makedirs(args.outdir, exist_ok=True)

    tol = (args.forest_w2 - args.forest_w1) * args.skip
    zmin_qso = args.wave1 / (args.forest_w2 - tol) - 1
    zmax_qso = args.wave2 / (args.forest_w1 + tol) - 1

    # read catalog
    full_catalog = qsonic.catalog.mpi_read_quasar_catalog(
        args.catalog, comm, mpi_rank, is_mock=args.mock_analysis,
        is_tile=args.tile_format, keep_surveys=args.keep_surveys,
        zmin=zmin_qso, zmax=zmax_qso)

    local_queue = qsonic.catalog.mpi_get_local_queue(
        full_catalog, mpi_rank, mpi_size, args.tile_format)

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

    # Remove spectra with respect to forest snr
    spectra_list = [spec for spec in spectra_list
                    if spec.get_effective_meansnr() >= args.min_forestsnr]

    # Create smoothed ivar as intermediate variable
    if args.smoothing_scale > 0:
        for spec in spectra_list:
            spec.set_smooth_forestivar(args.smoothing_scale)

    # Continuum fitting
    spectra_list = mpi_continuum_fitting(spectra_list, args, comm, mpi_rank)

    # Save deltas
    logging.info("Saving deltas.")
    qsonic.io.save_deltas(
        spectra_list, args.outdir,
        save_by_hpx=args.save_by_hpx, mpi_rank=mpi_rank)


def main():
    start_time = time.time()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG if mpi_rank == 0 else logging.CRITICAL)

    try:
        mpi_run_all(comm, mpi_rank, mpi_size)
    except QsonicException as e:
        logging.exception(e)
        exit(1)
    except Exception as e:
        logging.critical(f"Unexpected error on Rank{mpi_rank}. Abort.")
        logging.exception(e)
        comm.Abort()

    etime = (time.time() - start_time) / 60  # min
    logging.info(f"Total time spent is {etime:.1f} mins.")
