import argparse
import glob
import logging
import time
from os import makedirs as os_makedirs
import warnings

import numpy as np

from qsonic import QsonicException
import qsonic.catalog
import qsonic.io
from qsonic.masks import BALMask
from qsonic.mpi_utils import logging_mpi, mpi_parse, mpi_fnc_bcast, MPISaver
from qsonic.picca_continuum import VarLSSFitter
from qsonic.spectrum import add_wave_region_parser


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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Note: Not every option is used in this script.')

    iogroup = parser.add_argument_group(
        'Input/output parameters and selections')
    iogroup.add_argument(
        "--input-dir", '-i', required=True,
        help="Input directory.")
    iogroup.add_argument(
        "--outdir", '-o', required=True,
        help="Output directory to save files.")
    iogroup.add_argument("--fbase", help="Basename", default="qsonic-eta-fits")
    iogroup.add_argument(
        "--catalog",
        help="Catalog filename to enable catalog related removals.")
    iogroup.add_argument(
        "--mock-analysis", action="store_true",
        help="Input folder is mock. Uses nside=16")
    iogroup.add_argument(
        "--keep-surveys", nargs='*',
        help="Surveys to keep. Empty keeps all.")
    iogroup.add_argument(
        "--remove-bal-qsos", action="store_true",
        help="Removes BAL sightlines in the catalog option.")
    iogroup.add_argument(
        "--remove-targetid-list",
        help="Text file with TARGETIDs to exclude from analysis.")

    vargroup = parser.add_argument_group(
        'Variance fitting parameters')
    vargroup.add_argument(
        "--nvarbins", help="Number of variance bins (logarithmically spaced).",
        default=100, type=int)
    vargroup.add_argument(
        "--var-use-cov", action="store_true",
        help="Use covariance in varlss-eta fitting.")
    vargroup.add_argument(
        "--nwbins", default=None, type=int,
        help="Number of wavelength bins. None creates bins with 120 A spacing")
    vargroup.add_argument(
        "--var1", help="Lower variance bin.", default=1e-4, type=float)
    vargroup.add_argument(
        "--var2", help="Upper variance bin.", default=20., type=float)
    vargroup.add_argument(
        "--min-snr", help="Minimum SNR of the forest.",
        default=0, type=float)
    vargroup.add_argument(
        "--max-snr", help="Maximum SNR of the forest.",
        default=100, type=float)

    parser = add_wave_region_parser(parser)

    return parser


def mpi_set_targetid_list_to_remove(args, comm=None, mpi_rank=0):
    """ Return a ndarray of TARGETIDs to remove from the sample.

    Can be used without MPI by passing ``comm=None`` (which is the default.)

    Arguments
    ---------
    args: argparse.Namespace
        Options passed to script.
    comm: MPI.COMM_WORLD or None, default: None
        Communication object broadcast data.
    mpi_rank: int, default: 0
        Rank of the MPI process.

    Returns
    -------
    ids_to_remove: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        TARGETIDs to remove from the sample.

    Raises
    ------
    QsonicException
        If error occurs while reading ``args.remove_targetid_list`` or if
        ``--remove_bal_qsos`` is passed but the input catalog is missing BAL
        columns .
    """
    ids_to_remove = np.array([], dtype=int)
    if args.remove_targetid_list:
        ids_to_remove = mpi_fnc_bcast(
            np.loadtxt, comm, mpi_rank,
            "Error while reading remove_targetid_list.",
            args.remove_targetid_list, dtype=int)

    if args.catalog:
        catalog = qsonic.io.mpi_read_quasar_catalog(
            args.catalog, comm, mpi_rank, args.mock_analysis)
    else:
        catalog = None

    if catalog is not None and args.remove_bal_qsos:
        logging_mpi("Checking BAL mask.", mpi_rank)
        BALMask.check_catalog(catalog)

        sel_ai = (catalog['VMIN_CIV_450'] > 0) & (catalog['VMAX_CIV_450'] > 0)
        sel_bi = (
            (catalog['VMIN_CIV_2000'] > 0) & (catalog['VMAX_CIV_2000'] > 0))
        sel_bal = np.any(sel_ai, axis=1) | np.any(sel_bi, axis=1)
        bal_targetids = catalog['TARGETID'][sel_bal]
        ids_to_remove = np.concatenate((ids_to_remove, bal_targetids))
        logging_mpi(f"Removing {bal_targetids.size} BAL sightlines", mpi_rank)

    if catalog is not None and args.keep_surveys:
        sel_survey = np.isin(catalog['SURVEY'], args.keep_surveys)
        remove_sur_tids = catalog['TARGETID'][~sel_survey]
        logging_mpi(
            f"Removing {remove_sur_tids.size} non survey sightlines", mpi_rank)
        ids_to_remove = np.concatenate((ids_to_remove, remove_sur_tids))

    return ids_to_remove


def mpi_read_all_deltas(args, comm=None, mpi_rank=0, mpi_size=1):
    start_time = time.time()
    logging_mpi("Reading deltas.", mpi_rank)

    all_delta_files = mpi_fnc_bcast(
        glob.glob, comm, mpi_rank,
        f"Delta files are not found in {args.input_dir}.",
        f"{args.input_dir}/delta-*.fits*")

    ndelta_all = len(all_delta_files)
    logging_mpi(f"There are {ndelta_all} delta files.", mpi_rank)
    if mpi_size > ndelta_all:
        warnings.warn(
            "There are more MPI processes then number of delta files.")

    nfiles_per_rank = max(1, ndelta_all // mpi_size)
    i1 = nfiles_per_rank * mpi_rank
    i2 = min(ndelta_all, i1 + nfiles_per_rank)
    files_this_rank = all_delta_files[i1:i2]

    deltas_list = [qsonic.io.read_deltas(fname) for fname in files_this_rank]

    etime = (time.time() - start_time) / 60  # min
    logging_mpi(f"Rank{mpi_rank} read {i2-i1} deltas in {etime:.1f} mins.", 0)

    return deltas_list


def mpi_stack_fluxes(args, comm, deltas_list):
    dwave = deltas_list[0].header['DELTA_LAMBDA']
    nwaveobs = int((args.wave2 - args.wave1) / dwave) + 1
    waveobs = np.arange(nwaveobs) * dwave + args.wave1
    stacked_flux = np.zeros(nwaveobs)
    weights = np.zeros(nwaveobs)

    for delta in deltas_list:
        flux = 1 + delta.delta
        idx = np.round((delta.wave - args.wave1) / dwave).astype(int)
        w = (idx >= 0) & (idx < nwaveobs)
        stacked_flux[idx[w]] += flux[w] * delta.weight[w]
        weights[idx[w]] += delta.weight[w]

    # Save stacked_flux to buffer, then used stacked_flux to store reduced
    # weights. Place them properly in the end.
    buf = np.zeros(nwaveobs)
    comm.Allreduce(stacked_flux, buf)
    stacked_flux *= 0
    comm.Allreduce(weights, stacked_flux)
    weights = stacked_flux
    stacked_flux = buf

    w = weights > 0
    stacked_flux[w] /= weights[w]
    stacked_flux[~w] = 0

    return waveobs, stacked_flux


def mpi_run_all(comm, mpi_rank, mpi_size):
    args = mpi_parse(get_parser(), comm, mpi_rank)
    if mpi_rank == 0:
        os_makedirs(args.outdir, exist_ok=True)

    varfitter = VarLSSFitter(
        args.wave1, args.wave2, args.nwbins,
        args.var1, args.var2, args.nvarbins,
        use_cov=args.var_use_cov, comm=comm)

    ids_to_remove = mpi_set_targetid_list_to_remove(args, comm, mpi_rank)

    def _is_kept(delta):
        return (
            (delta.targetid not in ids_to_remove)
            and (delta.mean_snr > args.min_snr)
            and (delta.mean_snr < args.max_snr)
        )

    deltas_list = mpi_read_all_deltas(args, comm, mpi_rank, mpi_size)
    # Flatten this list of lists and remove quasars
    deltas_list = [x for alist in deltas_list for x in alist if _is_kept(x)]

    for delta in deltas_list:
        varfitter.add(delta.wave, delta.delta, delta.ivar)

    logging_mpi("Fitting variance for VarLSS and eta", mpi_rank)
    fit_results = np.ones((varfitter.nwbins, 2))
    fit_results[:, 0] = 0.1
    fit_results, std_results = varfitter.fit(fit_results)

    # Save variance stats to file
    logging_mpi("Saving variance stats to files", mpi_rank)
    suffix = f"snr{args.min_snr:.1f}-{args.max_snr:.1f}"
    tmpfilename = f"{args.outdir}/{args.fbase}-{suffix}-variance-stats.fits"
    mpi_saver = MPISaver(tmpfilename, mpi_rank)
    varfitter.write(mpi_saver, args.min_snr, args.max_snr)
    logging_mpi(f"Variance stats saved in {tmpfilename}.", mpi_rank)

    # Save fits results as well
    mpi_saver.write([
        varfitter.waveobs, fit_results[:, 0], std_results[:, 0],
        fit_results[:, 1], std_results[:, 1]],
        names=['lambda', 'var_lss', 'e_var_lss', 'eta', 'e_eta'],
        extname="VAR_FUNC"
    )

    waveobs, stacked_flux = mpi_stack_fluxes(args, comm, deltas_list)
    mpi_saver.write(
        [waveobs, stacked_flux],
        names=["lambda", "stacked_flux"],
        extname="STACKED_FLUX")

    mpi_saver.close()


def main():
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
