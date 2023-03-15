import argparse
import glob
import logging
import time
from os import makedirs as os_makedirs
import warnings

import numpy as np

import qsonic.catalog
import qsonic.io
from qsonic.masks import BALMask
from qsonic.mpi_utils import logging_mpi, mpi_parse
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
        "--outdir", '-o',
        help="Output directory to save files.")
    iogroup.add_argument(
        "--mock-analysis", action="store_true",
        help="Input folder is mock. Uses nside=16")
    iogroup.add_argument(
        "--keep-surveys", nargs='+', default=['sv3', 'main'],
        help="Surveys to keep.")
    iogroup.add_argument("--fbase", help="Basename", default="qsonic-eta-fits")
    iogroup.add_argument(
        "--remove-bal-qsos", type="str",
        help="Catalog to remove BAL sightlines.")
    iogroup.add_argument(
        "--remove-targetid-list",
        help="Text file with TARGETIDs to exclude from analysis.")

    vargroup = parser.add_argument_group(
        'Variance fitting parameters')
    vargroup.add_argument(
        "--nvarbins", help="Number of variance bins (logarithmically spaced)",
        default=100, type=int)
    vargroup.add_argument(
        "--nwbins", help="Number of wavelength bins.",
        default=20, type=int)
    vargroup.add_argument(
        "--var1", help="Lower variance bin", default=1e-5, type=float)
    vargroup.add_argument(
        "--var2", help="Upper variance bin", default=2., type=float)
    vargroup.add_argument(
        "--min-snr", help="Minimum SNR of the forest", default=0, type=float)
    vargroup.add_argument(
        "--max-snr", help="Maximum SNR of the forest", default=100, type=float)

    parser = add_wave_region_parser(parser)

    return parser


def set_targetid_list_to_remove(args, comm, mpi_rank):
    ids_to_remove = None
    if mpi_rank == 0 and args.remove_targetid_list:
        try:
            ids_to_remove = np.loadtxt(args.remove_targetid_list, dtype=int)
        except Exception as e:
            logging_mpi(f"{e}", 0, "error")
            ids_to_remove = None

    ids_to_remove = comm.bcast(ids_to_remove)
    if ids_to_remove is None:
        raise Exception("Error while reading remove_targetid_list.")

    if args.remove_bal_qsos:
        catalog = qsonic.io.mpi_read_qso_catalog(
            args.remove_bal_qsos, comm, mpi_rank, args.mock_analysis,
            args.keep_surveys)
        logging_mpi("Checking BAL mask.", mpi_rank)
        BALMask.check_catalog(catalog)

        sel_ai = (catalog['VMIN_CIV_450'] > 0) & (catalog['VMAX_CIV_450'] > 0)
        sel_bi = (
            (catalog['VMIN_CIV_2000'] > 0) & (catalog['VMAX_CIV_2000'] > 0))
        sel_bal = np.any(sel_ai, axis=1) | np.any(sel_bi, axis=1)
        bal_targetids = catalog['TARGETID'][sel_bal]
        ids_to_remove = np.concatenate((ids_to_remove, bal_targetids))
        logging_mpi(f"Removing {bal_targetids.size} BAL sightlines", mpi_rank)

    if args.keep_surveys:
        sel_survey = np.isin(catalog['SURVEY'], args.keep_surveys)
        remove_sur_tids = catalog['TARGETID'][~sel_survey]
        logging_mpi(
            f"Removing {remove_sur_tids.size} non survey sightlines", mpi_rank)
        ids_to_remove = np.concatenate((ids_to_remove, remove_sur_tids))

    return ids_to_remove


def mpi_run_all(comm, mpi_rank, mpi_size):
    args = mpi_parse(get_parser(), comm, mpi_rank)

    ids_to_remove = set_targetid_list_to_remove(args, comm, mpi_rank)

    all_delta_files = None
    if mpi_rank == 0:
        all_delta_files = glob.glob(f"{args.input_dir}/delta-*.fits*")

    all_delta_files = comm.bcast(all_delta_files)

    if not all_delta_files:
        raise Exception(f"Delta files are not found in {args.input_dir}.")

    logging_mpi(f"There are {len(all_delta_files)} delta files.")
    if mpi_size > len(all_delta_files):
        warnings.warn(
            "There are more MPI processes then number of delta files.")

    nfiles_per_rank = max(1, len(all_delta_files) // mpi_size)
    i1 = nfiles_per_rank * mpi_rank
    i2 = min(len(all_delta_files), i1 + nfiles_per_rank)
    files_this_rank = all_delta_files[i1:i2]

    deltas_list = [qsonic.io.read_deltas(fname) for fname in files_this_rank]

    varfitter = VarLSSFitter(
        args.wave1, args.wave2, args.nwbins,
        args.var1, args.var2, args.nvarbins,
        nsubsamples=100)


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)

    try:
        mpi_run_all(comm, mpi_rank, mpi_size)
    except Exception as e:
        logging_mpi(f"{e}", mpi_rank, "error")
        return 1

    return 0
