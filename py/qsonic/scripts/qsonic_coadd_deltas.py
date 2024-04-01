import argparse
import glob
import logging

from collections import defaultdict
from multiprocessing import Pool

import fitsio
import healpy

import qsonic.io


def get_parser():
    """Constructs the parser needed for the script.

    Returns
    -------
    parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--input-dirs", "-i", nargs='+', type=str, required=True,
        help=('Directories that have delta-*.fits files. Multiple arguments '
              'are passed with a space')
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        help="Output directory.", required=True)
    parser.add_argument(
        "--nside", type=int, default=8, help='Output healpix nside')
    parser.add_argument(
        "--nest", action="store_true", help="Use NESTED pixelization.")
    parser.add_argument(
        "--nproc", type=int, default=None, help="Number of processes.")
    return parser


def read_onedir_to_dict(directory, forest_dict, nproc):
    """Reads a single directory full of ``delta-*.fits`` files, and coadds them
    into ``forest_dict`` dictionary.

    Arguments
    ---------
    directory: str
        Directory to search for `delta-*.fits`` files.
    forest_dict: dict(Delta)
        Dictionary of Delta objects by ``TARGETID`` as key.
    nproc: int
        Number of processes.

    Returns
    -------
    forest_dict: dict(Delta)
    """
    logging.info(f"Reading {directory}.")
    fnames = glob.glob(f"{directory}/delta-*.fits")
    with Pool(processes=nproc) as pool:
        deltas = pool.map(qsonic.io.read_deltas, fnames)

    logging.info(f"Coadding deltas.")
    for onelist in deltas:
        for delta in onelist:
            if delta.targetid in forest_dict:
                forest_dict[delta.targetid].coadd(delta)
            else:
                forest_dict[delta.targetid] = delta

    return forest_dict


def main():
    args = get_parser().parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG)

    forest_dict = {}
    for directory in args.input_dirs:
        forest_dict = read_onedir_to_dict(directory, forest_dict, args.nproc)

    logging.info(f"Grouping into healpixels.")
    forest_by_hpx = defaultdict(list)
    for targetid, delta in forest_dict.items():
        hpx = healpy.ang2pix(
            args.nside, delta.ra, delta.dec, lonlat=True, nest=args.nest)
        forest_by_hpx[hpx].append(delta)

    logging.info(f"Writing to files.")
    for hpx, list_deltas in forest_by_hpx:
        with fitsio.FITS(
                f"{args.output_dir}/delta-{hpx}.fits", 'rw', clobber=True
        ) as fts:
            for delta in list_deltas:
                delta.write(fts)
