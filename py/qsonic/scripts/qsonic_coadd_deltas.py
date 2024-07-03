import argparse
import functools
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


def read_dirs_to_dict(directories, nproc):
    """Reads all directories full of ``delta-*.fits`` files, and coadds them
    into ``forest_dict`` dictionary.

    Arguments
    ---------
    directories: list(str)
        List of directories to search for `delta-*.fits`` files.
    nproc: int
        Number of processes.

    Returns
    -------
    forest_dict: dict(Delta)
        Dictionary of Delta objects by ``TARGETID`` as key.
    """
    logging.info("Making list of all delta files.")
    fnames = []
    for directory in directories:
        fnames.extend(glob.glob(f"{directory}/delta-*.fits"))

    if not fnames:
        raise Exception("No delta files are found!")

    logging.info(f"There are {len(fnames)} files.")
    with Pool(processes=nproc) as pool:
        deltas = pool.map(qsonic.io.read_deltas, fnames)

    logging.info("Coadding deltas.")
    forest_dict = {}
    for onelist in deltas:
        for delta in onelist:
            if delta.targetid in forest_dict:
                forest_dict[delta.targetid].coadd(delta)
            else:
                forest_dict[delta.targetid] = delta

    return forest_dict


def write_one_hpx(hpx, list_deltas, output_dir):
    """Writes to one healpix delta file.

    Arguments
    ---------
    hpx: int
        Healpixel.
    list_deltas: list(Delta)
        List of delta files.
    output_dir: str
        Output directory.
    """
    with fitsio.FITS(
            f"{output_dir}/delta-{hpx}.fits", 'rw', clobber=True
    ) as fts:
        for delta in list_deltas:
            delta.write(fts)


def main():
    args = get_parser().parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG)

    forest_dict = read_dirs_to_dict(args.input_dirs, args.nproc)

    logging.info("Grouping into healpixels.")
    forest_by_hpx = defaultdict(list)
    for targetid, delta in forest_dict.items():
        hpx = healpy.ang2pix(
            args.nside, delta.ra, delta.dec, lonlat=True, nest=args.nest)
        forest_by_hpx[hpx].append(delta)

    logging.info(
        f"There are {len(forest_by_hpx)} healpixels. Writing to files.")

    _fnc = functools.partial(write_one_hpx, output_dir=args.output_dir)

    with Pool(processes=args.nproc) as pool:
        pool.starmap(_fnc, forest_by_hpx.items())
