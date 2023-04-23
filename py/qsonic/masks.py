""" Masking module with classes for sky, BAL and DLA."""
import argparse

from astropy.io.ascii import read as asread
import numpy as np
from numpy.lib.recfunctions import rename_fields
import fitsio

from qsonic import QsonicException
from qsonic.mpi_utils import mpi_fnc_bcast


LIGHT_SPEED = 299792.458
"""float: Speed of light in km/s."""
sqrt_pi = 1.77245385091
"""float: Square root of pi."""
sqrt_2 = 1.41421356237
"""float: Square root of 2."""


def add_mask_parser(parser=None):
    """ Adds masking related arguments to parser. These arguments are grouped
    under 'Masking options'.

    Arguments
    ---------
    parser: argparse.ArgumentParser, default: None

    Returns
    ---------
    parser: argparse.ArgumentParser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    mask_group = parser.add_argument_group('Masking options')

    mask_group.add_argument(
        "--sky-mask",
        help="Sky mask file.")
    mask_group.add_argument(
        "--bal-mask", action="store_true",
        help="Mask BALs (assumes it is in catalog).")
    mask_group.add_argument(
        "--dla-mask",
        help="DLA catalog to mask.")

    return parser


class SkyMask():
    """ Sky line masking object.

    Parameters
    ----------
    fname: str
        Filename to read by `astropy.io.ascii`. Must have four columns ordered
        as type, minimum wavelength, maximum wavelength and frame (must be 'RF'
        or 'OBS').
    comm: None or MPI.COMM_WORLD, default: None
    mpi_rank: int, default: 0
    """
    column_names = ('type', 'wave_min', 'wave_max', 'frame')
    """tuple: Assumed ordering of columns in the text file."""

    def __init__(self, fname, comm=None, mpi_rank=0):
        mask = mpi_fnc_bcast(
            asread, comm, mpi_rank,
            f"Error loading SkyMask from mask file {fname}.",
            fname, names=SkyMask.column_names)

        self.mask_rest_frame = mask[mask['frame'] == 'RF']
        self.mask_obs_frame = mask[mask['frame'] == 'OBS']

    def apply(self, spec):
        """ Apply the mask by setting **only** ``spec.forestivar`` to zero.

        Arguments
        ----------
        spec: Spectrum
            Spectrum object to mask.
        """
        for arm, wave_arm in spec.forestwave.items():
            w = np.zeros(wave_arm.size, dtype=bool)

            m1 = np.searchsorted(
                wave_arm,
                [self.mask_obs_frame['wave_min'],
                 self.mask_obs_frame['wave_max']]
            ).T

            m2 = np.searchsorted(
                wave_arm / (1.0 + spec.z_qso),
                [self.mask_rest_frame['wave_min'],
                 self.mask_rest_frame['wave_max']]
            ).T

            mask_idx_ranges = np.concatenate((m1, m2))
            for idx1, idx2 in mask_idx_ranges:
                w[idx1:idx2] = 1

            spec.forestivar[arm][w] = 0


class BALMask():
    """ BAL masking object.

    Does not need construction. Assumes BAL related columns are present in the
    catalog.
    """
    lines = np.array([
        ("lCIV", 1549),
        ("lSiIV1", 1394),
        ("lSiIV2", 1403),
        ("lNV", 1240.81),
        ("lLya", 1216.1),
        ("lCIII", 1175),
        ("lPV1", 1117),
        ("lPV2", 1128),
        ("lSIV1", 1062),
        ("lSIV2", 1074),
        ("lLyb", 1020),
        ("lOIV", 1031),
        ("lOVI", 1037),
        ("lOI", 1039)],
        dtype=[("name", "U10"), ("value", 'f8')])
    """:external+numpy:py:class:`ndarray <numpy.ndarray>`: Ion transition
    wavelengths in A."""

    expected_columns = [
        'VMIN_CIV_450', 'VMAX_CIV_450',
        'VMIN_CIV_2000', 'VMAX_CIV_2000'
    ]
    """list(str): Columns needed in catalog to mask wavelength ranges."""

    @staticmethod
    def check_catalog(catalog):
        """ Asserts if the required columns are present in the catalog.

        Arguments
        ----------
        catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
        """
        if not all(col in catalog.dtype.names
                   for col in BALMask.expected_columns):
            raise QsonicException("Input catalog is missing BAL columns.")

    @staticmethod
    def apply(spec):
        """ Apply the mask by setting **only** ``spec.forestivar`` to zero.

        Arguments
        ----------
        spec: Spectrum
            Spectrum object to mask.
        """
        min_velocities = np.concatenate(
            (spec.catrow['VMIN_CIV_450'], spec.catrow['VMIN_CIV_2000']))
        max_velocities = np.concatenate(
            (spec.catrow['VMAX_CIV_450'], spec.catrow['VMAX_CIV_2000']))
        w = (min_velocities > 0) & (max_velocities > 0)
        min_velocities = min_velocities[w]
        max_velocities = max_velocities[w]
        num_velocities = min_velocities.size

        if num_velocities == 0:
            return

        bal_obs_lines = BALMask.lines['value'] * (1 + spec.z_qso)
        min_velocities = 1 - min_velocities / LIGHT_SPEED
        max_velocities = 1 - max_velocities / LIGHT_SPEED

        mask = np.empty(num_velocities * BALMask.lines.size,
                        dtype=[('wave_min', 'f8'), ('wave_max', 'f8')])
        mask['wave_min'] = np.outer(bal_obs_lines, max_velocities).ravel()
        mask['wave_max'] = np.outer(bal_obs_lines, min_velocities).ravel()

        for arm, wave_arm in spec.forestwave.items():
            w = np.zeros(wave_arm.size, dtype=bool)

            mask_idx_ranges = np.searchsorted(
                wave_arm,
                [mask['wave_min'], mask['wave_max']]
            ).T
            # Make sure first index comes before the second
            mask_idx_ranges.sort(axis=1)
            for idx1, idx2 in mask_idx_ranges:
                w[idx1:idx2] = 1

            spec.forestivar[arm][w] = 0


class DLAMask():
    """ DLA masking object.

    Maximum numbers for oscillator strengths and Einstein coefficients are
    picked from NIST.

    Parameters
    ----------
    fname: str
        FITS filename to read.
    local_targetids: :class:`ndarray <numpy.ndarray>` or None, default: None
        Remove DLAs if they are present in these TARGETIDs.
    comm: None or MPI.COMM_WORLD, default: None
    mpi_rank: int, default: 0
    dla_mask_limit: float, default: 0.8
    """
    qe = 4.803204e-10
    """float: Charge of electron in statC (cm^3/2 g^1/2 s^-1)."""
    me = 9.109384e-28
    """float: Mass of electron in g."""
    c_cms = LIGHT_SPEED * 1e5
    """float: Speed of light in cm/s."""
    aij_coeff = np.pi * qe**2 / me / c_cms
    """float: Derived quantity for transition strength in cm^2 s^-1."""

    wave_lya_A = 1215.67
    """float: Lya wavelength in A."""

    # I suppose we are to pick maximum for each from NIST?
    f12_lya = 4.1641e-01
    """float: Lya oscillator strength."""
    A12_lya = 6.2648e+08
    """float: Lya Einstein coefficient."""

    wave_lyb_A = 1025.7220
    """float: Lyb wavelength in A."""
    f12_lyb = 7.9142e-02
    """float: Lyb oscillator strength."""
    A12_lyb = 1.6725e+08
    """float: Lyb Einstein coefficient."""

    accepted_zcolnames = ["Z_DLA", "Z"]
    """list(str): Column names for the DLA redshift."""

    @staticmethod
    def H_tepper_garcia(a, u):
        """ Tepper-Garcia H function.

        Arguments
        ---------
        a: float
            Gamma / sigma.
        u: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            (wave / lambda_12 - 1) / sigma.

        Returns
        -------
        :external+numpy:py:class:`ndarray <numpy.ndarray>`
        """
        P = u**2 + 1e-12
        Q = 1.5 / P
        R = np.exp(-P)
        corr = (R**2 * (4 * P**2 + 7 * P + 4 + Q) - Q - 1)
        corr *= a / P / sqrt_pi
        return R - corr

    @staticmethod
    def voigt_tepper_garcia(x, sigma, gamma):
        """Tepper-Garcia voight function.

        Arguments
        ---------
        x: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            wave / lambda_12 - 1.
        sigma: float
            b / c.
        gamma: float

        Returns
        -------
        :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Voight profile.
        """
        a = gamma / sigma
        u = x / sigma
        return DLAMask.H_tepper_garcia(a, u)

    @staticmethod
    def get_optical_depth(wave_A, lambda12_A, log10N, b, f12, A12):
        """Optical depth for a transition line.

        Arguments
        ---------
        wave_A: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Wavelength array in A.
        lambda12_A: float
            Transition wavelength in A.
        log10N: float
            Log10 of column density.
        b: float
            Doppler parameter in km/s.
        f12: float
            Oscillator strength.
        A12: float
            Einstein coefficient.

        Returns
        -------
        tau: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Optical depth.
        """
        a12 = DLAMask.aij_coeff * f12
        gamma = A12 * lambda12_A * 1e-8 / 4 / np.pi / DLAMask.c_cms

        sigma_gauss = b / LIGHT_SPEED

        tau = DLAMask.voigt_tepper_garcia(
            wave_A / lambda12_A - 1, sigma_gauss, gamma)
        tau *= a12 * 10**(log10N - 13) * (lambda12_A / b) / sqrt_pi

        return tau

    @staticmethod
    def get_dla_flux(wave, z_dla, nhi, b=10.):
        """Normalized flux from Lya and Lyb wings of DLA.

        Arguments
        ---------
        wave: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Wavelength array in A.
        z_dla: float
            DLA redshift
        nhi: float
            Log10 of DLA column density.
        b: float, default: 10
            Doppler parameter in km/s.

        Returns
        -------
        F: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Normalized flux for DLA.
        """
        wave_rf = wave / (1 + z_dla)
        tau = DLAMask.get_optical_depth(
            wave_rf, DLAMask.wave_lya_A, nhi, b,
            DLAMask.f12_lya, DLAMask.A12_lya)
        tau += DLAMask.get_optical_depth(
            wave_rf, DLAMask.wave_lyb_A, nhi, b,
            DLAMask.f12_lyb, DLAMask.A12_lyb)
        return np.exp(-tau)

    @staticmethod
    def get_all_dlas(wave, spec_dlas):
        """Normalized flux from all DLAs in a sightline.

        Arguments
        ---------
        wave: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Wavelength array in A.
        spec_dlas: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Named ndarray with 'Z_DLA' and 'NHI'.

        Returns
        -------
        F: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            Normalized flux.
        """
        transmission = np.ones(wave.size)
        for z_dla, nhi in spec_dlas[['Z_DLA', 'NHI']]:
            transmission *= DLAMask.get_dla_flux(wave, z_dla, nhi)

        return transmission

    @staticmethod
    def _read_catalog(fname):
        """ Read and return DLA catalog.

        Should be run on master. If error occurs, returns None.

        Arguments
        ---------
        fname: str
            FITS filename to read.

        Returns
        -------
        catalog: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            DLA catalog.
        """
        z_colname = DLAMask.accepted_zcolnames[0]
        fts = fitsio.FITS(fname)

        fts_colnames = set(fts["DLACAT"].get_colnames())
        z_colname = fts_colnames.intersection(DLAMask.accepted_zcolnames)

        if not z_colname:
            fts.close()
            raise ValueError(
                "DLA mask error::Z colname has to be one of "
                f"{', '.join(DLAMask.accepted_zcolnames)}")

        z_colname = z_colname.pop()
        columns_list = ["TARGETID", z_colname, "NHI"]
        catalog = fts['DLACAT'].read(columns=columns_list)

        if z_colname != 'Z_DLA':
            catalog = rename_fields(catalog, {z_colname: 'Z_DLA'})

        fts.close()

        return catalog

    def __init__(
            self, fname, local_targetids=None, comm=None, mpi_rank=0,
            dla_mask_limit=0.8):
        self.dla_mask_limit = dla_mask_limit

        catalog = mpi_fnc_bcast(
            DLAMask._read_catalog, comm, mpi_rank,
            f"Error loading DLAMask from file {fname}.",
            fname)

        if local_targetids is not None:
            w = np.isin(catalog['TARGETID'], local_targetids)
            catalog = catalog[w]
        catalog.sort(order='TARGETID')

        # Group DLA catalog into targetids
        self.unique_targetids, s = np.unique(
            catalog['TARGETID'], return_index=True)
        self.split_catalog = np.split(catalog, s[1:])

    def apply(self, spec):
        """ Apply the mask by setting **only** ``spec.forestivar`` to zero.

        Arguments
        ----------
        spec: Spectrum
            Spectrum object to mask.
        """
        w = np.nonzero(self.unique_targetids == spec.targetid)[0]
        if w.size == 0:
            return

        idx = w[0]
        spec_dlas = self.split_catalog[idx]
        for arm, wave_arm in spec.forestwave.items():
            transmission = DLAMask.get_all_dlas(wave_arm, spec_dlas)
            w = transmission < self.dla_mask_limit
            transmission[w] = 1

            spec.forestivar[arm][w] = 0
            spec.forestflux[arm][w] = 0
            spec.forestflux[arm] /= transmission
            spec.forestivar[arm] *= transmission**2
