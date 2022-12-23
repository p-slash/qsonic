from astropy.io.ascii import read as asread
import numpy as np
from numpy.lib.recfunctions import rename_fields
import fitsio

class SkyMask():
    def __init__(self, fname):
        try:
            mask = asread(fname, names=('type', 'wave_min', 'wave_max', 'frame'))

            self.mask_rest_frame = mask[mask['frame'] == 'RF']
            self.mask_obs_frame = mask[mask['frame'] == 'OBS']
        except:
            raise Exception(f"Error loading SkyMask from mask file {fname}.")

    def apply(self, spec):
        for arm, wave_arm in spec.forestwave.items():
            w = np.ones(wave_arm.size, dtype=bool)

            m1 = np.searchsorted(
                wave_arm,
                [self.mask_obs_frame['wave_min'],
                self.mask_obs_frame['wave_max']]
            ).T

            m2 = np.searchsorted(
                wave_arm/(1.0 + spec.z_qso),
                [self.mask_rest_frame['wave_min'],
                self.mask_rest_frame['wave_max']]
            ).T

            mask_idx_ranges = np.concatenate((m1, m2))
            for idx1, idx2 in mask_idx_ranges:
                w[idx1:idx2] = 0

            spec.forestivar[arm][~w] = 0

class BALMask():
    # Wavelengths in Angstroms
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
        ("lOI", 1039),
        ], dtype=[("name", "U10"), ("value", 'f8')])
    LIGHT_SPEED = 299792.458

    expected_columns = [
        'VMIN_CIV_450', 'VMAX_CIV_450',
        'VMIN_CIV_2000', 'VMAX_CIV_2000'
    ]

    @staticmethod
    def check_catalog(catalog):
        if not all(col in catalog.dtype.names for col in BALMask.expected_columns):
            raise Exception("Input catalog is missing BAL columns.")

    @staticmethod
    def apply(spec):
        # vmin_ai, vmax_ai = spec.catrow['VMIN_CIV_450', 'VMAX_CIV_450']
        # vmin_bi, vmax_bi = spec.catrow['VMIN_CIV_2000', 'VMAX_CIV_2000']

        min_velocities = spec.catrow['VMIN_CIV_450', 'VMIN_CIV_2000']
        max_velocities = spec.catrow['VMAX_CIV_450', 'VMAX_CIV_2000']
        w = (min_velocities>0) & (max_velocities>0)
        min_velocities = min_velocities[w]
        max_velocities = max_velocities[w]
        num_velocities = min_velocities.size

        if num_velocities == 0:
            return

        min_velocities = 1 - min_velocities / BALMask.LIGHT_SPEED
        max_velocities = 1 - max_velocities / BALMask.LIGHT_SPEED

        mask = np.empty(num_velocities * lines.size,
            dtype=[('wave_min', 'f8'), ('wave_max', 'f8')]
        )
        mask['wave_min'] = np.outer(BALMask.lines['value'], max_velocities).ravel()
        mask['wave_max'] = np.outer(BALMask.lines['value'], min_velocities).ravel()

        for arm, wave_arm in spec.forestwave.items():
            w = np.ones(wave_arm.size, dtype=bool)

            mask_idx_ranges = np.searchsorted(
                wave_arm/(1.0 + spec.z_qso),
                [self.mask['wave_min'], self.mask['wave_max']]
            ).T
            # Make sure first index comes before the second
            mask_idx_ranges.sort(axis=1)
            for idx1, idx2 in mask_idx_ranges:
                w[idx1:idx2] = 0

            spec.forestivar[arm][~w] = 0

class DLAMask():
    LIGHT_SPEED = 299792.458
    qe = 4.803204e-10 # statC, cm^3/2 g^1/2 s^-1
    me = 9.109384e-28 # g
    c_cms = LIGHT_SPEED * 1e5 # km to cm
    aij_coeff = np.pi * qe**2 / me / c_cms # cm^2 s^-1
    sqrt_pi = 1.77245385091
    sqrt_2  = 1.41421356237

    wave_lya_A = 1215.67
    # I suppose we are to pick maximum for each from NIST?
    f12_lya = 4.1641e-01
    A12_lya = 6.2648e+08

    wave_lyb_A = 1025.7220
    f12_lyb = 7.9142e-02
    A12_lyb = 1.6725e+08

    @staticmethod
    def H_tepper_garcia(a, u):
        P = u**2+1e-12
        Q = 1.5/P
        R = np.exp(-P)
        corr = (R**2*(4*P**2 + 7*P + 4 +Q) - Q - 1)*a/P/DLAMask.sqrt_pi
        return R - corr

    @staticmethod
    def voigt_tepper_garcia(x, sigma, gamma):
        a = gamma/sigma
        u = x/sigma
        return DLAMask.H_tepper_garcia(a, u)

    @staticmethod
    def get_optical_depth(wave_A, lambda12_A, log10N, b, f12, A12):
        a12 = DLAMask.aij_coeff * f12
        gamma = A12 * lambda12_A*1e-8 / 4 / np.pi / DLAMask.c_cms

        sigma_gauss = b / DLAMask.LIGHT_SPEED

        vfnc = DLAMask.voigt_tepper_garcia(wave_A/lambda12_A - 1, sigma_gauss, gamma)
        tau = a12 * 10**(log10N-13) * (lambda12_A/b) * vfnc / DLAMask.sqrt_pi

        return tau

    @staticmethod
    def get_dla_flux(wave, z_dla, nhi, b=10.):
        wave_rf = wave/(1+z_dla)
        tau  = DLAMask.get_optical_depth(wave_rf, DLAMask.wave_lya_A, nhi, b, DLAMask.f12_lya, DLAMask.A12_lya)
        tau += DLAMask.get_optical_depth(wave_rf, DLAMask.wave_lyb_A, nhi, b, DLAMask.f12_lyb, DLAMask.A12_lyb)
        return np.exp(-tau)

    @staticmethod
    def get_all_dlas(wave, spec_dlas):
        transmission = np.ones(wave.size)
        for z_dla, nhi in spec_dlas['Z_DLA', 'NHI']:
            transmission *= DLAMask.get_dla_flux(wave, z_dla, nhi)

        return transmission

    def __init__(self, fname, local_targetids, comm, mpi_rank, dla_mask_limit=0.8):
        catalog = None
        self.dla_mask_limit = dla_mask_limit

        if mpi_rank == 0:
            accepted_zcolnames = ["Z_DLA", "Z"]
            z_colname = accepted_zcolnames[0]
            fts = fitsio.FITS(fname)

            fts_colnames = set(fts["DLACAT"].get_colnames())
            z_colname = fts_colnames.intersection(accepted_zcolnames)
            if not z_colname:
                raise ValueError(f"Z colname has to be one of {', '.join(accepted_zcolnames)}")
            z_colname = z_colname.pop()
            columns_list = ["TARGETID", z_colname, "NHI"]
            catalog = fts['DLACAT'].read(columns=columns_list)
            catalog = rename_fields(catalog, {z_colname:'Z_DLA'})

            fts.close()

        catalog = comm.bcast(catalog)
        w = np.isin(catalog['TARGETID'], local_targetids)
        catalog = catalog[w]
        catalog.sort(order='TARGETID')

        # Group DLA catalog into targetids
        self.unique_targetids, s = np.unique(catalog['TARGETID'], return_index=True)
        self.split_catalog = np.split(catalog, s[1:])

    def apply(self, spec):
        w = np.isin(self.unique_targetids, spec.targetid)
        if not any(w):
            return

        idx = np.nonzero(w)[0][0]
        spec_dlas = self.split_catalog[idx]
        for arm, wave_arm in spec.forestwave.items():
            transmission = DLAMask.get_all_dlas(wave_arm, spec_dlas)
            w = transmission < self.dla_mask_limit
            transmission[w] = 1

            spec.forestivar[arm][w] = 0
            spec.forestflux[arm][w] = 0
            spec.forestflux[arm] /= transmission
            spec.forestivar[arm] *= transmission**2

        



            

