from astropy.io.ascii import read as asread
import numpy as np

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
        if not all(col in catalog.dtype.names for col in expected_columns):
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
    c_cms = DLAMask.LIGHT_SPEED * 1e5 # km to cm
    aij_coeff = np.pi * DLAMask.qe**2 / DLAMask.me / DLAMask.c_cms # cm^2 s^-1
    sqrt_pi = 1.77245385091
    sqrt_2  = 1.41421356237

    f12_lya = 4.1641e-01
    A12_lya = 6.2648e+08

    wave_lya_A=1215.67

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
    def getOpticalDepth(wave_A, lambda12_A, log10N, b, f12, A12):
        a12 = DLAMask.aij_coeff * f12
        gamma = A12 * lambda12_A*1e-8 / 4 / np.pi / DLAMask.c_cms

        sigma_gauss = b / DLAMask.LIGHT_SPEED

        vfnc = DLAMask.voigt_tepper_garcia(wave_A/lambda12_A - 1, sigma_gauss, gamma)
        tau = a12 * 10**(log10N-13) * (lambda12_A/b) * vfnc / DLAMask.sqrt_pi

        return tau

    @staticmethod
    def getDLAFlux(wave, z_abs, nhi, b=10.):
        wave_rf = wave/(1+z_abs)
        return np.exp(-getOpticalDepth(wave_rf, DLAMask.wave_lya_A, nhi, b, DLAMask.f12_lya, DLAMask.A12_lya))
