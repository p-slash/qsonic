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
        vmin_ai, vmax_ai = spec.catrow['VMIN_CIV_450', 'VMAX_CIV_450']
        vmin_bi, vmax_bi = spec.catrow['VMIN_CIV_2000', 'VMAX_CIV_2000']

        min_velocities = np.array([vmin_ai, vmin_bi])
        max_velocities = np.array([vmax_ai, vmax_bi])
        w = (min_velocities>0) & (max_velocities>0)
        min_velocities = min_velocities[w]
        max_velocities = max_velocities[w]
        num_velocities = min_velocities.size

        if num_velocities == 0:
            return

        mask = np.empty(num_velocities * lines.size,
            dtype=[('wave_min', 'f8'), ('wave_max', 'f8')]
        )
        mask['wave_min'] = np.outer(lines['value'], 1 - max_velocities / BALMask.LIGHT_SPEED).ravel()
        mask['wave_max'] = np.outer(lines['value'], 1 - min_velocities / BALMask.LIGHT_SPEED).ravel()

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





