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
