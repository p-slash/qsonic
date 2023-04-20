""" Calibration module for noise and flux. """
import argparse

import fitsio
import numpy as np

from qsonic import QsonicException
from qsonic.mathtools import Fast1DInterpolator


def add_calibration_parser(parser=None):
    """ Adds calibration related arguments to parser. These arguments are
    grouped under 'Noise and flux calibation options'.

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

    calib_group = parser.add_argument_group(
        'Noise and flux calibation options')

    calib_group.add_argument(
        "--noise-calibration", help="Noise calibration file.")
    calib_group.add_argument(
        "--flux-calibration", help="Flux calibration file.")

    return parser


class NoiseCalibrator():
    """ Noise calibration object.

    .. math:: i \\rightarrow i / \\eta,

    where i is IVAR.

    FITS file must have 'VAR_FUNC' extension. This extension columns for 'wave'
    and 'eta'. Wavelength array must be linearly and equally spaced.

    Parameters
    ----------
    fname: str
        Filename to read by ``fitsio``.
    """

    def __init__(self, fname):
        try:
            with fitsio.FITS(fname) as fts:
                data = fts['VAR_FUNC'].read()

            waves = data['wave']
            waves_0 = waves[0]
            dwave = waves[1] - waves[0]

            if not np.allclose(np.diff(waves), dwave):
                raise Exception(
                    "Failed to construct noise calibration from "
                    f"{fname}::wave is not equally spaced.")

            eta = np.array(data['eta'], dtype='d')
            eta[eta == 0] = 1
            self.eta_interp = Fast1DInterpolator(waves_0, dwave, eta)
        except Exception as e:
            raise QsonicException(
                f"Error loading NoiseCalibrator from file {fname}.") from e

    def apply(self, spectra_list):
        """ Apply the noise calibration by **only** scaling
        :attr:`forestivar <qsonic.spectrum.Spectrum.forestivar>`. Smooth
        component must be set after this.

        Arguments
        ----------
        spec: Spectrum
            Spectrum object to mask.
        """
        for spec in spectra_list:
            for arm, wave_arm in spec.forestwave.items():
                eta = self.eta_interp(wave_arm)
                spec.forestivar /= eta


class FluxCalibrator():
    """ Flux calibration object.

    .. math::

        f &\\rightarrow f / s

        i &\\rightarrow i \\times s^2,

    where i is IVAR and s is the stacked flux.

    FITS file must have 'STACKED_FLUX' extension. This extension columns for
    'wave' and 'stacked_flux'. Wavelength array must be linearly and equally
    spaced.

    Parameters
    ----------
    fname: str
        Filename to read by ``fitsio``.
    """

    def __init__(self, fname):
        try:
            with fitsio.FITS(fname) as fts:
                data = fts['STACKED_FLUX'].read()

            waves = data['wave']
            waves_0 = waves[0]
            dwave = waves[1] - waves[0]

            if not np.allclose(np.diff(waves), dwave):
                raise Exception(
                    "Failed to construct flux calibration from "
                    f"{fname}::wave is not equally spaced.")

            stacked_flux = np.array(data['stacked_flux'], dtype='d')
            stacked_flux[stacked_flux == 0] = 1
            self.flux_interp = Fast1DInterpolator(waves_0, dwave, stacked_flux)
        except Exception as e:
            raise QsonicException(
                f"Error loading FluxCalibrator from file {fname}.") from e

    def apply(self, spectra_list):
        """ Apply the flux calibration by **only** scaling
        :attr:`forestflux <qsonic.spectrum.Spectrum.forestflux>` and
        :attr:`forestivar <qsonic.spectrum.Spectrum.forestivar>`. Smooth
        component must be set after this.

        Arguments
        ----------
        spec: Spectrum
            Spectrum object to mask.
        """
        for spec in spectra_list:
            for arm, wave_arm in spec.forestwave.items():
                stacked_flux = self.stacked_flux_interp(wave_arm)
                spec.forestflux /= stacked_flux
                spec.forestivar *= stacked_flux**2
