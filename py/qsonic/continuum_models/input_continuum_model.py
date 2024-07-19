import logging
import warnings

import fitsio
import numpy as np

from healpy import ang2pix

from qsonic.mathtools import FastCubic1DInterp
from qsonic.continuum_models.base_continuum_model import BaseContinuumModel


def _find_append_continuum(
        specs, wave_rf_1, dwave_rf, common_targetids, continua
):
    for spec in specs:
        idx = np.nonzero(common_targetids == spec.targetid)[0]
        if idx.size == 0:
            continue

        idx = idx[0]
        spec.cont_params['valid'] = True
        spec.cont_params['input_w1'] = wave_rf_1
        spec.cont_params['input_dwave'] = dwave_rf
        spec.cont_params['input_data'] = continua[idx]


class InputContinuumModel(BaseContinuumModel):
    """Input continuum model class.

    Input continua should be in the rest frame with equal wavelength spacing.
    These continua are then interpolated using a cubic spline. Uses fiducials
    for mean flux and varlss interpolation.

    The files are assumed to be organized by healpix **nside=8** with
    **nested** ordering. For example, ``input-continuum-{nside}-{pixnum}.fits``
    , where
    ``pixnum = healpy.ang2pix(nside, ra_deg, dec_deg, lonlat=True, nest=True)``
    . Each file should have at least two extensions: **FIBERMAP** and
    **CONTINUA**.

    - **FIBERMAP** extension should have the typical DESI fibermap columns:
      ``TARGETID, Z, RA, DEC, PROGRAM, SURVEY``. Each ``TARGETID`` should occur
      once.

    - **CONTINUA** extention should be an ImageHDU and have the quasar continua
      in the same order of ``TARGETID`` s in **FIBERMAP**. Its header should
      declare the initial rest-frame wavelength by ``WAVE1`` key and the
      rest-frame wavelength step by ``DWAVE`` key. The data should be in
      ``(nqso, nwave)`` shape.

    Parameters
    ----------
    input_dir: str
        Input directory for all input-continuum FITS files.
    meanflux_interp: FastLinear1DInterp
        Interpolator for mean flux. If fiducial is not set, this equals to 1.
    varlss_interp: FastLinear1DInterp or FastCubic1DInterp
        Cubic spline for var_lss if fitting. Linear if from file.
    eta_interp: FastCubic1DInterp
        Interpolator for eta.
    nside: int, default: 8
        Healpy nside.
    """

    def __init__(
            self, input_dir, meanflux_interp, varlss_interp, eta_interp,
            nside=8
    ):
        self.input_dir = input_dir
        self.meanflux_interp = meanflux_interp
        self.varlss_interp = varlss_interp
        self.eta_interp = eta_interp
        self.nside = nside

    def _read_continua_one_file(self, pixnum, targetids):
        """Reads one input continuum file.

        Arguments
        ---------
        pixnum: int
            Healpix number
        targetids: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            TARGETIDs in this healpix

        Returns
        -------
        wave_rf_1: float
            First wavelength in the rest frame.
        dwave_rf: float
            Rest-frame wavelength step
        common_targetids: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            TARGETIDs that are found in the file.
        continua: :external+numpy:py:class:`ndarray <numpy.ndarray>`
            2D array of shape ``(common_targetids.size, nwave)``

        Raises
        ------
        Exception
            If the number of quasars in **FIBERMAP** and **CONTINUA** extension
            do not match.
        RuntimeWarning
            If the file cannot be read. In this case, all objects are assumed
            to have invalid continuum.
        """
        fname = f"{self.input_dir}/input-continuum-{self.nside}-{pixnum}.fits"

        try:
            with fitsio.FITS(fname) as fitsfile:
                fbrmap = fitsfile['FIBERMAP'].read(columns='TARGETID')
                hdr = fitsfile['CONTINUA'].read_header()
                continua = fitsfile['CONTINUA'].read().astype(np.float64)
        except Exception as e:
            warnings.warn(
                f"InputContinuumModel cannot read file {fname}. "
                f"Reason {e}.")
            return 0, 0, None, None

        if fbrmap.size != continua.shape[0]:
            raise Exception(
                "InputContinuumModel file error. The number of quasars do not "
                f"match between FIBERMAP and CONTINUA in {fname}.")

        common_targetids, idx_fbr, _ = np.intersect1d(
            fbrmap, targetids, return_indices=True)
        return hdr['WAVE1'], hdr['DWAVE'], common_targetids, continua[idx_fbr]

    def _read_continua(self, spectra_list):
        """Reads and sets input continuum for all elements in ``spectra_list``.
        If a TARGETID is not found in its expected file, its continuum will be
        set to invalid.

        If found, :attr:`qsonic.spectrum.Spectrum.cont_params` dictionary gains
        the following::

            cont_params['valid'] = True
            cont_params['input_w1'] (float): First rest-frame wavelength
            cont_params['input_dwave'] (float): Rest-frame wavelength spacing
            cont_params['input_data'] (ndarray): Input continuum

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects.
        """
        local_catalog = np.hstack([_.catrow for _ in spectra_list]).copy()
        local_catalog['HPXPIXEL'] = ang2pix(
            self.nside, local_catalog['RA'], local_catalog['DEC'],
            lonlat=True, nest=True)

        # local_catalog is not sorted
        idx_sort = local_catalog.argsort(order=['HPXPIXEL', 'TARGETID'])
        spectra_list = np.array(spectra_list)[idx_sort]
        local_catalog = local_catalog[idx_sort]

        unique_pix, s = np.unique(local_catalog['HPXPIXEL'], return_index=True)
        hpx_split_catalog = np.split(local_catalog, s[1:])
        hpx_split_spectra_list = np.split(spectra_list, s[1:])

        for cat, specs in zip(hpx_split_catalog, hpx_split_spectra_list):
            pixnum = cat['HPXPIXEL'][0]
            targetids = cat['TARGETID']
            wave_rf_1, dwave_rf, common_targetids, continua = \
                self._read_continua_one_file(pixnum, targetids)

            if wave_rf_1 == 0:
                continue

            _find_append_continuum(
                specs, wave_rf_1, dwave_rf, common_targetids, continua)

    def fit_continuum(self, spec):
        """Input continuum reduction. Uses fiducials for mean flux and varlss
        interpolation. Continuum is interpolated using a cubic spline.

        Arguments
        ---------
        spec: Spectrum
            Spectrum object to add true continuum.
        """
        if not spec.cont_params['valid']:
            spec.cont_params['cont'] = None
            spec.cont_params['chi2'] = -1
            return

        input_cont_interp = FastCubic1DInterp(
            spec.cont_params['input_w1'], spec.cont_params['input_dwave'],
            spec.cont_params['input_data'])

        for arm, wave_arm in spec.forestwave.items():
            cont_est = input_cont_interp(wave_arm / (1 + spec.z_qso))

            if any(cont_est <= 0) or any(np.isnan(cont_est)):
                spec.cont_params['valid'] = False
                break

            cont_est *= self.meanflux_interp(wave_arm)
            spec.cont_params['cont'][arm] = cont_est

        spec.set_forest_weight(self.varlss_interp, self.eta_interp)
        spec.calc_continuum_chi2()

        if not spec.cont_params['valid']:
            spec.cont_params['cont'] = None
            spec.cont_params['chi2'] = -1

    def init_spectra(self, spectra_list):
        """ Reads input continuum. Initializes
        :attr:`cont_params <qsonic.spectrum.Spectrum.cont_params>` for a list
        of Spectrum objects.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            Spectrum objects.
        """
        logging.info("Initializing input continuum.")

        for spec in spectra_list:
            spec.cont_params['method'] = 'input'
            spec.cont_params['valid'] = False
            spec.cont_params['x'] = np.zeros(1)
            spec.cont_params['xcov'] = np.eye(1)
            spec.cont_params['dof'] = spec.get_real_size()
            spec.cont_params['cont'] = {}

        self._read_continua(spectra_list)

    def stacks_residual_flux(self):
        """:meth:`stack_spectra` stacks actual continuum values.

        Returns
        -------
        False: bool
        """
        return False
