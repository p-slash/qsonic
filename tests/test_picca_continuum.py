import argparse
import os
import pytest

import fitsio
import numpy as np
import numpy.testing as npt

from qsonic.mpi_utils import mpi_parse
import qsonic.spectrum
from qsonic.picca_continuum import (
    PiccaContinuumFitter, VarLSSFitter, add_picca_continuum_parser)


@pytest.fixture
def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir", '-o',
        help="Output directory to save deltas.")
    qsonic.spectrum.add_wave_region_parser(parser)
    parser = add_picca_continuum_parser(parser)

    return parser


@pytest.fixture
def get_fiducials(tmp_path):
    fname = tmp_path / "fiducials.fits"
    nsize = 100
    data = np.empty(
        nsize,
        dtype=[('LAMBDA', 'f8'), ('MEANFLUX', 'f8'), ('VAR_LSS', 'f8')]
    )
    data['LAMBDA'] = np.linspace(3600, 6000, nsize)
    data['MEANFLUX'] = 2 * np.ones(nsize)
    data['VAR_LSS'] = 3 * np.ones(nsize)

    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        fts.write(data, extname='STATS')

    return fname


def test_add_parser(setup_parser):
    parser = setup_parser
    parser.parse_args([])

    args = parser.parse_args(["--rfdwave", "1.2"])
    npt.assert_almost_equal(args.rfdwave, 1.2)


@pytest.mark.mpi
class TestPiccaContinuum(object):
    def test_init(self, setup_parser, get_fiducials):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        assert mpi_size > 0

        parser = setup_parser
        args = mpi_parse(parser, comm, mpi_rank, [])
        qcfit = PiccaContinuumFitter(args)
        # assert rfwave is centers and forest_w1 is the edge
        w1rf_truth = args.forest_w1 + qcfit.dwrf / 2
        w2rf_truth = args.forest_w2 - qcfit.dwrf / 2

        npt.assert_allclose(
            qcfit.rfwave[[0, -1]], [w1rf_truth, w2rf_truth])
        npt.assert_almost_equal(qcfit.meancont_interp.xp0, w1rf_truth)
        npt.assert_allclose(qcfit.meancont_interp.fp, 1)
        assert (qcfit.varlss_fitter is not None)

        # Test fiducial
        if mpi_rank == 0:
            fname = str(get_fiducials)
        else:
            fname = ""

        args = mpi_parse(
            parser, comm, mpi_rank,
            ["--fiducial-meanflux", fname, "--fiducial-varlss", fname])
        qcfit = PiccaContinuumFitter(args)
        if mpi_rank == 0:
            os.remove(fname)

        npt.assert_allclose(qcfit.meanflux_interp.fp, 2)
        npt.assert_allclose(qcfit.varlss_interp.fp, 3)
        assert (qcfit.varlss_fitter is None)

    def test_get_continuum_model(self, setup_parser):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        assert mpi_size > 0

        parser = setup_parser
        args = mpi_parse(parser, comm, mpi_rank, [])
        qcfit = PiccaContinuumFitter(args)

        # np.log(wave_rf_arm / self.rfwave[0]) / self._denom
        slope = np.linspace(0, 1, 200)
        wave_rf_arm = qcfit.rfwave[0] * np.exp(qcfit._denom * slope)
        expected_cont = np.linspace(0, 2, 200)

        x = np.array([1., 1.])
        cont_est = qcfit.get_continuum_model(x, wave_rf_arm)
        npt.assert_allclose(cont_est, expected_cont)

        x = np.array([1., 1., 0])
        cont_est = qcfit.get_continuum_model(x, wave_rf_arm)
        npt.assert_allclose(cont_est, expected_cont)

    def test_fit_continuum(self, setup_parser, setup_data):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        assert mpi_size > 0

        parser = setup_parser
        args = mpi_parse(parser, comm, mpi_rank, [])
        qcfit = PiccaContinuumFitter(args)
        qcfit.varlss_interp.fp *= 0
        npt.assert_allclose(qcfit.meanflux_interp.fp, 1)

        cat_by_survey, npix, data = setup_data(1)
        data['ivar']['B'] *= 10
        data['ivar']['R'] *= 10
        spec = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)[0]
        spec.set_forest_region(3600., 6000., args.forest_w1, args.forest_w2)

        qcfit.fit_continuum(spec)
        assert (spec.cont_params['valid'])
        npt.assert_almost_equal(spec.cont_params['x'], [2.1, 0])

    def test_project_normalize_meancont(self, setup_parser):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        assert mpi_size > 0

        parser = setup_parser
        args = mpi_parse(parser, comm, mpi_rank, ["--cont-order", "3"])
        qcfit = PiccaContinuumFitter(args)
        x = np.log(qcfit.rfwave / qcfit.rfwave[0]) / qcfit._denom
        x = 2 * x - 1

        # add Legendre-like distortions
        new_meancont = 1.5 * np.ones_like(x)
        for ci in range(1, args.cont_order + 1):
            new_meancont += 0.5 / ci * x**ci

        new_meancont, _ = qcfit._project_normalize_meancont(new_meancont)
        npt.assert_allclose(new_meancont, 1, rtol=1e-3)
        npt.assert_almost_equal(new_meancont.mean(), 1, decimal=4)

    def test_update_mean_cont(self, setup_parser, setup_data):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        assert mpi_size > 0

        parser = setup_parser
        # There are empty bins in the rest-frame, they should be extrapolated
        # correctly.
        args = mpi_parse(parser, comm, mpi_rank, [])
        qcfit = PiccaContinuumFitter(args)

        cat_by_survey, npix, data = setup_data(3)
        spectra_list = qsonic.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)

        for spec in spectra_list:
            spec.set_forest_region(
                3600., 6000., args.forest_w1, args.forest_w2)
            spec.cont_params['valid'] = True
            spec.cont_params['cont'] = {}
            for arm, wave_arm in spec.forestwave.items():
                spec.cont_params['cont'][arm] = np.ones_like(wave_arm)

        qcfit.update_mean_cont(spectra_list)
        npt.assert_allclose(qcfit.meancont_interp.fp, 1, rtol=1e-3)
        npt.assert_almost_equal(qcfit.meancont_interp.fp.mean(), 1)


class TestVarLSSFitter(object):
    def test_add(self, setup_data):
        nwbins = 4
        nsnrbins = 2
        nvarbins = 3
        binning_vector_size = (nwbins + 2) * (nsnrbins + 2) * (nvarbins + 2)

        def _get_idx(v, s, w):
            return v + (nvarbins + 2) * (s + (nsnrbins + 2) * w)

        dwbins = 300.
        wbins_truth = 3600. + (0.5 + np.arange(nwbins)) * dwbins

        true_ivar_bin = 1
        true_snr_bin = 1
        full_bins = np.zeros(binning_vector_size, dtype=bool)

        varlss_fitter = VarLSSFitter(
            3600, 4800, nwbins=nwbins, var1=1e-5, var2=2., nvarbins=nvarbins,
            icont2bins=nsnrbins)
        npt.assert_almost_equal(varlss_fitter.dwobs, dwbins)
        npt.assert_allclose(varlss_fitter.waveobs, wbins_truth)

        cat_by_survey, npix, data = setup_data(1)
        cont = np.ones_like(data['wave']['B'])
        varlss_fitter.add(
            data['wave']['B'], data['flux']['B'][0], data['ivar']['B'][0],
            cont)

        # Blue arm only fills up to nwbins - 1 wavelength bin
        true_wave_bins = np.arange(1, nwbins)
        true_all_idx = _get_idx(true_ivar_bin, true_snr_bin, true_wave_bins)
        full_bins[true_all_idx] = 1

        npt.assert_equal(varlss_fitter._num_pixels[~full_bins], 0)

        expected_numqso = np.zeros(binning_vector_size, dtype=int)
        expected_numqso[full_bins] = 1
        npt.assert_equal(varlss_fitter._num_qso, expected_numqso)

        cont = np.ones_like(data['wave']['R'])
        varlss_fitter.add(
            data['wave']['R'], data['flux']['R'][0], data['ivar']['R'][0],
            cont)

        # Red arm fills the last 3 wavelength bins. First two overlap with
        # previous wavelength bins.
        true_wave_bins = np.arange(2, nwbins)
        true_all_idx = _get_idx(true_ivar_bin, true_snr_bin, true_wave_bins)
        expected_numqso[true_all_idx] = 2
        expected_numqso[_get_idx(true_ivar_bin, true_snr_bin, nwbins)] = 1
        npt.assert_equal(varlss_fitter._num_qso, expected_numqso)

        expected_size = nwbins * nsnrbins * nvarbins
        varlss_fitter._calc_subsampler_stats()

        npt.assert_equal(varlss_fitter.wvalid_bins.sum(), expected_size)
        npt.assert_equal(varlss_fitter.mean_delta.size, expected_size)

    def test_fit(self):
        varlss_fitter = VarLSSFitter(
            3600, 4800, nwbins=1, var1=1e-5, var2=2.)

        true_var_lss = np.array([0.1])
        nspec = 1000
        nwave = int((4800 - 3600) / 0.8) + 1
        wave = np.linspace(3600, 4800, nwave)
        RNST = np.random.default_rng()
        for jj in range(nspec):
            var_pipe = RNST.uniform(np.log10(1e-5), np.log10(2), size=nwave)
            var_pipe = 10**var_pipe
            std_gen = np.sqrt(var_pipe + true_var_lss)
            delta = RNST.normal(0, std_gen, nwave)
            cont = np.ones_like(delta)

            varlss_fitter.add(wave, delta, 1 / var_pipe, cont)

        fit_results, std_results = varlss_fitter.fit(
            true_var_lss, smooth=False)
        atol = 3 * std_results[0]
        rtol = atol / true_var_lss[0]
        npt.assert_allclose(fit_results, true_var_lss, rtol=rtol, atol=atol)


if __name__ == '__main__':
    pytest.main()
