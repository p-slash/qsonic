import argparse
import os
import pytest

import fitsio
import numpy as np
import numpy.testing as npt

from qcfitter.mpi_utils import mpi_parse
import qcfitter.spectrum
from qcfitter.picca_continuum import PiccaContinuumFitter


@pytest.fixture
def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir", '-o',
        help="Output directory to save deltas.")
    qcfitter.spectrum.add_wave_region_parser(parser)
    PiccaContinuumFitter.add_parser(parser)

    return parser


@pytest.fixture
def get_fiducials(tmp_path):
    fname = tmp_path / "fiducials.fits"
    nsize = 100
    data = np.empty(
        nsize,
        dtype=[('LAMBDA', 'f8'), ('MEANFLUX', 'f8'), ('VAR', 'f8')]
    )
    data['LAMBDA'] = np.linspace(3600, 6000, nsize)
    data['MEANFLUX'] = 2 * np.ones(nsize)
    data['VAR'] = 3 * np.ones(nsize)

    with fitsio.FITS(fname, 'rw', clobber=True) as fts:
        fts.write(data, extname='STATS')

    return fname


def test_add_parser(setup_parser):
    parser = setup_parser
    parser.parse_args([])

    args = parser.parse_args(["--rfdwave", "1.2"])
    npt.assert_almost_equal(args.rfdwave, 1.2)


@pytest.mark.mpi(min_size=2)
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

        npt.assert_allclose(
            qcfit.rfwave[[0, -1]], [args.forest_w1, args.forest_w2])
        npt.assert_almost_equal(qcfit.meancont_interp.xp0, args.forest_w1)
        npt.assert_allclose(qcfit.meancont_interp.fp, 1)
        assert (qcfit.varlss_fitter is not None)

        # Test fiducial
        if mpi_rank == 0:
            fname = str(get_fiducials)
        else:
            fname = ""

        args = mpi_parse(parser, comm, mpi_rank, ["--fiducials", fname])
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
        spec = qcfitter.spectrum.generate_spectra_list_from_data(
            cat_by_survey, data)[0]
        spec.set_forest_region(3600., 6000., args.forest_w1, args.forest_w2)

        qcfit.fit_continuum(spec)
        assert (spec.cont_params['valid'])
        npt.assert_almost_equal(spec.cont_params['x'], [2.1, 0])


if __name__ == '__main__':
    pytest.main()
