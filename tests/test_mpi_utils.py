import argparse
import pytest
from unittest import TestCase

import numpy as np
import numpy.testing as npt

from qsonic import QsonicException
import qsonic.mpi_utils


class TestMPIUtils(TestCase):
    @pytest.mark.mpi(min_size=2)
    def test_mpi_parse(self):
        from mpi4py import MPI
        from qsonic.io import add_io_parser
        comm = MPI.COMM_WORLD
        assert comm.size > 0

        mpi_rank = comm.Get_rank()
        parser = argparse.ArgumentParser()
        add_io_parser(parser)

        options = "--input-dir indir --catalog incat -o outdir".split(' ')
        args = qsonic.mpi_utils.mpi_parse(parser, comm, mpi_rank, options)
        assert (args.input_dir == "indir")
        assert (args.catalog == "incat")
        assert (args.outdir == "outdir")

        with pytest.raises(SystemExit):
            options = "--catalog incat -o outdir".split(' ')
            qsonic.mpi_utils.mpi_parse(parser, comm, mpi_rank, options)

        # Test logic functions
        from qsonic.scripts.qsonic_fit import (
            get_parser, args_logic_fnc_qsonic_fit)

        parser = get_parser()

        options = ("--input-dir indir --catalog incat -o outdir "
                   "--mock-analysis --continuum-model true "
                   "--fiducial-meanflux mflux "
                   "--fiducial-varlss varlss").split(' ')
        args = qsonic.mpi_utils.mpi_parse(parser, comm, mpi_rank, options,
                                          args_logic_fnc_qsonic_fit)
        assert args.continuum_model == "true"
        assert args.true_continuum
        assert args.mock_analysis

        with pytest.raises(SystemExit):
            options = ("--input-dir indir --catalog incat -o outdir "
                       "--continuum-model true").split(' ')
            qsonic.mpi_utils.mpi_parse(parser, comm, mpi_rank, options,
                                       args_logic_fnc_qsonic_fit)

        with pytest.raises(SystemExit):
            qsonic.mpi_utils.mpi_parse(parser, comm, mpi_rank, ['-h'],
                                       args_logic_fnc_qsonic_fit)

    def test_mpi_fnc_bcast(self):
        matrix = np.arange(20).reshape(4, 5)
        u, s, vh = np.linalg.svd(matrix)
        result = qsonic.mpi_utils.mpi_fnc_bcast(
            np.linalg.svd, None, 0, "Error", matrix)

        npt.assert_allclose(result[0], u)
        npt.assert_allclose(result[1], s)
        npt.assert_allclose(result[2], vh)

        s = np.linalg.svd(matrix, compute_uv=False)
        result = qsonic.mpi_utils.mpi_fnc_bcast(
            np.linalg.svd, None, 0, "Error", matrix, compute_uv=False)

        npt.assert_allclose(result, s)

        with pytest.raises(QsonicException):
            qsonic.mpi_utils.mpi_fnc_bcast(np.zeros, None, 0, "Error", -5)

    def test_balance_load(self):
        split_catalog = [
            np.ones(3), 2 * np.ones(4), 3 * np.ones(5), 4 * np.ones(1)]

        sorted_catalog = [
            3 * np.ones(5), 2 * np.ones(4), np.ones(3), 4 * np.ones(1)]

        mpi_size = 3
        q0, q1, q2 = qsonic.mpi_utils.balance_load(split_catalog, mpi_size)

        for idx in range(len(split_catalog)):
            npt.assert_allclose(split_catalog[idx], sorted_catalog[idx])
        assert (len(q0) == 1)
        assert (len(q1) == 1)
        assert (len(q2) == 2)
        npt.assert_allclose(q0, 3)
        npt.assert_allclose(q1, 2)


if __name__ == '__main__':
    pytest.main()
