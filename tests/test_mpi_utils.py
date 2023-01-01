import argparse
import pytest
from unittest import TestCase

import numpy as np
import numpy.testing as npt

import qcfitter.mpi_utils
import qcfitter.spectrum


class TestMPIUtils(TestCase):
    @pytest.mark.mpi(min_size=2)
    def test_mpi_parse(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        assert comm.size > 0

        mpi_rank = comm.Get_rank()
        parser = argparse.ArgumentParser()
        qcfitter.spectrum.add_io_parser(parser)

        options = "--input-dir indir --catalog incat -o outdir".split(' ')
        args = qcfitter.mpi_utils.mpi_parse(parser, comm, mpi_rank, options)
        assert (args.input_dir == "indir")
        assert (args.catalog == "incat")
        assert (args.outdir == "outdir")

        with pytest.raises(SystemExit):
            options = "--catalog incat -o outdir".split(' ')
            qcfitter.mpi_utils.mpi_parse(parser, comm, mpi_rank, options)

    def test_logging_mpi(self):
        with self.assertLogs(level='INFO') as cm:
            qcfitter.mpi_utils.logging_mpi("test1", 0)
            qcfitter.mpi_utils.logging_mpi("test2", 1)
            qcfitter.mpi_utils.logging_mpi("test3", 0, "error")
        self.assertEqual(cm.output, ["INFO:root:test1", "ERROR:root:test3"])

    def test_balance_load(self):
        split_catalog = [
            np.ones(3), 2 * np.ones(4), 3 * np.ones(5), 4 * np.ones(1)]

        sorted_catalog = [
            3 * np.ones(5), 2 * np.ones(4), np.ones(3), 4 * np.ones(1)]

        mpi_size = 3
        q0 = qcfitter.mpi_utils.balance_load(split_catalog, mpi_size, 0)
        q1 = qcfitter.mpi_utils.balance_load(split_catalog, mpi_size, 1)
        q2 = qcfitter.mpi_utils.balance_load(split_catalog, mpi_size, 2)
        for idx in range(len(split_catalog)):
            npt.assert_allclose(split_catalog[idx], sorted_catalog[idx])
        assert (len(q0) == 1)
        assert (len(q1) == 1)
        assert (len(q2) == 2)
        npt.assert_allclose(q0, 3)
        npt.assert_allclose(q1, 2)


if __name__ == '__main__':
    pytest.main()
