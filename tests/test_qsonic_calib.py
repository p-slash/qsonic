import pytest
from unittest import TestCase

import qsonic.scripts.qsonic_calib


class TestQsonicCalib(TestCase):
    def test_parser(self):
        parser = qsonic.scripts.qsonic_calib.get_parser()

        options = "--input-dir indir --catalog incat -o outdir".split(' ')
        args = parser.parse_args(options)
        assert (args.input_dir == "indir")
        assert (args.catalog == "incat")
        assert (args.outdir == "outdir")
        assert (args.nwbins is None)

        with pytest.raises(SystemExit):
            options = "--catalog incat -o outdir".split(' ')
            parser.parse_args(options)


if __name__ == '__main__':
    pytest.main()
