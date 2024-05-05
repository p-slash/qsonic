import pytest
from unittest import TestCase

import qsonic.scripts.qsonic_fit


class TestQsonicFit(TestCase):
    def test_parser(self):
        parser = qsonic.scripts.qsonic_fit.get_parser()

        options = (
            "--input-dir indir --catalog incat -o outdir "
            "--continuum-model true --mock-analysis --fiducial-meanflux mf "
            "--fiducial-varlss vs"
        ).split(' ')
        args = parser.parse_args(options)
        assert (args.input_dir == "indir")
        assert (args.catalog == "incat")
        assert (args.outdir == "outdir")

        x = qsonic.scripts.qsonic_fit.args_logic_fnc_qsonic_fit(args)
        assert x

        options = (
            "--input-dir indir --catalog incat -o outdir "
            "--true-continuum --fiducial-meanflux mf --fiducial-varlss vs"
        ).split(' ')
        args = parser.parse_args(options)
        x = qsonic.scripts.qsonic_fit.args_logic_fnc_qsonic_fit(args)
        assert not x

        options = (
            "--input-dir indir --catalog incat -o outdir "
            "--true-continuum --mock-analysis --fiducial-meanflux mf "
            "--fiducial-varlss vs --noise-calibration nc "
            "--varlss-as-additive-noise --flux-calibration fc"
        ).split(' ')
        args = parser.parse_args(options)
        x = qsonic.scripts.qsonic_fit.args_logic_fnc_qsonic_fit(args)
        assert x


if __name__ == '__main__':
    pytest.main()
