import argparse
import pytest

import qcfitter.io


class TestIOParsers(object):
    def setup_method(self):
        self.parser = argparse.ArgumentParser()

    def test_add_io_parser(self):
        qcfitter.io.add_io_parser(self.parser)
        text = "--input-dir indir --catalog incat -o outdir"
        args = self.parser.parse_args(text.split(' '))
        assert (args.input_dir == "indir")
        assert (args.catalog == "incat")
        assert (args.outdir == "outdir")

        with pytest.raises(SystemExit):
            text = "--catalog incat -o outdir"
            self.parser.parse_args(text.split(' '))

        with pytest.raises(SystemExit):
            text = "--input-dir indir --catalog incat --arms T"
            self.parser.parse_args(text.split(' '))

        with pytest.raises(SystemExit):
            text = "--input-dir indir --catalog incat --skip 1.2"
            self.parser.parse_args(text.split(' '))


if __name__ == '__main__':
    pytest.main()
