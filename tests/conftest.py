# from https://docs.pytest.org/en/latest/example/
# simple.html#control-skipping-of-tests-according-to-command-line-option
# accessed on Dec 31, 2022

import pytest
import numpy as np


@pytest.fixture
def setup_data():
    cat_dtype = np.dtype([
        ('TARGETID', '>i8'), ('Z', '>f8'), ('RA', '>f8'), ('DEC', '>f8'),
        ('HPXPIXEL', '>i8'), ('SURVEY', '<U4')])
    cat_by_survey = np.array([
        (39627939372861215, 2.328, 229.861, 6.1925, 8258, b'main')],
        dtype=cat_dtype)

    npix = 1000
    data = {
        'wave': {
            'B': 3600. + 0.8 * np.arange(npix),
            'R': 4000. + 0.8 * np.arange(npix)},
        'flux': {
            'B': 2.1 * np.ones(npix).reshape(1, npix),
            'R': 2.1 * np.ones(npix).reshape(1, npix)},
        'ivar': {
            'B': np.ones(npix).reshape(1, npix),
            'R': np.ones(npix).reshape(1, npix)},
        'mask': {
            'B': np.zeros(npix, dtype='i4').reshape(1, npix),
            'R': np.zeros(npix, dtype='i4').reshape(1, npix)},
        'reso': {}
    }

    return cat_by_survey, npix, data


def pytest_addoption(parser):
    parser.addoption(
        "--with-mpi", action="store_true", default=False,
        help="Run MPI tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: mark test to run with mpi")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--with-mpi"):
        return
    skip_mpi = pytest.mark.skip(reason="need --with-mpi option to run")
    for item in items:
        if "mpi" in item.keywords:
            item.add_marker(skip_mpi)
