# from https://docs.pytest.org/en/latest/example/
# simple.html#control-skipping-of-tests-according-to-command-line-option
# accessed on Dec 31, 2022

import pytest


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
