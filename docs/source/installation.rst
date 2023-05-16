Installation
============

Explicit NERSC instructions for the lazy:

.. code-block:: shell

    module load python
    module load PrgEnv-gnu
    conda create -n qsonic python numpy scipy numba iminuit fitsio healpy
    conda activate qsonic
    MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    pip install qsonic

General
-------

It is recommended that you create a separate conda environment and run

.. code-block:: shell
    
    pip install qsonic

You may also want to install required packages in ``requirements.txt`` while creating your conda environment, but pip should take care of it.

Latest from the source
----------------------

If you want to get the latest (possibly unstable) version:

.. code-block:: shell

    pip install "git+https://github.com/p-slash/qsonic.git"

For developers:

.. code-block:: shell

    git clone https://github.com/p-slash/qsonic.git
    cd qsonic
    pip install -e '.[dev]'

NERSC
-----

``mpi4py`` needs special attention. Follow these `instructions <https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment>`_ to clone a mpi4py installed conda environment on NERSC.

