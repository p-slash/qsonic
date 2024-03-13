Installation
============

General
-------

The following commands will create a conda environment and install qsonic.

.. code-block:: shell
    
    conda create -n qsonic python=VERSION
    conda activate qsonic
    pip install qsonic

- Change VERSION to the desired python version. The only restriction is ``VERSION>=3.7`` and there is no upper version restriction. However, I recommend using a python version that is one or two minor points behind the latest version to be safe.
- Due to occasional compatibility problems between numpy, healpy and astropy, major dependencies are version limited to the following (see ``requirements.txt``):

.. parsed-literal::

    numpy~=1.24.0
    numba~=0.57.0
    scipy~=1.10.0
    astropy~=5.2.0
    healpy~=1.16.0
    fitsio~=1.1.0
    iminuit~=2.18.0
    mpi4py 

Note: MPI is necessary to run qsonic scripts.

NERSC
-----
If you are a NERSC user, I recommend using the pre-installed environment. This is installed on /global/common/software and has better performance at start up.

.. code:: shell
    
    source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh

To have a Jupyter kernel, run the following command after activation:

.. code:: shell

    python -m ipykernel install --user --name qsonic --display-name qsonic


*Explicit NERSC installation* instructions with pip:

.. code-block:: shell

    module load cpu
    module load python

    conda create -n qsonic python
    conda activate qsonic
    MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    pip install qsonic

Note: These instructions recompile ``mpi4py`` with NERSC specific `MPI libraries <https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment>`_. ``cc`` command here is a NERSC compiler wrapper that provides optimal compiler options and paths to MPI libraries. Other platforms may use ``mpicc`` instead.

If you want Jupyter kernel, you need to additionally install ``ipykernel`` into your environment and run the same command. See this `NERSC guide <https://docs.nersc.gov/services/jupyter/how-to-guides/>`_ on Jupyter.


Latest from the source
----------------------

If you want to get the latest (possibly unstable) version:

.. code-block:: shell

    pip install git+ssh://git@github.com/p-slash/qsonic.git

For developers:

.. code-block:: shell

    git clone https://github.com/p-slash/qsonic.git
    cd qsonic
    pip install -e '.[dev]'

