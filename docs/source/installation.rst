Installation
============

If you are on NERSC, I recommend using the pre-installed environment. This is installed on /global/common/software and has better performance at start up.

.. code:: shell
    
    source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh

To have a Jupyter kernel, run the following command after activation:

.. code:: shell

    python -m ipykernel install --user --name qsonic --display-name qsonic


Install from PyPi
-----------------

Explicit NERSC instructions for the lazy:

.. code-block:: shell

    module load cpu
    module load python

    conda create -n qsonic python=3.10
    conda activate qsonic
    MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    pip install qsonic

If you want Jupyter kernel, you need to additionally install ``ipykernel`` into your environment and run the same command. See this `NERSC guide <https://docs.nersc.gov/services/jupyter/how-to-guides/>`_ on Jupyter.


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

    pip install git+ssh://git@github.com/p-slash/qsonic.git

For developers:

.. code-block:: shell

    git clone https://github.com/p-slash/qsonic.git
    cd qsonic
    pip install -e '.[dev]'

NERSC
-----

``mpi4py`` needs special attention. Follow these `instructions <https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment>`_ to clone a mpi4py installed conda environment on NERSC.

