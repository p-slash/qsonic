========
qcfitter
========

.. image:: https://github.com/p-slash/qcfitter/actions/workflows/main.yml/badge.svg
    :target: https://github.com/p-slash/qcfitter/actions/workflows/main.yml
    :alt: Tests Status

.. image:: https://readthedocs.org/projects/qcfitter/badge/?version=latest
    :target: https://qcfitter.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**qcfitter** is an MPI parallelized, highly optimized quasar continuum fitter package built on the same algorithm as `picca <https://github.com/igmhub/picca>`_.

The key differences
-------------------
- Coadding of spectrograph arms is optional and performed after continuum fitting.
- Continuum is multiplied by a fiducial mean flux when provided.
- You can pass fiducial var_lss (column ``VAR``) and mean flux (column ``MEANFLUX``) for observed wavelength ``LAMBDA`` in ``STATS`` extention of a FITS file. Wavelength should be linearly and equally spaced. This is the same format as rawio output from picca.
- If no fiducial is passed, we fit only for var_lss (no eta fitting currently).
- Internal weights for continuum fitting and coadding are based on smoothed ``IVAR``, but output ``WEIGHT`` is **not** smoothed.
- Chi2 information as well as best fits are saved in continuum_chi2_catalog.fits. Chi2 is calculated using smooth ivar and var_lss, and does not subtract sum of ln(weights).

Similarities
------------
+ Delta files are the same. `CONT` column is mean flux times continuum even when fiducial mean flux is passed.
+ ``MEANSNR`` in header file and chi2 catalog is average of flux times square root of positive ivar values. Header values are per arm, but catalog values are the average over all arms.

Programs
--------
Scripts under ``bin`` folder are executable. Pass ``--help`` for arguments. The most important ones are these four scripts:

+ ``qcfit`` performs picca-like continuum fitting. Runs on MPI, so on some machines you need to execute ``srun -n 1 qcfit.py --help`` for help.

Source
------
+ `catalog.py` provides function to read catalog and broadcast from master pe. Needs following columns: ``TARGETID, Z, TARGET_RA, RA, TARGET_DEC, DEC, SURVEY, HPXPIXEL, VMIN_CIV_450, VMAX_CIV_450, VMIN_CIV_2000, VMAX_CIV_2000``.
+ `masks.py` provides maskers for sky, BAL and DLA. Sky mask is a simple text file, where colums are  *type* (unused), *wave_min* mininum wavelength, *wave_max* maximum wavelength, *frame* rest-frame (RF) or observed (OBS). BAL mask assumes related ``VMIN_CIV_450, VMAX_CIV_450, VMIN_CIV_2000, VMAX_CIV_2000`` values are already present in `catalog`. DLA masking takes both Lya and Lyb profiles into account, masks ``F<0.8`` and corrects for the wings.
+ `mathtools.py` hosts some functions and classes that are purely numerical.
+ `mpi_utils.py` hosts MPI related functions such as logging.
+ `picca_continuum.py` fits each quasar continuum with a global mean as done in picca.
+ `spectrum.py` has ``Spectrum`` class that stores each quasar spectrum and provides io functions such as ``read_spectra_onehealpix`` and ``save_deltas``.

Installation
------------
Create a conda environment and run ``pip install -e .``. You may want to install required packages in ``requirements.txt`` while creating your conda environment, but pip should take care of it.

NERSC
-----
``mpi4py`` needs special attention. Follow these `instructions <https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment>`_ to clone a mpi4py installed conda environment on NERSC.


