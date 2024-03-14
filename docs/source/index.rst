.. qsonic documentation master file, created by
   sphinx-quickstart on Tue Mar  7 17:05:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qsonic's documentation!
====================================

.. include:: ../../README.rst

Programs
--------
+ ``qsonic-fit`` performs picca-like continuum fitting. See :ref:`here <qsonic fit arguments>` or execute ``qsonic-fit --help`` for help.
+ ``qsonic-calib`` calculates stacked flux and performs variance fitting with noise calibration (eta) enabled.  See :ref:`here <qsonic calib arguments>` ``qsonic-calib --help`` for help.

Note: Some platforms require these help commands to begin with ``mpirun -np 1`` or ``srun -n 1``.

Source
------
See :doc:`/generated/modules` for details.

+ `calibration.py` containes noise and flux calibration classes. Can be imported without MPI. The output file of the continuum fitting ``attibutes.fits`` can be used as an input for these calibrations. As an additional feature, large-scale variance term ``var_lss`` can be applied as an additive correction in noise calibration by passing ``--varlss-as-additive-noise``.
+ `catalog.py` provides function to read catalog and broadcast from master pe. Needs following columns: ``TARGETID, Z, TARGET_RA, RA, TARGET_DEC, DEC, SURVEY, HPXPIXEL, VMIN_CIV_450, VMAX_CIV_450, VMIN_CIV_2000, VMAX_CIV_2000``.
+ `io.py` module contains functions to read spectra and save deltas. Can be imported without MPI.
+ `masks.py` provides maskers for sky, BAL and DLA. Can be imported without MPI. Sky mask is a simple text file, where colums are  *type* (unused), *wave_min* mininum wavelength, *wave_max* maximum wavelength, *frame* rest-frame (RF) or observed (OBS). BAL mask assumes related ``VMIN_CIV_450, VMAX_CIV_450, VMIN_CIV_2000, VMAX_CIV_2000`` values are already present in `catalog`. DLA masking takes both Lya and Lyb profiles into account, masks ``F<0.8`` and corrects for the wings.
+ `mathtools.py` hosts some functions and classes that are purely numerical. Can be imported without MPI.
+ `mpi_utils.py` hosts MPI related functions such as logging. Can be imported without MPI.
+ `picca_continuum.py` fits each quasar continuum with a global mean as done in picca. Needs MPI.
+ `spectrum.py` has ``Spectrum`` class that stores each quasar spectrum. Can be imported without MPI.

.. toctree::
   :maxdepth: 2
   :glob:

   installation
   quick_start
   examples/lookChi2Catalog
   examples/simple_coadd_showcase
   howitworks
   generated/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
