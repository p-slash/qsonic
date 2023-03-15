.. include:: ../../README.rst

Programs
--------
+ ``qsonic-fit`` performs picca-like continuum fitting. See :ref:`here <qsonic_fit_arguments_label>` or execute ``srun -n 1 qsonic-fit.py --help`` for help.

Source
------
See :doc:`/generated/modules` for details.

+ `catalog.py` provides function to read catalog and broadcast from master pe. Needs following columns: ``TARGETID, Z, TARGET_RA, RA, TARGET_DEC, DEC, SURVEY, HPXPIXEL, VMIN_CIV_450, VMAX_CIV_450, VMIN_CIV_2000, VMAX_CIV_2000``.
+ `masks.py` provides maskers for sky, BAL and DLA. Sky mask is a simple text file, where colums are  *type* (unused), *wave_min* mininum wavelength, *wave_max* maximum wavelength, *frame* rest-frame (RF) or observed (OBS). BAL mask assumes related ``VMIN_CIV_450, VMAX_CIV_450, VMIN_CIV_2000, VMAX_CIV_2000`` values are already present in `catalog`. DLA masking takes both Lya and Lyb profiles into account, masks ``F<0.8`` and corrects for the wings.
+ `mathtools.py` hosts some functions and classes that are purely numerical.
+ `mpi_utils.py` hosts MPI related functions such as logging.
+ `picca_continuum.py` fits each quasar continuum with a global mean as done in picca.
+ `spectrum.py` has ``Spectrum`` class that stores each quasar spectrum and provides io functions such as ``read_spectra_onehealpix`` and ``save_deltas``.
