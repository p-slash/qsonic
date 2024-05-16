======
QSOnic
======

*Lightining-fast continuum fitting*

.. image:: https://joss.theoj.org/papers/10.21105/joss.06373/status.svg
   :target: https://doi.org/10.21105/joss.06373

.. image:: https://img.shields.io/pypi/v/qsonic?color=blue
    :target: https://pypi.org/project/qsonic

.. image:: https://img.shields.io/badge/Source-qsonic-red
    :target: https://github.com/p-slash/qsonic

.. image:: https://github.com/p-slash/qsonic/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/p-slash/qsonic/actions/workflows/testing.yml
    :alt: Tests Status

.. image:: https://readthedocs.org/projects/qsonic/badge/?version=latest
    :target: https://qsonic.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**QSOnic** is an MPI-parallelized, highly optimized quasar continuum fitting package for DESI built on the same algorithm as `picca <https://github.com/igmhub/picca>`_, but *faster*. It also provides an efficient API to read DESI quasar spectra. If you use this software, please cite the article in the Journal of Open Source Software (JOSS).

The key differences
-------------------
- You can use any desired input continuum from another continuum prediction method as input continuum model.
- Coadding of spectrograph arms can be performed after continuum fitting or disabled entirely.
- Continuum is multiplied by a fiducial mean flux when provided.
- You can pass fiducial var_lss (column **VAR_LSS**) and mean flux (column **MEANFLUX**) for observed wavelength **LAMBDA** in **STATS** extention of a FITS file. Wavelength should be linearly and equally spaced. This is the same format as rawio output from picca, except **VAR** column in picca is the variance on flux not deltas. We break away from that convention by explicitly requiring variance on deltas in a new column.
- If no fiducial is passed, we fit only for var_lss (no eta fitting by default). Eta fitting can be enabled by passing ``--var-fit-eta``. A covariance matrix is calculated based on delete-one Jackknife over subsamples between var_pipe and var_obs data points, and used in var_lss, eta fitting if ``--var-use-cov`` passed (soft recommendation to always enable this).
- Internal weights for continuum fitting and coadding are based on smoothed ``IVAR``, and output ``WEIGHT`` is based on this smoothed ivar. This smoothing can be turned off.
- Chi2 information as well as best fits are saved in continuum_chi2_catalog.fits. Chi2 is calculated using smooth ivar and var_lss, and does not subtract sum of ln(weights).

Similarities
------------
+ Delta files are the same. ``CONT`` column is mean flux times continuum even when fiducial mean flux is passed.
+ ``MEANSNR`` in header file and chi2 catalog is average of flux times square root of positive ivar values. Header values are per arm, but catalog values are the average over all arms.
+ Eta fitting does not rescale ``IVAR`` output. Pipeline noise will be modified with explicit calibration option.
