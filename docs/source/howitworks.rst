How it works
============

QSOnic follows the same algorithm as `picca <https://github.com/igmhub/picca>`_, which was developed over the last few years and has been applied to both 3D analyses [Bautista2017]_, [duMasdesBourboux2019]_, [duMasdesBourboux2020]_ and P1D measurements [Chabanier2019]_.

One important aspect of the algorithm is that the definition of the quasar continuum absorbs the mean flux :math:`\overline{F}(z)` of the IGM. Specifically, we model every quasar "continuum" :math:`\overline{F} C_q(\lambda_{\mathrm{RF}})` by a global mean continuum :math:`\overline{C}(\lambda_{\mathrm{RF}})` and two quasar "diversity" parameters, amplitude :math:`a_q` and slope :math:`b_q`.

.. math::

    \overline{F} C_q(\lambda_{\mathrm{RF}}) &= \overline{C}(\lambda_{\mathrm{RF}}) (a_q + b_q \Lambda)

    \Lambda &= \frac{\log\lambda_\mathrm{RF} - \log\lambda_\mathrm{RF}^{(1)}}{\log\lambda_\mathrm{RF}^{(2)} - \log\lambda_\mathrm{RF}^{(1)}},

where :math:`\lambda_\mathrm{RF}` is the wavelength in quasar's rest frame. We assume that the global mean continuum :math:`\overline{C}(\lambda_\mathrm{RF})` does not depend on redshift, and therefore our model only adjusts :math:`\overline{F}(z)`, as well as solves for the  :math:`a_q` and :math:`b_q` parameters for each quasar. In other words, the amplitude and slope parameters do not only fit for intrinsic quasar diversity such as brightness, but also for the IGM mean flux. This can be alleviated by passing a FITS file to ``--fiducial-meanflux`` option in QSOnic.
Given these definitions, transmitted flux fluctuations are given by

.. math::

    \delta_q(\lambda) = \frac{f_q(\lambda)}{\overline{F}C_q(\lambda)} - 1,

where :math:`\lambda=(1+z_\mathrm{qso})\lambda_\mathrm{RF}` is the observed wavelength and :math:`f_q(\lambda)` is the observed flux. The effect of spectrograph resolution has been ignored for simplicity, since the affected scales are small for 3D analysis.
The features in the continuum are also wider than the spectrograph resolution, so this assumption should also hold for P1D.

In QSOnic, each quasar spectrum is a :class:`Spectrum <qsonic.spectrum.Spectrum>` class. This class stores wavelength, flux, inverse variance and resolution matrix as dictionaries, where keys correspond to spectrograph arms. The underlying wavelength grid is stored statically and all instances share the same array. The forest region is accessed by ``forest*`` attributes, which are only set after :meth:`qsonic.spectrum.Spectrum.set_forest_region` is called.

QSOnic continuum fitting is organized under five steps:

#. Reading
#. Priming
#. Fitting
#. Cleaning
#. Saving

The umbrella function is :func:`qsonic.scripts.qsonic_fit.mpi_run_all`.

Reading
-------
This stage includes reading the quasar catalog, files needed for masking and the quasar spectra. Here is what happens in chronological order:

We read the quasar catalog, apply a quasar redshift cut. The minimum and maximum quasar redshifts are calculated as follows:

.. literalinclude:: ../../py/qsonic/scripts/qsonic_fit.py
   :language: python3
   :linenos:
   :lines: 240-242

The catalog is then grouped into healpixels. We balance the load of each MPI process by roughly having equal number of spectra per process.

We determine if the data needs to be blinded in :meth:`qsonic.spectrum.Spectrum.set_blinding`.

We read masks in :func:`qsonic.scripts.qsonic_fit.mpi_read_masks`. See :mod:`qsonic.masks` for :class:`SkyMask <qsonic.masks.SkyMask>`, :class:`BALMask <qsonic.masks.BALMask>` and :class:`DLAMask <qsonic.masks.DLAMask>` file conventions.

We read spectra in :func:`qsonic.scripts.qsonic_fit.mpi_read_spectra_local_queue`. This is done by reading all **quasar** spectra in a healpix file using the function :func:`qsonic.io.read_spectra_onehealpix`, which uses slices to read **only** the quasar spectra in a given file. Then, the analysis region is enforced by the given wavelength arguments, which also populate ``forest*`` attributes of :class:`qsonic.spectrum.Spectrum`. Even though underlying API can keep the entire spectrum, we drop all regions that are outside the analysis region to save memory. Furthermore, a right-side SNR (RSNR) cut is applied to remove quasars if such an option is passed while reading, but techinally this is part of priming.

Priming
-------
This stage includes eliminating quasars based on remaining pixels and SNR, reading files that are needed for calibration and then calibrating, masking pixels for sky, BAL and DLA features, and finally smoothing the pipeline variance (which only affects the weights and not the output IVAR column in delta files).

As noted above, a right-side SNR (RSNR) cut is applied to remove quasars (if such an option is passed) while reading the spectra. The spectra are further recalibrated, masked, corrected and eliminated if short.

We read and apply noise and flux calibrations in :func:`qsonic.scripts.qsonic_fit.mpi_noise_flux_calibrate`. See :mod:`qsonic.calibration` for :class:`NoiseCalibrator <qsonic.calibration.NoiseCalibrator>` and :class:`FluxCalibrator <qsonic.calibration.FluxCalibrator>`.

We apply masks in :func:`qsonic.scripts.qsonic_fit.apply_masks`. See :mod:`qsonic.masks` for :class:`SkyMask <qsonic.masks.SkyMask>`, :class:`BALMask <qsonic.masks.BALMask>` and :class:`DLAMask <qsonic.masks.DLAMask>`. Masking is set by setting ``forestivar=0``. :class:`DLAMask <qsonic.masks.DLAMask>` further corrects for Lya and Lyb damping wings. Empty arms are removed after masking. Short spectra are also removed from the list.

Inverse variance is smoothed by ``--smoothing-scale``, which will be the ingredient of ``forestweights``. 'IVAR' column of the delta files are not smoothed, but 'WEIGHT' column will be.

Continuum fitting
-----------------
This stage starts with the construction of a :class:`PiccaContinuumFitter <qsonic.picca_continuum.PiccaContinuumFitter>` class. At initialization:

- Number of bins for the mean continuum is calculated. This number aims to achieve rest-frame wavelength step size that is close to ``--rfdwave``, but not exactly. Rest-frame forest wavelengths are strictly enforced, where ``--forest_w1`` and ``--forest_w2`` are the edges (not the centers).
- A fast cubic spline with initial value of one is contructed for the mean continuum interpolation.
- Minimizer is chosen for the given option (iminuit by default).
- Fiducial mean flux and varlss values are read from file if passed. This file must be a FITS file with a 'STATS' extension. This extension must have an **equally and linearly spaced** 'LAMBDA' column for the wavelength in Angstrom. Fiducial flux is read from the 'MEANFLUX' column, whereas varlss is read from the 'VAR' column. Note you can use the same file for both options. These fiducial values are **linearly** interpolated.
- If no fiducial varlss is provided, a :class:`VarLSSFitter <qsonic.picca_continuum.VarLSSFitter>` class and a fast cubic spline with initial value of 0.1 are constructed. The fitter class uses bin size of approximately 120 A in the observed frame, where ``--wave1`` and ``--wave2`` are the edges. It applies the delete-one block Jackknife method to calculate the errors (and the covariance if requested) over 10,000 subsamples.
- A fast cubic spline with initial value of zero is constructed for the noise calibration parameter :math:`\eta`. Fitting for this parameter is enabled only if ``--var-fit-eta`` option is passed.


We then start iterating, which itself consists of three major steps: initialization, fitting, updating the global variables. The initialization sets ``cont_params`` variable of every Spectrum object. Continuum polynomial order is carried by setting ``cont_params[x]``. At each iteration:

#. Global variables (mean continuum, var_lss) are saved to ``attributes.fits`` file. This ensures the order of what is used in each iteration.
#. All spectra are fit.
#. Mean continuum is updated by stacking, smoothing and removing degenarate modes. We check for convergence (update is small).
#. If we are fitting for var_lss, we fit and update by calculating variance statistics.

.. [Bautista2017] Bautista J. E., et al., 2017, `A&A, 603, A12 <https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..12B/abstract>`_
.. [duMasdesBourboux2019] du Mas des Bourboux H., et al., 2019, `ApJ, 878, 47 <https://ui.adsabs.harvard.edu/abs/2019ApJ...878...47D>`_
.. [duMasdesBourboux2020] du Mas des Bourboux H., et al., 2020, `ApJ, 901, 153 <https://ui.adsabs.harvard.edu/abs/2020ApJ...901..153D/abstract>`_
.. [Chabanier2019] Chabanier S., et al., 2019, `J. Cosmology Astropart. Phys., 2019, 017 <https://iopscience.iop.org/article/10.1088/1475-7516/2019/07/017>`_
