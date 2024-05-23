class BaseContinuumModel():
    """The contract for BaseContinuumModel"""

    def __init__(self):
        raise NotImplementedError

    def stack_spectra(self, valid_spectra_list, flux_stacker):
        """Stacks spectra in the observed and rest-frame. Observed frame
        stacking is performed over f/C. Rest-frame stacking is performed over C
        only.

        Arguments
        ---------
        valid_spectra_list: list(Spectrum)
            Valid spectra objects to iterate.
        flux_stacker: FluxStacker
            Flux stacker object.
        """
        for spec in valid_spectra_list:
            for arm, wave_arm in spec.forestwave.items():
                wave_rf_arm = wave_arm / (1 + spec.z_qso)

                cont = spec.cont_params['cont'][arm]
                weight = spec.forestweight[arm] * cont**2
                weighted_flux = weight * spec.forestflux[arm] / cont
                weighted_cont = weight * cont

                flux_stacker.add(
                    wave_arm, wave_rf_arm, weighted_flux, weighted_cont, weight
                )

    def fit_continuum(self, spec):
        """Fits the continuum for a single Spectrum. Should modify
        :attr:`cont_params <qsonic.spectrum.Spectrum.cont_params>`
        dictionary's ``valid`` and ``cont`` keys at least. ``x, xcov, chi2``
        and ``dof`` keys should be set either here or in :meth:`init_spectra`.

        Arguments
        ---------
        spec: Spectrum
            Spectrum object to fit.
        """
        raise NotImplementedError

    def init_spectra(self, spectra_list):
        """ Initializes
        :attr:`cont_params <qsonic.spectrum.Spectrum.cont_params>` for a list
        of Spectrum objects. Specifically, set ``method`` key and any unused
        keys in ``x, xcov, chi2, dof``.

        Arguments
        ---------
        spectra_list: list(Spectrum)
            List of Spectrum objects.
        """
        raise NotImplementedError

    def stacks_residual_flux(self):
        """Return ``True`` if the model stacks residual fluxes in the
        rest-frame in :meth:`stack_spectra` such as
        :class:`PiccaContinuumModel`. Otherwise, return ``False``.

        Returns
        -------
        bool
        """
        raise NotImplementedError
