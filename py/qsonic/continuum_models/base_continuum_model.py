class BaseContinuumModel():
    """The contract for BaseContinuumModel"""

    def __init__(self):
        raise NotImplementedError

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
        """
        raise NotImplementedError
