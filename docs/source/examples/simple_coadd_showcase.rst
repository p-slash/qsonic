Simple coadd showcase
=====================

This tutorial builds on the DESI early data release example detailed in :ref:`Quick Start <edr example and workaround>`. I assume you have access to the EDR data, and created ``QSO_cat_fuji_healpix_only_qso_targets_sv3_fix.fits`` file that fixes the compatibility issue. An empty notebook can be found in the GitHub repo under ``docs/nb/simple_coadd_showcase.ipynb``.

For this example, we are going to read all arms (B, R, Z), but will not read the resolution matrix.

.. code:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    
    import qsonic.catalog
    import qsonic.io

    fname_catalog = "QSO_cat_fuji_healpix_only_qso_targets_sv3_fix.fits"
    indir = "${EDR_DIRECTORY}/spectro/redux/fuji/healpix"
    arms = ['B', 'R', 'Z']
    is_mock = False
    skip_resomat = True

    # Setup reader function
    readerFunction = qsonic.io.get_spectra_reader_function(
        indir, arms, is_mock, skip_resomat,
        read_true_continuum=False, is_tile=False)

First, we read the catalog. Since ``qsonic`` sorts this the catalog by
HPXPIXEL, we can find the unique healpix values and split the catalog
into healpix groups. For example purposes, we are picking a single healpix and
reading all the quasar spectra in that file.

.. code:: python3

    catalog = qsonic.catalog.read_quasar_catalog(fname, is_mock=is_mock)

    # Group into unique pixels
    unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(catalog, s[1:])

    # Pick one healpix to illustrate
    hpx_cat = split_catalog[1]
    healpix = hpx_cat['HPXPIXEL'][0]

    spectra_by_hpx = readerFunction(hpx_cat)

    print(f"There are {len(spectra_by_hpx)} spectra in healpix {healpix}.")


.. parsed-literal::

    There are 71 spectra in healpix 9145.


Let’s investigate one spectrum. Wavelength, flux and inverse variance
are stored as dictionaries similar to
`desispec.spectra.Spectra <https://desispec.readthedocs.io/en/latest/api.html#desispec-spectra>`_.

.. code:: python3

    spec = spectra_by_hpx[3]
    print(spec.wave)
    print(spec.flux)


.. parsed-literal::

    {'B': array([3600. , 3600.8, 3601.6, ..., 5798.4, 5799.2, 5800. ]),
     'R': array([5760. , 5760.8, 5761.6, ..., 7618.4, 7619.2, 7620. ]),
     'Z': array([7520. , 7520.8, 7521.6, ..., 9822.4, 9823.2, 9824. ])}
    {'B': array([0.16013083, 2.1076498 , 6.495008  , ..., 2.2043223 , 1.6862453 ,
        1.8163666 ], dtype=float32),
     'R': array([-1.287594  ,  1.731283  ,  0.62619126, ...,  0.9037217 ,
        1.3648763 ,  1.652868  ], dtype=float32),
     'Z': array([0.83304965, 1.031328  , 1.6591258 , ..., 0.81355166, 1.0301682 ,
        1.0923132 ], dtype=float32)}


.. code:: python3

    plt.figure(figsize=(12, 5))
    for arm, wave_arm in spec.wave.items():
        plt.plot(wave_arm, spec.flux[arm], label=arm, alpha=0.7)
    plt.legend()
    plt.ylim(-1, 5)
    plt.show()



.. image:: ../_static/simple_coadd_showcase_3arms.png


Now, we coadd the arms using inverse variance and replot. The spectrum
attributes will still be dictionaries with a single key ``brz`` no
matter which arms are used to coadd.

.. code:: python3

    spec.simple_coadd()
    print(spec.wave)
    print(spec.flux)


.. parsed-literal::

    {'brz': array([3600. , 3600.8, 3601.6, ..., 9822.4, 9823.2, 9824. ])}
    {'brz': array([0.16013082, 2.10764978, 6.49500805, ..., 0.81355166, 1.03016822,
        1.0923132 ])}


.. code:: python3

    plt.figure(figsize=(12, 5))
    for arm, wave_arm in spec.wave.items():
        plt.plot(wave_arm, spec.flux[arm], label=arm, alpha=0.7, c='k')
    plt.legend()
    plt.ylim(-1, 5)
    plt.show()



.. image:: ../_static/simple_coadd_showcase_coadded.png

