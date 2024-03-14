Simple coadd showcase
=====================

This tutorial builds on the DESI early data release example detailed in :ref:`Quick Start <edr example and workaround>`. I assume you have access to the EDR data, and created ``QSO_cat_fuji_healpix_only_qso_targets_sv3_fix.fits`` file that fixes the compatibility issue.

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
    hpx_cat = split_catalog[0]
    healpix = hpx_cat['HPXPIXEL'][0]

    spectra_by_hpx = readerFunction(hpx_cat)

    print(f"There are {len(spectra_by_hpx)} spectra in healpix {healpix}.")


.. parsed-literal::

    There are 207 spectra in healpix 0.


Let’s investigate one spectrum. Wavelength, flux and inverse variance
are stored as dictionaries similar to
`desispec.spectra.Spectra <https://desispec.readthedocs.io/en/latest/api.html#desispec-spectra>`_.

.. code:: python3

    spec = spectra_by_hpx[0]
    print(spec.wave)
    print(spec.flux)


.. parsed-literal::

    {'B': array([3600. , 3600.8, 3601.6, ..., 5798.4, 5799.2, 5800. ]),
     'R': array([5760. , 5760.8, 5761.6, ..., 7618.4, 7619.2, 7620. ]),
     'Z': array([7520. , 7520.8, 7521.6, ..., 9822.4, 9823.2, 9824. ])}
    {'B': array([ 2.8399339 ,  5.5985265 ,  1.6996089 , ..., -0.02623173,
            0.12586579,  0.17652267], dtype=float32),
     'R': array([ 0.59253263, -4.5705295 ,  0.7594522 , ...,  2.133106  ,
            0.19382767, -0.8677871 ], dtype=float32),
     'Z': array([0.5383575 , 0.4969392 , 0.83893967, ..., 0.44942966, 0.642498  ,
           0.68266016], dtype=float32)}


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
    {'brz': array([ 1.82258034,  4.42979435, -0.97420972, ...,  0.48654886,
            0.54619423,  0.3381963 ])}


.. code:: python3

    plt.figure(figsize=(12, 5))
    for arm, wave_arm in spec.wave.items():
        plt.plot(wave_arm, spec.flux[arm], label=arm, alpha=0.7, c='k')
    plt.legend()
    plt.ylim(-1, 5)
    plt.show()



.. image:: ../_static/simple_coadd_showcase_coadded.png

