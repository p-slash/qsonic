Simple coadd showcase
=====================

.. code:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    
    import qsonic.catalog
    import qsonic.io

For this example, we are using v0 catalog for iron release. We are going
to read all arms (B, R, Z), but will not read the resolution matrix.

.. code:: python3

    fname = "/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-tests/catalogs/QSO_cat_iron_main_dark_healpix_v0-altbal.fits"
    indir = "/global/cfs/cdirs/desi/spectro/redux/iron/healpix"
    arms = ['B', 'R', 'Z']
    is_mock = False
    skip_resomat = True

First, we read the catalog. Since ``qsonic`` sorts this the catalog by
HPXPIXEL, we can find the unique healpix values and split the catalog
into healpix groups. For example purposes, we are picking a single healpix and
reading all the quasar spectra in that file.

.. code:: python3

    catalog = qsonic.catalog.read_quasar_catalog(fname)
    
    # Group into unique pixels
    unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(catalog, s[1:])
    
    # Pick one healpix to illustrate
    hpx_cat = split_catalog[0]
    healpix = hpx_cat['HPXPIXEL'][0]
    
    spectra_by_hpx = qsonic.io.read_spectra_onehealpix(
        hpx_cat, indir, arms, is_mock, skip_resomat
    )
    
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

    {'B': array([3600. , 3600.8, 3601.6, ..., 5798.4, 5799.2, 5800. ]), 'R': array([5760. , 5760.8, 5761.6, ..., 7618.4, 7619.2, 7620. ]), 'Z': array([7520. , 7520.8, 7521.6, ..., 9822.4, 9823.2, 9824. ])}
    {'B': array([ 1.8225803 ,  4.4297943 , -0.9742097 , ...,  0.36117804,
            1.2084235 ,  0.53845596], dtype=float32), 'R': array([ 0.8617237 ,  0.7800246 ,  1.2475883 , ...,  1.3106785 ,
           -1.0977093 , -0.06496035], dtype=float32), 'Z': array([-0.02818574,  0.47825524,  0.25585857, ...,  0.48654887,
            0.54619426,  0.3381963 ], dtype=float32)}


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

