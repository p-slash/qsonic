Quick Start
===========
On NERSC, run: ``source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh``

Running qsonic-fit
-------------------------

See :ref:`qsonic-fit arguments <qsonic fit arguments>` for all options.

.. code-block:: shell
    :caption: For an imaginary Y5 DESI quickquasars mock

    srun -n 128 -c 2 qsonic-fit \
    --input-dir desi-y5-1000/spectra-16/ \
    --catalog desi-y5-1000/zcat.fits \
    -o desi-y5-1000/deltas/ \
    --mock-analysis \
    --no-iterations 10 \
    --rfdwave 0.8 --skip 0.2 \
    --wave1 3600 --wave2 5500 \
    --coadd-arms \
    --skip-resomat


Reading spectra
---------------

Here's an example code snippet to use IO interface.

.. code-block:: python

    import numpy as np
    import qsonic.catalog
    import qsonic.io

    fname = "iron_catalog_v0.fits"
    indir = "desi_redux"
    arms = ['B', 'R']
    is_mock = False
    skip_resomat = True

    w1 = 3600.
    w2 = 6000.
    fw1 = 1050.
    fw2 = 1180.
    remove_nonforest_pixels = True # Less memory use

    catalog = qsonic.catalog.read_quasar_catalog(fname)

    # Group into unique pixels
    unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(catalog, s[1:])

    # You can parallelize this such that each process reads a healpix.
    # e.g., pool.map(parallel_reading, split_catalog)
    for hpx_cat in split_catalog:
        healpix = hpx_cat['HPXPIXEL'][0]

        spectra_by_hpx = qsonic.io.read_spectra_onehealpix(
            hpx_cat, indir, arms, is_mock, skip_resomat
        )

        # Do stuff with spectra in this healpix
        ...
