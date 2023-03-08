Examples
========

Running qcfit
-------------------------
.. code-block::
    :caption: For an imaginary Y5 DESI quickquasars mock

        srun -n 128 -c 2 qcfit \
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
.. code-block::
    :caption: Here's an example code snippet to use IO interface.

        import numpy as np
        import qcfitter.catalog
        import qcfitter.io

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

        catalog = qcfitter.catalog.read_quasar_catalog(fname)

        # Group into unique pixels
        unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
        split_catalog = np.split(catalog, s[1:])

        # List of Spectrum objects
        spectra_list = []

        # You can parallelize this such that each process reads a healpix.
        # e.g., pool.map(parallel_reading, split_catalog)
        for cat in split_catalog:
            local_specs = qcfitter.io.read_spectra_onehealpix(
                cat, indir, arms, is_mock, skip_resomat
            )
            for spec in local_specs:
                spec.set_forest_region(w1, w2, fw1, fw2)

                if remove_nonforest_pixels:
                    spec.remove_nonforest_pixels()

            spectra_list.extend(local_specs)


