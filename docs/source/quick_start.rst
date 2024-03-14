Quick Start
===========

Running qsonic-fit
-------------------------

``qsonic-fit`` performs the continuum fitting. You need MPI to run this command, therefore you need to use ``mpirun -np NUMBER_OF_TASKS qsonic-fit ...`` in general or ``srun -n NUMBER_OF_TASKS -c NUMBER_OF_CPUS qsonic-fit ...`` on systems with SLURM. Here is a typical qsonic command for SLURM:

.. code-block:: shell

    srun -n 128 -c 2 qsonic-fit \
    --input-dir INPUT_DIRECTORY \
    --catalog QUASAR_CATALOG \
    -o OUTPUT_FOLDER \
    --num-iterations 10 \
    --wave1 3600 --wave2 5500 \
    --forest-w1 1050 --forest-w2 1180 \
    --skip-resomat


.. _edr example and workaround:

Example: DESI early data release
--------------------------------

If you have a NERSC account, e.g. through DESI, DES, LSST-DESC, or other DOE-sponsored projects, DESI early data release (EDR) is available at ``/global/cfs/cdirs/desi/public/edr``. Otherwise follow the instructions to download the spectra using `Globus <https://data.desi.lbl.gov/doc/access/>`_. Note that you need 80 TB storage space. Let us call this directory ``${EDR_DIRECTORY}``. The coadded spectra are in ``${EDR_DIRECTORY}/spectro/redux/fuji/healpix`` and the quasar catalog for the Lyman-alpha forest analysis is ``${EDR_DIRECTORY}/vac/edr/qso/v1.0/QSO_cat_fuji_healpix_only_qso_targets.fits``. QSOnic is designed to work on DESI data release 1 (DR1) (will be publicly available in late 2024/early 2025) and above. Unfortunately, the blinding strategy implemented for DR1 is not compatible with the EDR quasar catalogs, so we need a workaround.

**Workaround for EDR catalogs:** The blinding strategy requires ``LASTNIGHT`` column to be present in the quasar catalog, which is missing from ``QSO_cat_fuji_healpix_only_qso_targets.fits``. Furthermore, we need to limit our analysis to the relevant survey and program for the Lyman-alpha forest clustering analyses which is the 1% survey (SV3) and dark program. Following commands will read the EDR catalog, select SV3 quasars and append a constant ``LASTNIGHT`` column that marks the last night of observing for EDR.


.. code-block:: python

    import fitsio
    from numpy.lib.recfunctions import append_fields

    org_catalog = fitsio.read(
        "${EDR_DIRECTORY}/vac/edr/qso/v1.0/QSO_cat_fuji_healpix_only_qso_targets.fits",
        ext=1)

    # select quasars in SV3-dark and append LASTNIGHT column
    new_catalog = append_fields(
        org_catalog[(org_catalog['SURVEY'] == 'sv3') & (org_catalog['PROGRAM'] == 'dark')],
        'LASTNIGHT', [20210513], dtypes=int, usemask=False)

    with fitsio.FITS(
            "QSO_cat_fuji_healpix_only_qso_targets_sv3_fix.fits", 'rw',
            clobber=True
    ) as fts:
        fts.write(new_catalog, extname='ZCATALOG')


Now we can pass this catalog into ``qsonic-fit`` script as the quasar catalog.

.. code-block:: shell

    srun -n 128 -c 2 qsonic-fit \
    --input-dir ${EDR_DIRECTORY}/spectro/redux/fuji/healpix \
    --catalog QSO_cat_fuji_healpix_only_qso_targets_sv3_fix.fits \
    --keep-surveys sv3 \
    -o OUTPUT_FOLDER \
    --num-iterations 10 \
    --wave1 3600 --wave2 5500 \
    --forest-w1 1040 --forest-w2 1200 \
    --skip-resomat

Note, we passed ``--keep-surveys sv3`` argument to the script, and you still need to pass a valid output folder.  See :ref:`qsonic-fit arguments <qsonic fit arguments>` for all options you can tweak in the continuum fitting.


Reading spectra
---------------

Here's an example code snippet to use IO interface following the EDR instructions above.

.. code-block:: python

    import numpy as np
    import qsonic.catalog
    import qsonic.io

    fname_catalog = "QSO_cat_fuji_healpix_only_qso_targets_sv3_fix.fits"
    indir = "${EDR_DIRECTORY}/spectro/redux/fuji/healpix"
    arms = ['B', 'R']
    is_mock = False
    skip_resomat = True

    # Setup reader function
    readerFunction = qsonic.io.get_spectra_reader_function(
        indir, arms, is_mock, skip_resomat,
        read_true_continuum=False, is_tile=False)

    w1 = 3600.
    w2 = 6000.
    fw1 = 1050.
    fw2 = 1180.

    catalog = qsonic.catalog.read_quasar_catalog(fname_catalog, is_mock=is_mock)

    # Group into unique pixels
    unique_pix, s = np.unique(catalog['HPXPIXEL'], return_index=True)
    split_catalog = np.split(catalog, s[1:])

    # You can parallelize this such that each process reads a healpix.
    # e.g., pool.map(parallel_reading, split_catalog)
    for hpx_cat in split_catalog:
        healpix = hpx_cat['HPXPIXEL'][0]

        spectra_by_hpx = readerFunction(hpx_cat)

        # Do stuff with spectra in this healpix
        ...
