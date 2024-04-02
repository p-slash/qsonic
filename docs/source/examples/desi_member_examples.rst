Examples for DESI members
=========================

Following examples use unpublished data and for DESI members only. Set ``OUTPUT_FOLDER`` to your own destination in all examples.

Y1 no syst. mocks
-----------------

**True continuum**

.. code:: shell

    source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh

    srun -N 1 -n 128 -c 2 qsonic-fit \
    -i /global/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.5-2/spectra-16 \
    --catalog /global/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.5-2/zcat.fits \
    -o OUTPUT_FOLDER \
    --rfdwave 0.8 --skip 0.2 \
    --num-iterations 20 \
    --cont-order -1 \
    --wave1 3600.0 --wave2 6600.0 \
    --forest-w1 1050.0 --forest-w2 1180.0 \
    --mock-analysis \
    --true-continuum \
    --coadd-arms "before" \
    --fiducial-meanflux /dvs_ro/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/ohio-p1d-true-stats-obs3600-6600-rf1050-1180-dw0.8.fits \
    --fiducial-varlss /dvs_ro/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/ohio-p1d-true-stats-obs3600-6600-rf1050-1180-dw0.8.fits \
    --var-fit-eta --var-use-cov --min-rsnr 1.0 --min-forestsnr 0.3


**Continuum fitting**

1st order polynomial. Note we also use the optional fiducial mean flux argument.

.. code:: shell

    source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh

    srun -N 1 -n 128 -c 2 qsonic-fit \
    -i /global/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.5-2/spectra-16 \
    --catalog /global/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.5-2/zcat.fits \
    -o OUTPUT_FOLDER \
    --rfdwave 0.8 --skip 0.2 \
    --num-iterations 20 \
    --cont-order 1 \
    --wave1 3600.0 --wave2 6600.0 \
    --forest-w1 1050.0 --forest-w2 1180.0 \
    --mock-analysis \
    --coadd-arms "before" \
    --fiducial-meanflux /dvs_ro/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/ohio-p1d-true-stats-obs3600-6600-rf1050-1180-dw0.8.fits \
    --var-fit-eta --var-use-cov --min-rsnr 1.0 --min-forestsnr 0.3


Y1 DLA + BAL mocks
------------------


**Continuum fitting**

Masking DLA and BAL features, and sky lines.

.. code:: shell

    source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh

    srun -N 1 -n 128 -c 2 qsonic-fit \
    -i /dvs_ro/cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.r145-2/spectra-16 \
    --catalog /dvs_ro//cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.r145-2/zcat.fits \
    -o OUTPUT_FOLDER \
    --rfdwave 0.8 --skip 0.2 \
    --num-iterations 20 \
    --cont-order 1 \
    --wave1 3600.0 --wave2 6600.0 \
    --forest-w1 1050.0 --forest-w2 1180.0 \
    --mock-analysis \
    --dla-mask /dvs_ro//cfs/cdirs/desicollab/users/naimgk/ohio-p1d/v2.1/iron/main/QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918/v2.1.0/desi-2.r145-2/dla_cat.fits \
    --bal-mask \
    --sky-mask /dvs_ro/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-tests/catalogs/list_mask_p1d_DESI_EDR.txt \
    --coadd-arms "before" \
    --var-fit-eta --var-use-cov --min-rsnr 1.0 --min-forestsnr 0.3


Y1 Data v1 reduction
--------------------

v0 catalog (not up-to-date!) with a fiducial Becker+13 mean flux. DLA, BAL and sky masking.

.. code:: shell

    source /global/cfs/projectdirs/desi/science/lya/scripts/activate_qsonic.sh

    srun -N 1 -n 128 -c 2 qsonic-fit \
    -i /dvs_ro/cfs/cdirs/desi/spectro/redux/iron/healpix \
    --catalog /dvs_ro/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-tests/catalogs/QSO_cat_iron_main_dark_cumulative_v0_zlya-altbal_zwarn_cut_20231214.fits \
    -o OUTPUT_FOLDER \
    --rfdwave 0.8 --skip 0.2 \
    --num-iterations 20 \
    --cont-order 1 \
    --wave1 3600.0 --wave2 6600.0 \
    --forest-w1 1050.0 --forest-w2 1180.0 \
    --dla-mask /dvs_ro/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-tests/catalogs/DLA_cnn_iron_main_dark_healpix_v0-nhi20.3-cnnSNR3.0-highcut0.0-lowcut0.3.fits \
    --bal-mask \
    --sky-mask /dvs_ro/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-tests/catalogs/list_mask_p1d_DESI_EDR.txt \
    --coadd-arms "before" \
    --fiducial-meanflux /dvs_ro/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-tests/catalogs/becker_meanflux.fits \
    --var-fit-eta --var-use-cov --min-rsnr 1.0 --min-forestsnr 0.3
