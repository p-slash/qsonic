Look into output files
========================

.. code:: python3

    import fitsio
    import numpy as np
    import matplotlib.pyplot as plt

.. code:: python3

    fchi2 = fitsio.FITS("Delta-co1/continuum_chi2_catalog.fits")[1]
    fchi2

.. parsed-literal::

    
      file: Delta-co1/continuum_chi2_catalog.fits
      extension: 1
      type: BINARY_TBL
      extname: CHI2_CAT
      rows: 471137
      column info:
        TARGETID            i8  
        Z                   f4  
        HPXPIXEL            i8  
        MPI_RANK            i4  
        MEANSNR             f4  
        RSNR                f4  
        CONT_valid          b1  
        CONT_chi2           f4  
        CONT_dof            i4  
        CONT_x              f4  array[2]
        CONT_xcov           f4  array[4]



.. code:: python3

    chi2_data = fchi2.read()
    is_valid = chi2_data['CONT_valid']

    chi2_v = chi2_data[is_valid]['CONT_chi2'] / chi2_data[is_valid]['CONT_dof']
    
    plt.hist(chi2_v, bins=100)
    plt.axvline(1, c='k')
    plt.xlabel(r"$\chi^2_\nu$")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.show()



.. image:: ../_static/chi2cat_hist.png


.. code:: python3

    fattr = fitsio.FITS("Delta-co1/attributes.fits")
    fattr




.. parsed-literal::

    
      file: Delta-co1/attributes.fits
      mode: READONLY
      extnum hdutype         hduname[v]
      0      IMAGE_HDU       
      1      BINARY_TBL      CONT-1
      2      BINARY_TBL      VAR_FUNC-1
      3      BINARY_TBL      CONT-2
      4      BINARY_TBL      VAR_FUNC-2
      5      BINARY_TBL      CONT-3
      6      BINARY_TBL      VAR_FUNC-3
      7      BINARY_TBL      CONT-4
      8      BINARY_TBL      VAR_FUNC-4
      9      BINARY_TBL      CONT-5
      10     BINARY_TBL      VAR_FUNC-5
      11     BINARY_TBL      CONT-6
      12     BINARY_TBL      VAR_FUNC-6
      13     BINARY_TBL      CONT-7
      14     BINARY_TBL      VAR_FUNC-7



.. code:: python3

    fstats = fitsio.FITS("Delta-co1/var_stats/qsonic-eta-fits-snr0.0-100.0-variance-stats.fits")
    fstats




.. parsed-literal::

    
      file: Delta-co1/var_stats/qsonic-eta-fits-snr0.0-100.0-variance-stats.fits
      mode: READONLY
      extnum hdutype         hduname[v]
      0      IMAGE_HDU       
      1      BINARY_TBL      VAR_STATS
      2      BINARY_TBL      VAR_FUNC
      3      BINARY_TBL      STACKED_FLUX



.. code:: python3

    fstats['VAR_STATS']


.. parsed-literal::

    
      file: Delta-co1/var_stats/qsonic-eta-fits-snr0.0-100.0-variance-stats.fits
      extension: 1
      type: BINARY_TBL
      extname: VAR_STATS
      rows: 2500
      column info:
        wave                f8  
        var_pipe            f8  
        e_var_pipe          f8  
        var_delta           f8  
        e_var_delta         f8  
        mean_delta          f8  
        var2_delta          f8  
        num_pixels          i8  
        num_qso             i8  
        cov_var_delta       f8  array[100]



.. code:: python3

    fstats['VAR_STATS'].read_header()




.. parsed-literal::

    
    XTENSION= 'BINTABLE'           / binary table extension
    BITPIX  =                    8 / 8-bit bytes
    NAXIS   =                    2 / 2-dimensional binary table
    NAXIS1  =                  872 / width of table in bytes
    NAXIS2  =                 2500 / number of rows in table
    PCOUNT  =                    0 / size of special data area
    GCOUNT  =                    1 / one data group (required keyword)
    TFIELDS =                   10 / number of fields in each row
    TTYPE1  = 'wave'               / label for field   1
    TFORM1  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE2  = 'var_pipe'           / label for field   2
    TFORM2  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE3  = 'e_var_pipe'         / label for field   3
    TFORM3  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE4  = 'var_delta'          / label for field   4
    TFORM4  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE5  = 'e_var_delta'        / label for field   5
    TFORM5  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE6  = 'mean_delta'         / label for field   6
    TFORM6  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE7  = 'var2_delta'         / label for field   7
    TFORM7  = 'D'                  / data format of field: 8-byte DOUBLE
    TTYPE8  = 'num_pixels'         / label for field   8
    TFORM8  = 'K'                  / data format of field: 8-byte INTEGER
    TTYPE9  = 'num_qso'            / label for field   9
    TFORM9  = 'K'                  / data format of field: 8-byte INTEGER
    TTYPE10 = 'cov_var_delta'      / label for field  10
    TFORM10 = '100D'               / data format of field: 8-byte DOUBLE
    EXTNAME = 'VAR_STATS'          / name of this binary table extension
    MINNPIX =                  500 / 
    MINNQSO =                   50 / 
    MINSNR  =                    0 / 
    MAXSNR  =                  100 / 
    WAVE1   =               3660.0 / 
    WAVE2   =               6540.0 / 
    NWBINS  =                   25 / 
    IVAR1   =                 0.05 / 
    IVAR2   =              10000.0 / 
    NVARBINS=                  100 / 



.. code:: python3

    nwbins = fstats['VAR_STATS'].read_header()['NWBINS']
    nvarbins = fstats['VAR_STATS'].read_header()['NVARBINS']

    var_stats_data = fstats['VAR_STATS'].read().reshape(nwbins, nvarbins)


    iw = 2
    dat = var_stats_data[iw]
    valid = (dat['num_pixels'] > 500) & (dat['num_qso'] > 50)
    dat = dat[valid]
    
    plt.errorbar(dat['var_pipe'], dat['var_delta'], dat['e_var_delta'], fmt='-')
    plt.xlabel("Pipeline variance")
    plt.ylabel("Observed variance")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
    plt.errorbar(dat['var_pipe'], dat['mean_delta'], fmt='.-')
    plt.xlabel("Pipeline variance")
    plt.ylabel("Observed mean delta")
    plt.xscale("log")
    plt.show()
    
    cov = dat['cov_var_delta'][:, valid]
    norm = np.sqrt(cov.diagonal())
    plt.imshow(cov / np.outer(norm, norm), vmin=-1, vmax=1, cmap=plt.cm.seismic)


.. image:: ../_static/chi2cat_varpipe-obs.png

.. image:: ../_static/chi2cat_varpipe-mean.png

.. image:: ../_static/chi2cat_covariance.png


