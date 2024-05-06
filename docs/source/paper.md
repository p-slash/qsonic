---
title: 'QSOnic: fast quasar continuum fitting'
tags:
  - Python
  - Cosmology
  - Quasars
  - Lyman-alpha forest
authors:
  - firstname: Naim Göksel
    surname: Karaçaylı
    orcid: 0000-0001-7336-8912
    corresponding: true
    affiliation: "1, 2, 3"
affiliations:
 - name: Center for Cosmology and AstroParticle Physics, The Ohio State University, 191 West Woodruff Avenue, Columbus, OH 43210, USA
   index: 1
 - name: Department of Astronomy, The Ohio State University, 4055 McPherson Laboratory, 140 W 18th Avenue, Columbus, OH 43210, USA
   index: 2
 - name: Department of Physics, The Ohio State University, 191 West Woodruff Avenue, Columbus, OH 43210, USA
   index: 3
date: 18 January 2024
bibliography: paper.bib

---

# Summary

Distant quasars show absorption features at wavelengths shorter than the Ly$\alpha$ emission line in their spectrum due to intervening neutral hydrogen gas. The Ly$\alpha$ forest technique uses these absorption lines to map out the matter distribution in vast volumes when the universe was 1.5--3 billion years old ($2\lesssim z \lesssim 5$). This map of the cosmos has successfully constrained cosmological models using the Baryon Acoustic Oscillations feature in the 3D correlation function [@bourbouxCompletedSDSSIVExtended:2020], various dark matter models [@armengaudConstrainingDarkMatter:2017; @irsicConstraintsWDM:2017] and the sum of the neutrino masses [@palanqueDelabrouilleNeutrinoMass:2015] using the 1D Ly$\alpha$ forest power spectrum [@karacayliOptimal1DLy:2022; @ravouxDesiLyaP1d:2023; @karacayliOptimal1DLy:2024].

The density of the intervening gas is inferred from the transmitted flux fraction, which is the absorption depth relative to the unabsorbed quasar continuum. Recent studies have devised a simple continuum model to homogenize the analysis across quasars so that the errors in the estimated continuum are contained to well-known modes, such as the amplitude and slope of the quasar continuum. This model assumes each quasar continuum $C_q$ is a scaling of a mean quasar continuum shape $\bar C$ by a polynomial of $\ln\lambda$. For example, an order 1 polynomial introduces amplitude $a_q$ and slope $b_q$ terms for each quasar such that $C_q = (a_q + b_q \ln \lambda) \bar{C}$. The quasar continuum fitting algorithm is then an iterative solution that minimizes $\chi^2$ for each quasar and updates the mean quasar continuum using the stack of all quasar continua after every iteration [@bourbouxCompletedSDSSIVExtended:2020; @karacayliOptimal1DLy:2024]. This method requires efficient parallelization due to the computational time and memory needed.


# Statement of need

`QSOnic` is an MPI-parallelized, highly optimized Python package specifically built for the Dark Energy Spectroscopic Instrument (DESI), which will collect approximately one million Ly$\alpha$ quasar spectra in the near future [@desicollaborationDESIExperimentPart:2016]. `QSOnic` is designed to overcome the numerical challenge posed by this large quantity of spectra by MPI parallelization that can distribute the memory and workload across multiple compute nodes. Each parallel process is responsible for independent sky regions based on HEALPix [@healpix]. Reading the data and fitting the continuum are done in parallel. After all quasars are fit with a continuum, all processes synchronize to calculate the mean continuum and check for convergence.

Furthermore, `QSOnic` provides an efficient API to read and manipulate DESI quasar spectra to allow implementation of other algorithms within the Ly$\alpha$ forest framework. It saves detailed intermediate data products including best-fit parameters for each quasar and the covariance matrix used in the pipeline noise calibration correction [$\eta$ parameter in references, @bourbouxCompletedSDSSIVExtended:2020; @karacayliOptimal1DLy:2024]. These intermediate products are crucial to ascertain the precision of the fitted continua and the pipeline noise corrections.


`QSOnic` is built on the same principle of [picca](https://github.com/igmhub/picca) continuum fitting algorithm. Unlike `picca`, `QSOnic` has robust convergence criteria, provides a generic API for DESI quasar spectra, and saves more intermediate data products. Due to optimized IO routines, MPI parallelization and efficient numerical algorithms, it may be faster than `picca`. DESI Ly$\alpha$ forest working group and its ancillary topical groups are responsible for coordinating the efforts between different packages.

`QSOnic` will be used in future scientific publications of the 1D Ly$\alpha$ forest power spectrum and the 3D correlation function measurements from DESI data.

# References
