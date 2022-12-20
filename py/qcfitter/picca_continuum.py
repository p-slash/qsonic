import numpy as np
import fitsio
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from mpi4py import MPI

from qcfitter.mpi_utils import logging_mpi
from qcfitter.mathtools import Fast1DInterpolator

class PiccaContinuumFitter(object):
    """ Picca continuum fitter. Fits spectra without coadding.
    Pipeline ivar is smoothed before using in weights.
    Mean continuum is smoothed using inverse weights and cubic spline.
    """
    def _set_fiducials(self, fiducial_fits):
        if self.mpi_rank == 0:
            with fitsio.FITS(fiducial_fits) as fts:
                data = fts['STATS'].read()
            waves = data['LAMBDA']
            waves_0 = waves[0]
            dwave = waves[1]-waves[0]
            nsize = waves.size
            if not np.allclose(np.diff(waves), dwave):
                raise Exception(
                    "Failed to construct fiducial mean flux and varlss from "
                    f"{fiducial_fits}::LAMBDA is not equally spaced."
                )

            meanflux = data['MEANFLUX']
            varlss   = data['VAR']
        else:
            waves_0 = 0.
            dwave   = 0.
            nsize   = 0

        nsize    = self.comm.bcast(nsize)
        waves_0  = self.comm.bcast(waves_0)
        dwave    = self.comm.bcast(dwave)

        if self.mpi_rank != 0:
            meanflux = np.empty(nsize, dtype=float)
            varlss   = np.empty(nsize, dtype=float)

        self.comm.Bcast([meanflux, MPI.DOUBLE])
        self.comm.Bcast([varlss, MPI.DOUBLE])

        self.meanflux_interp = Fast1DInterpolator(waves_0, dwave,
            meanflux)
        self.varlss_interp = Fast1DInterpolator(waves_0, dwave,
            varlss)

    def __init__(self, w1rf, w2rf, dwrf, comm, mpi_rank, fiducial_fits=None):
        self.nbins = int((w2rf-w1rf)/dwrf)+1
        self.dwrf = dwrf
        self.rfwave = w1rf + np.arange(self.nbins)*dwrf
        self._denom = np.log(self.rfwave[-1]/self.rfwave[0])

        self.mean_cont = np.ones(self.nbins)
        self.meancont_interp = Fast1DInterpolator(w1rf, dwrf,
            self.mean_cont)

        self.comm = comm
        self.mpi_rank = mpi_rank

        if fiducial_fits:
            self._set_fiducials(fiducial_fits)
        else:
            self.meanflux_interp = Fast1DInterpolator(0., 1., np.ones(3))
            self.varlss_interp   = Fast1DInterpolator(0., 1., np.zeros(3))

    def _continuum_chi2(self, x, wave, flux, ivar_sm, z_qso):
        chi2 = 0

        for arm, wave_arm in wave.items():
            cont_est  = self.get_continuum_model(x, wave_arm/(1+z_qso))
            no_neg = np.sum(cont_est<0)
            penalty = wave_arm.size * no_neg**2

            cont_est *= self.meanflux_interp(wave_arm)

            var_lss = self.varlss_interp(wave_arm)*cont_est**2
            weight  = ivar_sm[arm] / (1+ivar_sm[arm]*var_lss)
            w = weight > 0

            chi2 += np.sum(
                weight * (flux[arm] - cont_est)**2
            ) - np.log(weight[w]).sum() + penalty

        return chi2

    def get_continuum_model(self, x, wave_rf_arm):
        Slope = np.log(wave_rf_arm/self.rfwave[0])/self._denom

        cont = self.meancont_interp(wave_rf_arm) * (x[0] + x[1]*Slope)
        # Multiply with resolution
        # Edges are difficult though

        return cont

    def fit_continuum(self, spec):
        # constr = ({'type': 'eq',
        #            'fun': lambda x: self.get_continuum_model(x, wave_arm/(1+spec.z_qso))
        #           } for wave_arm in spec.forestwave.values() if wave_arm.size>0
        # )

        result = minimize(
            self._continuum_chi2,
            spec.cont_params['x'],
            args=(spec.forestwave,
                  spec.forestflux,
                  spec.forestivar_sm,
                  spec.z_qso),
            # constraints=constr,
            # method='COBYLA',
            bounds=[(0, None), (None, None)],
            jac=None
        )

        spec.cont_params['valid'] = result.success

        if result.success:
            for wave_arm in spec.forestwave.values():
                _cont = self.get_continuum_model(result.x, wave_arm/(1+spec.z_qso))

                if any(_cont<0):
                    spec.cont_params['valid'] = False
                    break

        if spec.cont_params['valid']:
            spec.cont_params['x'] = result.x
            spec.cont_params['cont'] = {}
            for arm, wave_arm in spec.forestwave.items():
                _cont  = self.get_continuum_model(result.x, wave_arm/(1+spec.z_qso))
                _cont *= self.meanflux_interp(wave_arm)
                spec.cont_params['cont'][arm] = _cont
        else:
            spec.cont_params['cont'] = None

    def fit_continua(self, spectra_list):
        no_valid_fits=0
        no_invalid_fits=0

        # For each forest fit continuum
        for spec in spectra_list:
            spec.cont_params['method'] = 'picca'
            self.fit_continuum(spec)

            if not spec.cont_params['valid']:
                no_invalid_fits+=1
                # logging_mpi(f"mpi:{mpi_rank}:Invalid continuum TARGETID: {spec.targetid}.", 0)
            else:
                no_valid_fits += 1

        no_valid_fits = self.comm.reduce(no_valid_fits, root=0)
        no_invalid_fits = self.comm.reduce(no_invalid_fits, root=0)
        logging_mpi(f"Number of valid fits: {no_valid_fits}", self.mpi_rank)
        logging_mpi(f"Number of invalid fits: {no_invalid_fits}", self.mpi_rank)

    def update_mean_cont(self, spectra_list):
        norm_flux = np.zeros_like(self.mean_cont)
        counts = np.zeros_like(self.mean_cont)

        for spec in spectra_list:
            if not spec.cont_params['valid']:
                continue 

            for arm, wave_arm in spec.forestwave.items():
                wave_rf_arm = wave_arm/(1+spec.z_qso)
                bin_idx = ((wave_rf_arm - self.rfwave[0])/self.dwrf + 0.5).astype(int)

                cont  = self.get_continuum_model(spec.cont_params['x'], wave_rf_arm)
                cont *= self.meanflux_interp(wave_arm)

                flux_  = spec.forestflux[arm]/cont
                # Deconvolve resolution matrix ?

                var_lss = self.varlss_interp(wave_arm)
                weight  = spec.forestivar_sm[arm]*cont**2
                weight  = weight / (1+weight*var_lss)

                norm_flux += np.bincount(bin_idx, weights=flux_*weight, minlength=self.nbins)
                counts    += np.bincount(bin_idx, weights=weight, minlength=self.nbins)

        self.comm.Allreduce(MPI.IN_PLACE, norm_flux)
        self.comm.Allreduce(MPI.IN_PLACE, counts)
        norm_flux /= counts
        std_flux = 1/np.sqrt(counts)

        # Raw update
        # self.mean_cont *= norm_flux
        # norm_flux -= 1

        # Smooth new estimates
        spl = UnivariateSpline(self.rfwave, norm_flux-1, w=1/std_flux)
        norm_flux = spl(self.rfwave)
        self.mean_cont *= 1+norm_flux

        logging_mpi("Continuum updates", self.mpi_rank)
        for _w, _c, _e in zip(self.rfwave, norm_flux, std_flux):
            logging_mpi(f"{_w:10.2f}: 1+({_c:10.2e}) pm {_e:10.2e}", self.mpi_rank)

        has_converged = np.all(np.abs(norm_flux) < 1.5*std_flux)
        # np.allclose(np.abs(norm_flux), std_flux, rtol=0.5)

        return has_converged

    def iterate(self, spectra_list, niterations):
        has_converged = False

        for it in range(niterations):
            logging_mpi(f"Fitting iteration {it+1}/{niterations}", self.mpi_rank)
            # Fit all continua one by one
            self.fit_continua(spectra_list)
            # Stack all spectra in each process
            # Broadcast and recalculate global functions
            has_converged = self.update_mean_cont(spectra_list)

            if has_converged:
                logging_mpi("Iteration has converged.", self.mpi_rank)
                break

        if not has_converged:
            logging_mpi("Iteration has NOT converged.", self.mpi_rank, "warning")



        

