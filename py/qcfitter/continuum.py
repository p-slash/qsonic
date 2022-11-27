import numpy as np
from numba import njit
from scipy.optimize import minimize

from qcfitter.mpi_utils import logging_mpi

@njit
def fast_interp1d(x, xp0, dxp, fp):
    xx = (x - xp0)/dxp
    idx = int(np.clip(xx, 0, fp.size-2))
    d_idx = xx - idx
    y1, y2 = fp[idx], fp[idx+1]

    return y1*(1-d_idx) + y2*d_idx

class ContinuumFitter(object):
    def __init__(self, w1rf, w2rf, dwrf):
        self.nbins = int((w2rf-w1rf)/dwrf)+1
        self.dwrf = dwrf
        self.rfwave = w1rf + np.arange(self.nbins)*dwrf

        self.mean_cont = np.ones(self.nbins)

    def _continuum_chi2(x, wave, flux, ivar, z_qso):
        chi2 = 0

        for arm, wave_arm in wave.items():
            if wave_arm.size == 0:
                continue

            wv = wave_arm/(1+z_qso)
            cont_est = self.get_continuum_model(x, wv)

            chi2 += np.sum(
                ivar[arm] * (flux[arm] - cont_est)**2
            )

        return chi2

    def get_continuum_model(self, x, wave_rf_arm):
        Slope = np.log(wave_rf_arm)
        Slope = (Slope-Slope[0])/(Slope[-1]-Slope[0])

        cont = fast_interp1d(
            wave_rf_arm,
            self.rfwave[0], self.dwrf,
            self.mean_cont
        ) * (x[0] + x[1]*Slope)
        # Multiply with resolution
        # Edges are difficult though

        return cont

    def fit_continuum(self, spectrum):
        result = minimize(
            self._continuum_chi2,
            spectrum.cont_params['x'],
            args=(spectrum.forestwave,
                  spectrum.forestflux,
                  spectrum.forestivar,
                  spectrum.z_qso),
            bounds=[(0, 500), (-20, 20)],
            method=None,
            jac=None
        )

        spectrum.cont_params['valid'] = result.success
        if result.success:
            spectrum.cont_params['x'] = result.x

    def fit_continua(self, spectra_list, comm, mpi_rank):
        no_valid_fits=0
        no_invalid_fits=0

        # For each forest fit continuum
        for spectrum in spectra_list:
            self.cont_params['method'] = 'picca'
            self.fit_continuum(spectrum)

            if not spectrum.cont_params['valid']:
                no_invalid_fits+=1
                logging_mpi(f"mpi:{mpi_rank}:Invalid continuum TARGETID: {spectrum.targetid}.", 0)
            else:
                no_valid_fits += 1

        no_valid_fits = comm.reduce(no_valid_fits, op=MPI.SUM, root=0)
        no_invalid_fits = comm.reduce(no_invalid_fits, op=MPI.SUM, root=0)
        logging_mpi(f"Number of valid fits: {no_valid_fits}", mpi_rank)
        logging_mpi(f"Number of invalid fits: {no_invalid_fits}", mpi_rank)

    def update_mean_cont(self, spectra_list, comm):
        norm_flux = np.zeros_like(self.mean_cont)
        counts = np.zeros_like(self.mean_cont)

        for spectrum in spectra_list:
            if not spectrum.cont_params['valid']:
                continue 

            for arm, wave_arm in spectrum.forestwave.items():
                if wave_arm.size == 0:
                    continue

                wave_rf_arm = wave_arm/(1+spectrum.z_qso)
                bin_idx = ((wave_rf_arm - self.rfwave[0]-self.dwrf/2)/self.dwrf).astype(int)

                cont = self.get_continuum_model(spectrum.cont_params['x'], wave_rf_arm)

                norm_flux  = spectrum.forestflux[arm]/cont
                weight     = spectrum.forestivar[arm]*cont**2
                norm_flux += np.bincount(bin_idx, weights=norm_flux*weight, minlength=self.nbins)
                counts    += np.bincount(bin_idx, weights=weight, minlength=self.nbins)

        norm_flux = comm.allreduce(norm_flux)
        counts = comm.allreduce(counts)
        norm_flux /= counts

        has_converged = np.all(norm_flux < 1e-4)

        self.mean_cont *= norm_flux

        return has_converged

    def iterate(self, spectra_list, niterations, comm, mpi_rank):
        has_converged = False

        for it in range(niterations):
            logging_mpi(f"Fitting iteration {it+1}/{niterations}", mpi_rank)
            # Fit all continua one by one
            self.fit_continua(self, spectra_list, comm, mpi_rank)
            # Stack all spectra in each process
            # Broadcast and recalculate global functions
            has_converged = self.update_mean_cont(spectra_list, comm)

            if has_converged:
                logging_mpi("Iteration has converged.", mpi_rank)
                break

        if not has_converged:
            logging_mpi("Iteration has NOT converged.", mpi_rank, "warning")



        

