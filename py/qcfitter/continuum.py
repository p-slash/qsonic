import numpy as np
from scipy.optimize import minimize

from qcfitter.mpi_utils import logging_mpi

class ContinuumFitter(object):
    def __init__(self, w1rf, w2rf, dwrf):
        self.nbins = int((w2rf-w1rf)/dwrf)+1
        self.rfwave = w1rf + np.arange(self.nbins)*dwrf

        self.mean_cont = np.ones(self.nbins)

    def _continuum_chi2(x, wave_rf, flux, ivar):
        chi2 = 0
        cont_est = self.get_continuum_model(x, wave_rf)

        for arm in wave_rf.keys():
            wv = wave_rf[arm]
            if wv.size == 0:
                continue

            chi2 += np.sum(
                ivar[arm] * (flux[arm] - cont_est[arm])**2
            )

        return chi2

    def get_continuum_model(self, x, wave_rf):
        cont = {}

        for arm in wave_rf.keys():
            wv = wave_rf[arm]
            Slope = np.log(wv)
            Slope = (Slope-Slope[0])/(Slope[-1]-Slope[0])

            cont[arm] = np.interp(
                wv,
                self.rfwave,
                self.mean_cont
            ) * (x[0] + x[1]*Slope)
            # Multiply with resolution

        return cont

    def fit_continuum(self, spectrum):
        wave_rf = {arm: wave/(1+spectrum.z_qso) for arm, wave in spectrum.forestwave.items()}

        result = minimize(
            self._continuum_chi2,
            spectrum.cont_params['x'],
            args=(wave_rf, norm_flux, spectrum.forestivar),
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

    def stack_normalized_flux(self, spectra_list):
        norm_flux = np.zeros_like(self.mean_cont)

        for spectrum in spectra_list:
            wave_rf = {arm: wave/(1+spectrum.z_qso) for arm, wave in spectrum.forestwave.items()}
            cont = self.get_continuum_model(spectrum.cont_params['x'], wave_rf)
            norm_flux = {arm: flux/cont[arm] for arm, flux in spectrum.forestflux.items()}
            #####

    def iterate(self, spectra_list, comm, mpi_rank):
        self.fit_continua(self, spectra_list, comm, mpi_rank)
        # Stack all spectra in each process
        # Broadcast and recalculate global functions


        

