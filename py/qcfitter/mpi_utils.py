import numpy as np

def balance_load(split_catalog, mpi_size, mpi_rank):
    number_of_spectra = np.zeros(mpi_size, dtype=int)
    local_queue = []
    for cat in split_catalog:
        min_idx = np.argmin(number_of_spectra)
        number_of_spectra[min_idx] += cat.size

        if min_idx == mpi_rank:
            local_queue.append(cat)

    return local_queue
