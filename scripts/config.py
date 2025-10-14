import numpy as np

# global parameters
sampling_rate = 16000

# MVDR beamforming
n_mics = 8
mic_reorder_list = np.array([6, 5, 0, 3, 7, 4, 1, 2]) # mic position and channel mapping

# initialize parameters for Spectral Gating
aggressiveness = 0.5


