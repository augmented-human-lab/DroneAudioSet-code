# ==== Imports ====
import os
from typing import List
import noisereduce as nr #type: ignore
from scripts.config import sampling_rate as fs, aggressiveness
from scripts.utils.audio_processing import (read_audio_signal, write_audio_signal)

def perform_spectral_gating(audio_folder: str, mic_name: str,
                            file_list: List[str], is_save: bool,
                            thresh:float=aggressiveness) -> None:
    """Apply spectral gating (noisereduce) to beamformed files.

    Parameters
    ----------
    audio_folder : str
        Folder that contains beamformed outputs (i.e., `outputs/beamforming`).
    mic_name : str
        Microphone rig identifier (used in filenames).
    file_list : List[str]
        Filenames without extension (e.g., "File1", "File2").
    is_save : bool
        If True, write the enhanced wavs under `outputs/spectral-gating`.
    thresh : float, optional
        Aggressiveness of noise reduction (default from config.py).
    """
    print("Performing Spectral Gating using NoiseReduce")

    for file_name in file_list:
        print(f'File: {file_name}')
        bf_path = os.path.join(audio_folder, f'mvdr-{mic_name}-{file_name}.wav')
        bf_sig = read_audio_signal(bf_path, fs)
        assert bf_sig.shape[1] == 1, 'Input signal should be single channel!'
        nr_sig = nr.reduce_noise(y=bf_sig[:,0], sr=fs, stationary=False, 
                                    thresh_n_mult_nonstationary=thresh)
        # save audio
        if is_save:
            nr_folder = audio_folder.replace('beamforming', 'spectral-gating')
            os.makedirs(nr_folder, exist_ok=True)
            save_path = os.path.join(nr_folder, f'nr-{mic_name}-{file_name}-agg{aggressiveness}.wav')
            write_audio_signal(save_path, nr_sig, fs)