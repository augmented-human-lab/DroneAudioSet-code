"""This script rectifies offsets and resamples multichannel audio files.
It has already been executed before uploading the dataset to Hugging Face,
so it does not need to be re-run. It assumes the original sampling rate is 48kHz and resamples to 16kHz.

Usage
    Run `python -m scripts.0_preprocess_data` to execute the script.
    Ensure to set `read_path` and `write_path` variables before running.
"""

import os
import numpy as np
import soundfile as sf
from scripts.utils.audio_processing import (read_audio_signal, write_audio_signal,
                                    multichannel_resample)

orig_fs = 48000
target_fs = 16000

def rectify_offset_and_resample(read_path: str, write_path: str,
                                silence_threshold: int = orig_fs//2) -> None:
    """
    Removes initial silence/offset from multichannel audio and resamples from 48kHz to 16kHz.
    """
    if read_path.lower().endswith('.wav'):
        data = read_audio_signal(file_path=read_path, fs=orig_fs)
        if 'soundskrit' in read_path:
            n_mics = 6
        else:
            n_mics = 8
        assert data.shape[1] == n_mics
        start_idx = orig_fs//2
        for ch_idx in np.arange(n_mics):
            if ('soundskrit' in read_path) and (ch_idx > 1): # consider only first channel
                break
            start_idx_ch = np.argwhere(data[:, ch_idx])[0][0] # first non-zero element
            if start_idx_ch < start_idx:
                start_idx = start_idx_ch
        assert start_idx < silence_threshold, f"offset: {start_idx/orig_fs}s" # offset is always less than 0.5 seconds
        print(f'File: {read_path}, Offset: {start_idx}')
        data_without_offset = data[start_idx:, :]
        resampled_data = multichannel_resample(multi_sig=data_without_offset,
                                                    n_channels=n_mics,
                                                    orig_fs=orig_fs,
                                                    target_fs=target_fs)
        write_audio_signal(file_path=write_path, sig=resampled_data, fs=target_fs)

if __name__ == "__main__":
    read_path = '' # read audio file ending with .wav
    write_path = '' # provide write path
    rectify_offset_and_resample(read_path, write_path)