import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

orig_fs = 48000
target_fs = 16000

# read single channel audio files
def read_audio_signal(file_path, fs, always_2d=True):
    sig, sig_fs = sf.read(file_path, dtype='float32', always_2d=always_2d)
    assert sig_fs == fs
    return sig

# write audio signals, including multi-channel
def write_audio_signal(file_path, sig, fs):
	sf.write(file=file_path, data=sig, samplerate=fs)

# resample multi-channel audio
def multichannel_resample(multi_sig, n_channels, orig_fs, target_fs):
    # multi_sig dimensions (n_samples, n_channels)
    assert (multi_sig.shape[1] == n_channels), 'incorrect number of channels/format'
    # Resample each channel consistently using resample_poly
    resampled_channels = [
        resample_poly(multi_sig[:, idx], up=target_fs, down=orig_fs)  # Resample using the ratio of target to original sampling rates
        for idx in range(n_channels)
    ]
    # Stack resampled channels back into a single multi-channel signal
    resampled_multi_sig = np.stack(resampled_channels, axis=1)
    assert resampled_multi_sig.shape[1] == n_channels
    return resampled_multi_sig

def rectify_offset_and_resample(read_path, write_path, silence_threshold=orig_fs//2):
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