import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# read single channel audio files
def read_audio_signal(file_path:str, fs:int, always_2d:bool=True) -> np.ndarray:
    """ Read an audio file and return a float32 Numpy array

    Parameters
    ----------
    file_path: str
        Path to the audio file.
    fs: int
        Expected sampling rate.
    always_2d: bool
        If True, always return a 2D array (n_samples, n_channels).

    Returns
    -------
    np.ndarray
        Audio signal as a float32 Numpy array.
    """

    sig, sig_fs = sf.read(file_path, dtype='float32', always_2d=always_2d)
    assert sig_fs == fs
    return sig

# write audio signals, including multi-channel
def write_audio_signal(file_path: str, sig: np.ndarray, fs: int) -> None:
    """Write an audio signal to disk.

    Accepts NumPy arrays or torch tensors. Multi-channel is supported with shape [T, C].
    """
    sf.write(file=file_path, data=sig, samplerate=fs)

# resample multi-channel audio
def multichannel_resample(multi_sig: np.ndarray, n_channels: int,
                          orig_fs: int, target_fs: int) -> np.ndarray:
    """Resample a multi-channel audio signal to a target sampling rate.

    Parameters
    ----------
    multi_sig : np.ndarray
        Multi-channel audio signal with shape (n_samples, n_channels).
    n_channels : int
        Number of channels in the audio signal.
    orig_fs : int
        Original sampling rate of the audio signal.
    target_fs : int
        Desired sampling rate after resampling.

    Returns
    -------
    np.ndarray
        Resampled multi-channel audio signal with shape (new_n_samples, n_channels).
    """
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

