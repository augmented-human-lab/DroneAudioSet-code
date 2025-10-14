import numpy as np
import soundfile as sf

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

