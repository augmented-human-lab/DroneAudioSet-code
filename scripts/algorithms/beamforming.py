# ==== Imports ====
import os
import torch
from typing import List
from speechbrain.processing.features import STFT, ISTFT # type: ignore
from speechbrain.processing.multi_mic import Covariance, Mvdr # type: ignore
from scripts.config import sampling_rate as fs
from scripts.utils.mic_geometry_util import compute_doa_from_location
from scripts.utils.audio_processing import read_audio_signal, write_audio_signal

def perform_mvdr_beamforming(audio_folder: str, noise_folder: str,
                        mic_name:str, mic_dist_str:str,
                        mic_geometry: torch.Tensor, file_list: List[str],
                        speaker_dist_str: str,
                        is_save: bool,
                        N_FFT=2048, NOISE_SEGMENT_START=30, NOISE_SEGMENT_END=40) -> None:
    
    
    """Run MVDR beamforming using SpeechBrain components.

    Parameters
    ----------
    audio_folder : str
        Path to *drone-with-source* multichannel recordings.
    noise_folder : str
        Path to *drone-only* multichannel recordings (same naming as audio).
    mic_name : str
        Microphone identifier (used in filenames and vertical offset logic).
    mic_dist_str : str
        Distance string like "mic-dist-25cm" used to compute z-offset.
    mic_geometry : torch.Tensor
        (M, 3) microphone xyz positions in meters (2D circle in xy-plane).
    file_list : List[str]
        Filenames without extension (e.g., "File1", "File2").
    speaker_dist_str : str
        Distance string like "speaker-dist-1m" used to compute DOA.
    is_save : bool
        If True, write the beamformed wavs under `outputs/beamforming`.
    """
    print("Performing MVDR Beamforming using SpeechBrain")

    for file_name in file_list:
        print(f'File: {file_name}')
        audio_path = os.path.join(audio_folder, f'{mic_name}-{file_name}.wav')
        noise_path = os.path.join(noise_folder, f'{mic_name}-{file_name}.wav')
        # === Load multichannel audio ===
        audio_sig_orig = read_audio_signal(audio_path, fs)
        audio_sig = torch.tensor(audio_sig_orig, dtype=torch.float32) # convert to tensor
        audio_sig = audio_sig.unsqueeze(0) # dim: [1, time, channels]
        # === Load multichannel noise ===
        noise_sig_orig = read_audio_signal(noise_path, fs)
        # retain only a small sample noise -- taking samples in the middle to model stationary noise
        noise_sig_orig = noise_sig_orig[(NOISE_SEGMENT_START*fs):(NOISE_SEGMENT_END*fs), :] 
        noise_sig = torch.tensor(noise_sig_orig, dtype=torch.float32)
        noise_sig = noise_sig.unsqueeze(0) # dim: [1, time, channels]
        # === initialize modules ===
        stft = STFT(sample_rate=fs, n_fft=N_FFT)
        cov = Covariance()
        istft = ISTFT(sample_rate=fs, n_fft=N_FFT)
        mvdr = Mvdr()
        # === compute STFT and Covariance ===
        Xs = stft(audio_sig)
        Ns = stft(noise_sig)
        NNs = cov(Ns)
        # === match the number of time steps across noise and audio ===
        audio_time_steps = Xs.shape[1]
        noise_time_steps = NNs.shape[1]
        if noise_time_steps < audio_time_steps:
            num_repeats = (audio_time_steps // noise_time_steps) + 1
            NNs_repeated = NNs.repeat(1, num_repeats, 1, 1, 1)
            NNs_repeated = NNs_repeated[:, :audio_time_steps, :, :, :]
        assert Xs.shape[1] == NNs_repeated.shape[1], "Incompatible time steps!"
        # compute DOA from source location
        doas = compute_doa_from_location(speaker_str=speaker_dist_str,
                                        mic_dist_str=mic_dist_str,
                                        mic_name_str=mic_name,
                                        num_windows=Xs.shape[1])
        # compute MVDR and obtain the beamformed signal
        Ys_mvdr = mvdr(Xs, NNs_repeated, doas, doa_mode=True, mics=mic_geometry, fs=fs)
        beamformed_sig = istft(Ys_mvdr)
        if is_save:
            bf_folder = audio_folder.replace('drone-with-source-recordings', 'outputs/beamforming')
            os.makedirs(bf_folder, exist_ok=True)
            save_path = os.path.join(bf_folder, f'mvdr-{mic_name}-{file_name}.wav')
            write_audio_signal(save_path, beamformed_sig.squeeze(0), fs)