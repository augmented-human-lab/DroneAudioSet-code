"""
Script to download and optionally save DroneAudioSet subsets from Hugging Face.

Usage:
    - Set `is_save_locally=True` to save files to `root_folder`.
    - Optionally set `cache_dir` to redirect Hugging Face cache to a drive with >45 GB free space.

Total dataset size: ~42.6 GB
"""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset

def load_hf_dataset(hf_repo: str, subset_name: str, cache_dir: str) -> dict:
    """
    Load a subset of the DroneAudioSet dataset from Hugging Face.

    Parameters:
    hf_repo (str): The Hugging Face repository name.
    subset_name (str): The name of the subset to load.

    Returns:
    dict: The loaded dataset subset.
    """
    # Load the specified subset from the Hugging Face repository
    if cache_dir is None:
        ds = load_dataset(hf_repo, subset_name) # uses default cache 
    else:
        os.makedirs(cache_dir, exist_ok=True)
        ds = load_dataset(hf_repo, subset_name, cache_dir=cache_dir)
    return ds

def save_audio_dataset(ds: dict, root_folder: str) -> None:
    """
    Save audio data from the dataset to the specified local directory.

    Parameters:
    ds (dict): The dataset dictionary containing splits and audio data.
    root_folder (str): The root directory where audio files will be saved.
    """
    # Use default directory if none provided
    if root_folder is None or root_folder == '':
        raise ValueError("Please provide a valid root_folder path to save audio files.")
    # Iterate over top-level dataset splits (e.g., train_001)
    for top_split_key in ds.keys():
        for inner_split in ds[top_split_key]:
            # Extract relative path, audio signal, and sampling rate
            rel_path = inner_split['audio']['path']
            sig = inner_split['audio']['array']
            sig = np.array(sig)
            sr = inner_split['audio']['sampling_rate']
            # save data of shape [n_samples, n_channels] using soundfile
            save_path = os.path.join(root_folder, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sf.write(save_path, sig, sr)
            print(f"Saved {rel_path} with shape {sig.shape} and sampling rate {sr}")

if __name__ == "__main__":
    # When run directly, this script downloads specified DroneAudioSet subsets
    # from Hugging Face and saves them locally in the defined folder structure.
    hf_repo = "ahlab-drone-project/DroneAudioSet"
    audio_subset_list = ['ground-truth', 'drone-only', 'source-only', 'drone-with-source']
    is_save_locally = False # set this to True for saving
    if not is_save_locally:
        print("Dry run only. Set is_save_locally=True to actually save files.")
        exit()
    root_folder = '' # update this to where you want the downloaded data
    # Loop through each subset, load it, and save the audio files locally
    for subset_name in audio_subset_list:
        # make sure the cache directory has enough space (~43GB)
        # cache_dir = ''
        ds = load_hf_dataset(hf_repo=hf_repo, subset_name=subset_name, cache_dir=None)
        save_audio_dataset(ds=ds, root_folder=root_folder)