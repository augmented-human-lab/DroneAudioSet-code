"""Neural Noise Suppression Script using MPSENET on Beamformed DroneAudioSet Recordings.

This script performs neural noise suppression on beamformed audio recordings from the DroneAudioSet dataset using the MPSENET model.

Usage:
    python -m scripts.3_neural_noise_suppression
    Set `is_save = True` to run MPSENET and save outputs.

Steps:
1. Iterates through the beamformed audio files in the source directory.
2. Splits each multi-channel audio file into separate channel and time segments.
3. Processes these segments through the MPSENET model for noise suppression.
4. Stitches the processed segments back into full-length audio files.
5. Saves the denoised outputs in the target directory, maintaining the directory structure.
6. Cleans up temporary intermediate files.

External Dependencies:
- The MPSENET repository must be cloned and installed prior to running this script.
- The checkpoint file for MPSENET inference should be available at 'best_ckpt/g_best_dns'.
"""

import os
import shutil
import subprocess
from scripts.utils.mpsenet_util import (split_wav_by_channels_and_time,
                                        stitch_wav_files)

def process_mpsenet(input_folder:str, output_folder:str) -> None:
    """
    Run MPSENET noise suppression on split wav files.

    Parameters:
    - input_folder (str): Directory containing input noisy wav segments.
    - output_folder (str): Directory where processed wav files will be saved.

    This function executes the MPSENET inference script as a subprocess,
    passing the input and output directories along with the checkpoint file.
    """
    os.makedirs(output_folder, exist_ok=True)
    if any(f.endswith(".wav") for f in os.listdir(output_folder)):
        return  # Skip if already processed
    cmd = f"python mpsenet/inference.py --checkpoint_file best_ckpt/g_best_dns --input_noisy_wavs_dir {input_folder} --output_dir {output_folder}"
    subprocess.run(cmd, shell=True, check=True)

def main(source_root: str, target_root: str, is_save: bool) -> None:
    """
    Main processing loop over the beamformed audio dataset.

    Iterates through all files in the source_root directory, splits multi-channel wav files,
    runs noise suppression using MPSENET on the split segments, stitches the processed segments
    back together, and saves the final output in the target_root directory.

    Temporary folders are created for intermediate files and deleted after processing each folder.

    Parameters:
    - source_root (str): Root directory containing beamformed audio files.
    - target_root (str): Root directory where denoised outputs will be saved.
    - is_save (bool): Flag to enable or disable processing and saving outputs.

    """
    if not is_save:
        print("is_save is set to False. Exiting without processing.")
        return
    for root, _, files in os.walk(source_root):
        relative_path = os.path.relpath(root, source_root)
        new_root = os.path.join(target_root, relative_path)
        noisy_wav_folder = os.path.join(new_root, "noisy_wavfiles")
        processed_wav_folder = os.path.join(new_root, "generated_files")
        final_output_folder = new_root
        # create new folders
        os.makedirs(noisy_wav_folder, exist_ok=True)
        os.makedirs(processed_wav_folder, exist_ok=True)
        os.makedirs(final_output_folder, exist_ok=True)
        ### skip if wav file already exists in output folder
        # List all files in the directory (not subdirectories)
        checkwavfiles = [f for f in os.listdir(final_output_folder) 
                if os.path.isfile(os.path.join(final_output_folder, f))]
        # Check if any file ends with .wav (case-insensitive)
        has_wav = any(f.lower().endswith('.wav') for f in checkwavfiles)
        if has_wav:
            print("skipping",final_output_folder) 
            continue
        for file in files:
            if "File" in file and file.endswith(".wav"):
                file_path = os.path.join(root, file)
                _ = split_wav_by_channels_and_time(file_path, noisy_wav_folder)
        process_mpsenet(noisy_wav_folder, processed_wav_folder)
        for file in files:
            if "File" in file and file.endswith(".wav"):
                stitch_wav_files(os.path.join(root, file), processed_wav_folder, final_output_folder)
        # delete temp folders
        shutil.rmtree(noisy_wav_folder)
        shutil.rmtree(processed_wav_folder)
       
if __name__ == "__main__":
    # chosen setting
    speaker_loudness = '90db-speaker-volume' # options: '60db-speaker-volume','90db-speaker-volume'
    drone = 'drone1' # options: 'drone1','drone2'
    room = 'room1' # options: drone1: 'room1', 'room2', and for drone2: 'room1', 'room3'
    speaker_dist = 'speaker-dist-1m' # options: for room1:'speaker-dist-1m','speaker-dist-3m', 'speaker-dist-5m', for room2/room3: 'speaker-dist-3m','speaker-dist-6m','speaker-dist-9m'
    mic_dist = 'mic-dist-25cm' # options: 'mic-dist-25cm','mic-dist-50cm'
    throttle = 'throttle-high' # options: 'throttle-low','throttle-high' 
    mic = 'mic3_8array-up' # options: 'mic1_soundskrit','mic2_8array_down','mic3_8array-up'
    file_list = [f'File{idx}' for idx in range(1, 7)]
    is_save = False # set this to True to run mpsenet and save outputs

    print('='*50)
    print('Chosen Setting:')
    print(f'Source Loudness: {speaker_loudness}\nRoom: {room}\nDrone: {drone}\nDrone-Speaker Distance: {speaker_dist}')
    print(f'Mic: {mic}\nDrone-Mic Distance: {mic_dist}')
    print(f'File List: {file_list}')
    print('='*50)

    # initalize paths
    root_path = './audio-samples' #'../ComputeResourcesCheck/'
    source_folder = os.path.join(root_path, 'outputs', 'beamforming')
    target_folder = os.path.join(root_path, 'outputs', 'mpsenet')

    # once downloaded from HF, set root_path as shown below
    # root_path = './data/DroneAudioSet/' # path to the downloaded DroneAudioSet data
    # source_folder = os.path.join(root_path, 'outputs', 'beamforming',
    #                                       speaker_loudness, room, drone, speaker_dist,
    #                                       mic_dist, throttle)
    # target_folder = os.path.join(root_path, 'outputs', 'mpsenet',
    #                                       speaker_loudness, room, drone, speaker_dist,
    #                                       mic_dist, throttle)
    main(source_root=source_folder, target_root=target_folder, is_save=is_save)
    
