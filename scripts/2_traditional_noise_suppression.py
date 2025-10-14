"""Traditional multi-mic noise suppression pipeline.

Usage:
- Run `python -m scripts.2_traditional_noise_suppression` to execute the script.

Steps:
1) **MVDR Beamforming** (SpeechBrain):
   - Uses a short segment of the *drone-only* recording to estimate the noise spatial covariance.
   - Computes STFT, builds noise covariance, and applies MVDR with a DOA derived from the mic geometry.
2) **Spectral Gating** (noisereduce):
   - Applies non-stationary spectral gating on the beamformed single-channel output.

By default, the script *prints* operations it would perform. Set `is_save = True` to write outputs.
"""

# ==== Imports ====
import os
import noisereduce as nr #type: ignore
from scripts.config import n_mics, mic_reorder_list, sampling_rate as fs, aggressiveness
from scripts.utils.mic_geometry_util import circular_array_positions
from scripts.algorithms.beamforming import perform_mvdr_beamforming
from scripts.algorithms.noisereduce import perform_spectral_gating

if __name__ == '__main__':
    # chosen setting
    speaker_loudness = '90db-speaker-volume' # options: '60db-speaker-volume','90db-speaker-volume'
    drone = 'drone1' # options: 'drone1','drone2'
    room = 'room1' # options: drone1: 'room1', 'room2', and for drone2: 'room1', 'room3'
    speaker_dist = 'speaker-dist-1m' # options: for room1:'speaker-dist-1m','speaker-dist-3m', 'speaker-dist-5m', for room2/room3: 'speaker-dist-3m','speaker-dist-6m','speaker-dist-9m'
    mic_dist = 'mic-dist-25cm' # options: 'mic-dist-25cm','mic-dist-50cm'
    throttle = 'throttle-high' # options: 'throttle-low','throttle-high' 
    mic = 'mic3_8array-up' # options: 'mic1_soundskrit','mic2_8array_down','mic3_8array-up'
    file_list = [f'File{idx}' for idx in range(1, 7)]

    # set microphone geometry
    if drone == 'drone1':
        mic_diameter = 0.5
    elif drone == 'drone2':
        mic_diameter = 0.3
    mic_geometry = circular_array_positions(mic_diameter/2, n_mics, mic_reorder_list)
    is_save = False # set to True to save the outputs from beamforming and spectral gating
    
    print('='*50)
    print('Chosen Setting:')
    print(f'Source Loudness: {speaker_loudness}\nRoom: {room}\nDrone: {drone}\nDrone-Speaker Distance: {speaker_dist}')
    print(f'Mic: {mic}\nDrone-Mic Distance: {mic_dist}')
    print(f'File List: {file_list}')
    print('='*50)

    # initalize paths
    root_path = './audio-samples' #'../ComputeResourcesCheck/'
    drone_with_source_folder = os.path.join(root_path, 'drone-with-source-recordings')
    drone_only_folder = os.path.join(root_path, 'drone-only-recordings')

    # once downloaded from HF, set root_path as shown below
    # root_path = './data/DroneAudioSet/' # path to the downloaded DroneAudioSet data
    # drone_with_source_folder = os.path.join(root_path, 'drone-with-source-recordings',
    #                                       speaker_loudness, room, drone, speaker_dist,
    #                                       mic_dist, throttle)
    # drone_only_folder = os.path.join(root_path, 'drone-only-recordings',
    #                                drone, mic_dist, throttle)

    perform_mvdr_beamforming(audio_folder=drone_with_source_folder,
                        noise_folder=drone_only_folder,
                        mic_name=mic,
                        mic_dist_str=mic_dist,
                        mic_geometry=mic_geometry,
                        file_list=file_list,
                        speaker_dist_str=speaker_dist,
                        is_save=is_save)
    
    bf_drone_with_source_path = drone_with_source_folder.replace('drone-with-source-recordings', 'outputs/beamforming')
    perform_spectral_gating(audio_folder=bf_drone_with_source_path,
                        mic_name=mic,
                        file_list=file_list,
                        is_save=is_save,
                        thresh=aggressiveness)

