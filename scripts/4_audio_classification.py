"""This script performs audio classification on segments extracted from audio files using the SSLAM model inference pipeline.

Steps:
1. Scans a specified directory for audio files matching certain naming patterns.
2. For each audio file, reads corresponding timestamp files that define segments and their sound classes.
3. Extracts audio segments using ffmpeg, optionally selecting a specific channel.
4. Runs SSLAM model inference on each extracted segment to predict sound classes.
5. Maps original sound classes to ground truth categories (Human or Non-Human).
6. Aggregates and saves classification results to a CSV file.

Usage:
    python -m scripts.4_audio_classification
    Set `is_save = True` to run MPSENET and save outputs.

External Dependencies:
- SSLAM model and inference code must be installed and accessible.

"""

import os
import sys
import pandas as pd
import subprocess
from scripts.config import original_data_path as timestamps_root
from scripts.utils.classification_util import analyze_results

main_dir = "SSLAM"
sslam_dirname = "SSLAM_Inference"
# Add the inference directory to Python path
inference_path = os.path.join(main_dir, sslam_dirname, "inference")
if inference_path not in sys.path:
    sys.path.append(inference_path)

def process_audio_segment(audio_path: str, start_time: float, end_time: float, channel: int =None) -> list:
    """Extracts an audio segment using ffmpeg and runs SSLAM inference."""
    # Create temp segment file
    segment_file = f"/tmp/segment_{os.path.basename(audio_path)}"
    try:
        # Base ffmpeg command
        cmd = f"ffmpeg -i {audio_path} -ss {start_time} -to {end_time}"
        # Add channel selection if specified
        if channel is not None:
            # For 7.1 audio files, we need to use the pan filter to extract specific channels
            # Channel mapping for 7.1: 0=FL, 1=FR, 2=FC, 3=LFE, 4=BL, 5=BR, 6=SL, 7=SR
            cmd += f" -af 'pan=mono|c0=c{channel}'"
        cmd += f" {segment_file} -y"
        subprocess.run(cmd, shell=True, check=True)
        if not os.path.exists(segment_file):
            return None
        # Process the segment
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from inference import main
        original_argv = sys.argv
        try:
            sys.argv = [
                'inference.py',
                '--source_file', segment_file,
                '--label_file', os.path.join(main_dir, sslam_dirname, "inference", "labels.csv"),
                '--model_dir', os.path.join(main_dir, sslam_dirname),
                '--checkpoint_dir', os.path.join(main_dir, "checkpoint_best.pt"),
                '--target_length', '1024',
                '--top_k_prediction', '12',
                '--norm_mean', '-4.268',
                '--norm_std', '4.569'
            ]
            prediction_results = main()
            return prediction_results
        finally:
            sys.argv = original_argv
    except subprocess.CalledProcessError as e:
        print(f"Error processing segment: {e}")
        return None
    finally:
        if os.path.exists(segment_file):
            os.remove(segment_file)

def get_ground_truth_class(soundclass: str) -> str:
    """Map soundclass to ground truth (H or NH)"""
    human_classes = ['male', 'female', 'crying', 'humansounds']
    return "H" if soundclass in human_classes else "NH"

def process_audio_files(ref_root:str, output_csv: str, is_save: bool) -> None:
    """Process all audio files and generate classification results."""
    if not is_save:
        print("is_save is set to False. Exiting without processing.")
        return    
    results = []
    # Find all relevant audio files in throttle-0 folders
    # throttle0_path = os.path.join(ref_root, "throttle-0")
    audio_files = []
    
    for root, _, files in os.walk(ref_root):
        for file in files:
            # if "throttle-0" not in root: continue
            if "noisy_wavfiles" in root or "generated_files" in root or "throttle-0" in root: continue
            if ("mic1_soundskrit-File" in file or 
                "mic2_8array-down-File" in file or 
                "mic3_8array-up-File" in file):
                audio_files.append(os.path.join(root, file))
    # print(audio_files)
    # Process each audio file
    for audio_path in audio_files:
        audio_name = os.path.basename(audio_path)
        base_file_name = audio_name.split('.')[0].split('-')[-1]
        
        # Determine channel to use
        channel = None
        # if "soundskrit" in audio_name.lower():
        #     channel = 0  # First channel
        # elif "down" in audio_name.lower() or "up" in audio_name.lower():
        #     channel = get_lowest_rms_channel(audio_path)
        
        # Find corresponding timestamp file
        # print(audio_name)
        # if 'agg' in audio_name:
        #     tfilename = audio_name.replace('nr-','').replace('-agg0.5','')
        timestamp_file = os.path.join(timestamps_root, f"{base_file_name}.txt")
        if not os.path.exists(timestamp_file):
            print(f"Timestamp file not found for {timestamp_file}")
            continue
            
        # Read timestamps and soundclasses
        try:
            with open(timestamp_file, 'r') as f:
                segments = [line.strip().split('\t') for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading {timestamp_file}: {e}")
            continue
            
        # Process each segment
        for segment in segments:
            if len(segment) != 3:
                continue
                
            start_time, end_time, soundclass = segment
            ground_truth = get_ground_truth_class(soundclass)
            
            try:
                start_time = float(start_time)
                end_time = float(end_time)
            except ValueError:
                continue
                
            # Process the segment
            prediction_results = process_audio_segment(
                audio_path=audio_path,
                start_time=start_time,
                end_time=end_time,
                channel=channel
            )
            predicted_unmapped, predicted_class = analyze_results(prediction_results)
            results.append({
                'AudioFile': audio_path,
                'SegmentStart': start_time,
                'SegmentEnd': end_time,
                'GroundTruth': ground_truth,
                'Predicted': predicted_class,
                'GTSoundClass': soundclass,
                'PSoundClass': predicted_unmapped
            })
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        # Print summary statistics
        # print("\nClassification Report:")
        # print("Ground Truth vs Predicted:")
        # print(pd.crosstab(df['GroundTruth'], df['Predicted']))
        
        # print("\nBy Sound Class:")
        # print(df.groupby(['SoundClass', 'GroundTruth', 'Predicted']).size().unstack())
    else:
        print("No valid segments processed")

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
    source_folder = os.path.join(root_path, 'outputs', 'mpsenet')
    target_path = os.path.join(root_path, 'outputs', 'classification', 'classification_results.csv')    
    # once downloaded from HF, set root_path as shown below
    # root_path = './data/DroneAudioSet/' # path to the downloaded DroneAudioSet data
    # source_folder = os.path.join(root_path, 'outputs', 'mpsenet',
    #                                       speaker_loudness, room, drone, speaker_dist,
    #                                       mic_dist, throttle)
    # target_path = os.path.join(root_path, 'outputs', 'classification',
    #                                       speaker_loudness, room, drone, speaker_dist,
    #                                       mic_dist, throttle, 'classification_results.csv')
    #  
    process_audio_files(ref_root=source_folder, output_csv=target_path, is_save=is_save)