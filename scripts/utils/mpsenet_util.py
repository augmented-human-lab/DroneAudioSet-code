import os
import librosa
import soundfile as sf
import numpy as np
from collections import defaultdict

def get_num_channels(file_path):
    """Get the number of channels in a wav file."""
    data, sr = librosa.load(file_path, sr=None, mono=False)
    return data.shape[0] if len(data.shape) > 1 else 1

def split_wav_by_channels_and_time(file_path, output_folder):
    """Splits wav file into individual channels and then into 10-sec segments."""
    os.makedirs(output_folder, exist_ok=True)
    data, sr = librosa.load(file_path, sr=None, mono=False)
    # num_channels = 1 if "soundskrit" in file_path else (8 if "up" in file_path or "down" in file_path else 1)
    num_channels = 1
    if num_channels > data.shape[0]:
        num_channels = data.shape[0]  # Handle cases where actual channels are fewer
    split_files = []
    for ch in range(num_channels):
        if num_channels==1:
            channel_data = data#[0]
        else:
            channel_data = data[ch]
        duration = librosa.get_duration(y=channel_data, sr=sr)
        num_splits = int(np.ceil(duration / 10))
        for split_idx in range(num_splits):
            output_file = os.path.join(output_folder, f"{os.path.basename(file_path).replace('.wav', '')}_ch{ch+1}_split{split_idx+1}.wav")
            if os.path.exists(output_file):
                split_files.append(output_file)
                continue
            start_sample = split_idx * 10 * sr
            end_sample = min((split_idx + 1) * 10 * sr, len(channel_data))
            split_audio = channel_data[start_sample:end_sample]
            sf.write(output_file, split_audio, sr)
            split_files.append(output_file)
    return split_files

def stitch_wav_files(original_file, split_files_folder, output_folder):
    """
    Stitch processed split WAV files back into complete multichannel WAV files.
    
    Args:
        split_files_folder: Folder containing split WAV files with format:
            mic<micnumber>_<arraytype>-File<filenumber>_ch<channelnumber>_split<splitnumber>.wav
        output_folder: Folder to save stitched multichannel WAV files
    """
    os.makedirs(output_folder, exist_ok=True)
    # Group files by their base identifiers (mic, arraytype, filenumber)
    file_groups = defaultdict(lambda: defaultdict(list))
    # First, organize all split files by their base components
    for filename in os.listdir(split_files_folder):
        if not filename.endswith('.wav'):
            continue
        try:
            # Parse filename components
            parts = filename.split('_')
            mic_part = parts[0]  # <suffix>-mic<micnumber> eg. nr-mic1
            array_file_part = parts[1]  # <arraytype>-File<filenumber>
            channel_part = parts[2]  # ch<channelnumber>
            split_part = parts[3]  # split<splitnumber>.wav
            mic_number = mic_part[-4:]  # Extract micnumber
            array_type = array_file_part.split('-File')[0]  # Extract arraytype
            file_number = array_file_part.split('-File')[1]  # Extract filenumber
            channel_number = int(channel_part[2:])  # Extract channelnumber
            split_number = int(split_part[5:-4])  # Extract splitnumber
            # Create unique key for each original file
            file_key = (mic_number, array_type, file_number)
            # Add to our grouped dictionary
            file_groups[file_key][channel_number].append((split_number, filename))
        except (IndexError, ValueError) as e:
            print(f"Skipping malformed filename: {filename} - {str(e)}")
            continue
    
    # Process each file group
    for file_key, channel_data in file_groups.items():
        mic_number, array_type, file_number = file_key
        output_filename = f"mpsenetmvdr-{mic_number}_{array_type}-File{file_number}.wav"
        output_path = os.path.join(output_folder, output_filename)
        if os.path.exists(output_path):
            # print(f"Skipping existing file: {output_filename}")
            continue
        print(f"Processing {output_filename}...")
        # Determine number of channels from the data we found
        num_channels = len(channel_data.keys())
        if num_channels == 0:
            continue
        # For each channel, stitch its splits together in order
        stitched_channels = []
        sr = None
        max_length = 0
        for ch in sorted(channel_data.keys()):
            # Sort splits by their split number
            sorted_splits = sorted(channel_data[ch], key=lambda x: x[0])
            channel_pieces = []
            for split_num, split_file in sorted_splits:
                split_path = os.path.join(split_files_folder, split_file)
                try:
                    data, current_sr = librosa.load(split_path, sr=None, mono=True)
                    if sr is None:
                        sr = current_sr
                    elif current_sr != sr:
                        print(f"Warning: Sample rate mismatch in {split_file}")
                    channel_pieces.append(data)
                except Exception as e:
                    print(f"Error loading {split_file}: {str(e)}")
                    continue
            if channel_pieces:
                stitched_channel = np.concatenate(channel_pieces)
                stitched_channels.append(stitched_channel)
                max_length = max(max_length, len(stitched_channel))
        if not stitched_channels:
            print(f"No valid data for {output_filename}")
            continue
        # Pad all channels to equal length
        padded_channels = []
        for channel in stitched_channels:
            padding = max_length - len(channel)
            padded_channels.append(np.pad(channel, (0, padding), mode='constant'))
        # Combine channels into multichannel array
        multichannel_audio = np.vstack(padded_channels).T
        # Save the stitched file
        try:
            sf.write(output_path, multichannel_audio, sr)
            print(f"Successfully saved {output_filename}")
        except Exception as e:
            print(f"Error saving {output_filename}: {str(e)}")

