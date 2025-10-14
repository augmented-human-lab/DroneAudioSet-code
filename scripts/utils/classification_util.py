import os
import csv
import numpy as np
import soundfile as sf
from scripts.config import label_mapping_path

def get_lowest_rms_channel(audio_path):
    """Find the channel with lowest RMS in the audio file"""
    # audio_path = os.path.join(audio_folder, f"{filename_prefix}.wav")
    if not os.path.exists(audio_path):
        return None
    try:
        audio, _ = sf.read(audio_path)
        if len(audio.shape) == 1:  # Mono file
            return 1
        # Calculate RMS for each channel
        rms_values = [np.sqrt(np.mean(channel**2)) for channel in audio.T]
        return np.argmin(rms_values) + 1  # Channels are 1-indexed
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None
    
# Load the label mapping from CSV
def load_label_mapping(mapping_file):
    label_map = {}
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            index = int(row[0])
            label = row[2]
            category = row[3]
            label_map[label] = category
    return label_map

# Load label mapping
label_mapping = load_label_mapping(label_mapping_path)

def analyze_results(prediction_results):
    """Analyze prediction results and determine H/NH classification."""
    if not prediction_results:
        return "Unknown"
    sorted_predictions = sorted(prediction_results.items(), key=lambda x: x[1], reverse=True)
    top_prediction = sorted_predictions[0][0]  # Get the top prediction label
    return top_prediction, label_mapping.get(top_prediction, "Unknown")