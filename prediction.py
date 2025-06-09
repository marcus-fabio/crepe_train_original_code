import os
import numpy as np
import pandas as pd
import librosa

from keras import Model
from tqdm import tqdm

audios_folder = os.getenv("AUDIO_PATH") or '/mnt/e/mdb-stem-synth-multi/audio_stems'
# audios_folder = '/mnt/e/mdb-stem-synth-multi/audio'
annotations_folder = os.getenv("ANNOTATION_PATH") or '/mnt/e/mdb-stem-synth-multi/annotation_stems'
# annotations_folder = '/mnt/e/mdb-stem-synth-multi/annotation'

dataset_sampling_rate = 44100.
dataset_frame_size = 1024
dataset_hop_size = 128

creme_model_input_size = 1024
creme_model_sampling_rate = 16000.
dataset_hop_size_seconds = dataset_hop_size / dataset_sampling_rate
dataset_hop_size_resampled = int(creme_model_sampling_rate * dataset_hop_size_seconds)

predictions = []
reference_frequencies = []

def normalize(frames: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (frames - np.mean(frames, axis=1, keepdims=True)) / np.std(frames, axis=1, keepdims=True)

def run(model: Model):
    audios_list = os.listdir(audios_folder)

    for audio_name in tqdm(audios_list, desc="Progress", unit="file"):
        audio_path = os.path.join(audios_folder, audio_name)
        annotation_path = os.path.join(annotations_folder, audio_name.replace(".wav", ".csv"))

        # Get frames
        audio_samples, _ = librosa.load(audio_path, sr=creme_model_sampling_rate)
        audio_samples = librosa.util.pad_center(audio_samples, size=audio_samples.size + creme_model_input_size)
        frames = librosa.util.frame(audio_samples, frame_length=creme_model_input_size, hop_length=dataset_hop_size_resampled, axis=0)

        # Get frequency references
        annotations = pd.read_csv(annotation_path, header=None, names=['timestamp', 'frequency1', 'frequency2'])
        frequencies = annotations[['frequency1', 'frequency2']].values

        # Fit frames and annotation size
        if frames.shape[0] != frequencies.shape[0]:
            times = annotations['timestamp'].values
            frame_indexes = librosa.time_to_frames(times, sr=creme_model_sampling_rate, hop_length=dataset_hop_size_resampled)
            frames = frames[frame_indexes, :]

        # Remove unvoiced frames
        nonzero = np.any(frequencies > 0, axis=1)
        frames = frames[nonzero]
        frequencies = frequencies[nonzero]

        # Get prediction
        prediction = model.predict(normalize(frames))
        predictions.append(prediction)
        reference_frequencies.append(frequencies)

    print("Dataset processing complete.")
    return predictions, reference_frequencies, audios_list
