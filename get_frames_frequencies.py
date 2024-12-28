import os
import numpy as np
import pandas as pd
import librosa
import gzip
from tqdm import tqdm

audios_folder = 'data/test/mdbsynth/audio_stems'
annotations_folder = 'data/test/mdbsynth/annotation_stems'
mdbsynth_folder = "D:/mdbsynth"

dataset_sampling_rate = 44100.
dataset_frame_size = 1024
dataset_hop_size = 128

def save_frames_annotations():
    audios_list = sorted(os.listdir(audios_folder))
    # num_files = len(audios_list)

    # for idx, audio_name in enumerate(audios_list):
    for audio_name in tqdm(audios_list, desc="Progress", unit="file"):
        audio_path = os.path.join(audios_folder, audio_name)
        annotation_path = os.path.join(annotations_folder, audio_name.replace(".wav", ".csv"))

        # print(f"\nLoad audio: {audio_name} ({idx + 1}/{num_files})")
        audio_samples, _ = librosa.load(audio_path, None)

        # print(f"Pad audio to center frames")
        audio_samples = librosa.util.pad_center(audio_samples, audio_samples.size + dataset_frame_size)

        # print("Get audio frames")
        frames = librosa.util.frame(audio_samples, dataset_frame_size, dataset_hop_size, axis=0)

        # print("Load frequency annotations")
        annotations = pd.read_csv(annotation_path, header=None, names=['timestamp', 'frequency'])
        frequencies = annotations['frequency'].values

        # print("Check frames and frequencies vector sizes")
        if frames.shape[0] != frequencies.shape[0]:
            times = annotations['timestamp'].values
            frame_indexes = librosa.time_to_frames(times, dataset_sampling_rate, dataset_hop_size)
            frames = frames[frame_indexes, :]

        # print("Save frames as .npy.gz")
        frames_output_path = os.path.join(mdbsynth_folder, 'raw', f"{audio_name.replace('.wav', '.npy.gz')}")
        os.makedirs(os.path.dirname(frames_output_path), exist_ok=True)
        with gzip.GzipFile(frames_output_path, 'wb') as f:
            np.save(f, frames.T)

        # print("Save frequencies as .npy.gz\n")
        frequencies_output_path = os.path.join(mdbsynth_folder, 'frequencies', f"{audio_name.replace('.wav', '.npy.gz')}")
        os.makedirs(os.path.dirname(frequencies_output_path), exist_ok=True)
        with gzip.GzipFile(frequencies_output_path, 'wb') as f:
            np.save(f, frequencies)

    print("Dataset processing complete.")

if __name__ == '__main__':
    save_frames_annotations()