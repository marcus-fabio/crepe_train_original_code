import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import numpy as np
from tensorflow.keras.callbacks import Callback
import wandb
import pandas as pd
import librosa

from evaluation import (
    raw_pitches_accuracy,
    evaluate_rpa_and_errors,
    # accuracies,
)
from data_handlers import (
    Dataset,
    train_dataset,
    validation_dataset,
    to_local_average_cents_multi,
    freq2cents,
    # to_weighted_average_cents,
    # to_local_average_cents_fcn,
    # to_classifier_label_multi,
)
from config import (
    options,
    log_path,
    build_creme_model,
    get_callbacks
    # build_model,
)

load_dotenv()

os.environ['WANDB_SILENT'] = 'true'

if os.getenv("WANDB_ENABLED") == "true":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=os.getenv("WANDB_PROJECT_NAME"),
               resume=False,
               name=f"run-{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}")

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

def normalize(frames: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (frames - np.mean(frames, axis=1, keepdims=True)) / np.std(frames, axis=1, keepdims=True)

def run_prediction(audio_name: str, model: Model):
    audio_path = os.path.join(audios_folder, audio_name)
    annotation_path = os.path.join(annotations_folder, audio_name.replace(".wav", ".csv"))

    # Get frames
    audio_samples, _ = librosa.load(audio_path, sr=creme_model_sampling_rate)
    audio_samples = librosa.util.pad_center(audio_samples, size=audio_samples.size + creme_model_input_size)
    frames = librosa.util.frame(audio_samples, frame_length=creme_model_input_size,
                                hop_length=dataset_hop_size_resampled, axis=0)

    # Get frequency references
    annotations = pd.read_csv(annotation_path, header=None, names=['timestamp', 'frequency1', 'frequency2'])
    frequencies = annotations[['frequency1', 'frequency2']].values

    # Fit frames and annotation size
    if frames.shape[0] != frequencies.shape[0]:
        times = annotations['timestamp'].values
        frame_indexes = librosa.time_to_frames(times, sr=creme_model_sampling_rate,
                                               hop_length=dataset_hop_size_resampled)
        frames = frames[frame_indexes, :]

    # Remove unvoiced frames
    nonzero = np.any(frequencies > 0, axis=1)
    frames = frames[nonzero]
    frequencies = frequencies[nonzero]

    # Get prediction
    predictions = model.predict(normalize(frames))

    return predictions, frequencies


def prepare_datasets(train_dataset_names,
                     val_dataset_names) -> (Dataset, (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    train = train_dataset(train_dataset_names,
                          options['train_path'],
                          batch_size=options['batch_size'],
                          augment=options['augment'])
    print("Train dataset configured")

    validation = []
    validation_raw = []
    for name in val_dataset_names:
        print(f"Collecting validation set {name}: ", file=sys.stderr)
        dataset = (validation_dataset([name], options['test_path'], seed=42, take=10000)
                   .take(options['validation_take'])
                   .collect(verbose=True))
        dataset_raw = (validation_dataset([name], options['test_path'], seed=42, take=10000, target_vector=False)
                       .take(options['validation_take'])
                       .collect(verbose=True))
        validation.append(dataset)
        validation_raw.append(dataset_raw)
    print("Validation dataset configured")

    return train, validation, validation_raw


class PitchAccuracyCallback(Callback):
    def __init__(self, val_sets, val_dataset_names, local_average=False):
        super().__init__()
        self.val_dataset_names = val_dataset_names
        # self.val_sets = [(audio, to_local_average_cents_multi(pitch)) for audio, pitch in val_sets]
        self.val_sets = [(audio, freq2cents(pitch)) for audio, pitch in val_sets]
        self.local_average = local_average
        self.to_cents = to_local_average_cents_multi
        self.prefix = local_average and 'local-average-' or 'default-'
        # for filename in ["mae.tsv", "rpa.tsv", "rca.tsv"]:
        for filename in ["mae.tsv", "rpa.tsv"]:
            with open(log_path(self.prefix + filename), "w") as f:
                f.write('\t'.join(val_dataset_names) + '\n')

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        names = list(self.val_dataset_names)
        print(file=sys.stderr)

        # mae_list = []
        rpa_list = []
        # rca_list = []

        for audio_frames, true_cents in self.val_sets:
            predicted = self.model.predict(audio_frames)
            predicted_cents = self.to_cents(predicted)
            # diff = np.abs(true_cents - predicted_cents)
            # mae = np.mean(diff[np.isfinite(diff)])
            # rpa, rca = accuracies(true_cents, predicted_cents)
            rpa = raw_pitches_accuracy(true_cents, predicted_cents)
            # nans = np.mean(np.isnan(diff))

            # print(f"{names.pop(0)}: MAE = {mae}, RPA = {rpa}, RCA = {rca}, nans = {nans}", file=sys.stderr)
            # print(f"{names.pop(0)}: MAE = {mae}, RPA = {rpa}, nans = {nans}", file=sys.stderr)
            print(f"{names.pop(0)}: RPA = {rpa}", file=sys.stderr)
            # mae_list.append(mae)
            rpa_list.append(rpa)
            # rca_list.append(rca)

            if os.getenv("WANDB_ENABLED") == "true":
                # wandb.log({"epoch": epoch, "rpa": rpa, "rca": rca, "mae": mae})
                # wandb.log({"epoch": epoch, "rpa": rpa, "mae": mae})
                wandb.log({"epoch": epoch, "rpa": rpa})

        # with open(log_path(self.prefix + "mae.tsv"), "a") as f:
        #     f.write('\t'.join(['%.6f' % mae for mae in mae_list]) + '\n')
        with open(log_path(self.prefix + "rpa.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rpa for rpa in rpa_list]) + '\n')
        # with open(log_path(self.prefix + "rca.tsv"), "a") as f:
        #     f.write('\t'.join(['%.6f' % rca for rca in rca_list]) + '\n')

        print(file=sys.stderr)


def main(model: Model):
    # model = build_creme_model()
    validation_set_names = ['validation-set']
    dataset_names = ['set-1', 'set-2', 'set-3', 'set-4', 'set-5', 'set-6', 'set-7']
    train_set, val_sets, val_sets_raw = prepare_datasets(dataset_names, validation_set_names)
    val_data = Dataset.concat([Dataset(*val_set) for val_set in val_sets]).collect()

    callbacks = get_callbacks(PitchAccuracyCallback(val_sets_raw, validation_set_names, local_average=True))

    model.fit(train_set.tensorflow(),
              steps_per_epoch=options['steps_per_epoch'],
              epochs=options['epochs'],
              callbacks=callbacks,
              validation_data=val_data)


if __name__ == "__main__":
    creme_model = build_creme_model()

    if options['prediction']:
        if os.getenv("WANDB_ENABLED") == "true":
            audios_list = os.listdir(audios_folder)
            table = wandb.Table(columns=["audio_name", "rpa", "substitution_error", "miss_error", "false_alarm", "total_error"])

            for audio_name in audios_list:
                predictions, reference_frequencies = run_prediction(audio_name, creme_model)
                predicted_cents = to_local_average_cents_multi(predictions)
                true_cents = freq2cents(reference_frequencies)
                rpa, substitution_error, miss_error, false_alarm, total_error = evaluate_rpa_and_errors(
                    true_cents, np.array(predicted_cents))

                table.add_data(audio_name, rpa, substitution_error, miss_error, false_alarm, total_error)
                wandb.log({
                    "rpa": rpa,
                    "substitution_error": substitution_error,
                    "miss_error": miss_error,
                    "false_alarm": false_alarm,
                    "total_error": total_error
                })

            wandb.log({"RPA": table})
    else:
        main(creme_model)
