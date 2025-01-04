import os
import sys
from datetime import datetime

import numpy as np
from tensorflow.keras.callbacks import Callback
import wandb

from evaluation import accuracies
from config import (
    options,
    log_path,
    build_model,
    get_default_callbacks
)
from data_handlers import (
    Dataset,
    train_dataset,
    validation_dataset,
    to_weighted_average_cents,
    to_local_average_cents
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
wandb.init(project='crepe-retrain', resume=True, name=f"run-{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}")

def prepare_datasets(train_dataset_names, val_dataset_names) -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(*train_dataset_names, batch_size=options['batch_size'], augment=options['augment'])
    print("Train dataset:", train, file=sys.stderr)

    validation = []
    for name in val_dataset_names:
        print(f"Collecting validation set {name}: ", file=sys.stderr)
        dataset = validation_dataset(name, seed=42, take=100).take(options['validation_take']).collect(verbose=True)
        validation.append(dataset)

    return train, validation

class PitchAccuracyCallback(Callback):
    def __init__(self, val_sets, val_dataset_names, local_average=False):
        super().__init__()
        self.val_dataset_names = val_dataset_names
        self.val_sets = [(audio, to_weighted_average_cents(pitch)) for audio, pitch in val_sets]
        self.local_average = local_average
        self.to_cents = local_average and to_local_average_cents or to_weighted_average_cents
        self.prefix = local_average and 'local-average-' or 'default-'
        for filename in ["mae.tsv", "rpa.tsv", "rca.tsv"]:
            with open(log_path(self.prefix + filename), "w") as f:
                f.write('\t'.join(val_dataset_names) + '\n')

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        names = list(self.val_dataset_names)
        print(file=sys.stderr)

        mae_list = []
        rpa_list = []
        rca_list = []

        for audio_frames, true_cents in self.val_sets:
            predicted = self.model.predict(audio_frames)
            predicted_cents = self.to_cents(predicted)
            diff = np.abs(true_cents - predicted_cents)
            mae = np.mean(diff[np.isfinite(diff)])
            rpa, rca = accuracies(true_cents, predicted_cents)
            nans = np.mean(np.isnan(diff))

            print(f"{names.pop(0)}: MAE = {mae}, RPA = {rpa}, RCA = {rca}, nans = {nans}", file=sys.stderr)
            mae_list.append(mae)
            rpa_list.append(rpa)
            rca_list.append(rca)

            wandb.log({"rpa": rpa, "rca": rca, "mae": mae})

        with open(log_path(self.prefix + "mae.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % mae for mae in mae_list]) + '\n')
        with open(log_path(self.prefix + "rpa.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rpa for rpa in rpa_list]) + '\n')
        with open(log_path(self.prefix + "rca.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rca for rca in rca_list]) + '\n')

        print(file=sys.stderr)

def main():
    model = build_model()
    validation_set_names = ['mdbsynth']
    dataset_names = ['mdbsynth']
    train_set, val_sets = prepare_datasets(dataset_names, validation_set_names)
    val_data = Dataset.concat([Dataset(*val_set) for val_set in val_sets]).collect()

    callbacks = get_default_callbacks(
        PitchAccuracyCallback(val_sets, validation_set_names, local_average=True)
    )

    model.fit(train_set.tensorflow(),
              # steps_per_epoch=options['steps_per_epoch'],
              steps_per_epoch=5,
              # epochs=options['epochs'],
              epochs=5,
              callbacks=callbacks,
              validation_data=val_data)

if __name__ == "__main__":
    main()
