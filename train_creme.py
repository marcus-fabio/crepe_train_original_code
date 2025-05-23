import os
import sys
from datetime import datetime
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import numpy as np
from tensorflow.keras.callbacks import Callback
import wandb

from evaluation import (
    raw_pitches_accuracy
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


def main():
    model = build_creme_model()
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
    main()
