import sys
from config import *
from datasets import *
from evaluation import accuracies

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

validation_set_names = ['mdbsynth']
dataset_names = ['mdbsynth']


def prepare_datasets(names) -> (Dataset, (np.ndarray, np.ndarray)):
    train = train_dataset(*names, batch_size=options['batch_size'], augment=options['augment'])
    print("Train dataset:", train, file=sys.stderr)

    validation = []
    for name in validation_set_names:
        print("Collecting validation set {}:".format(name), file=sys.stderr)
        dataset = validation_dataset(name, seed=42, take=100).take(options['validation_take']).collect(verbose=True)
        validation.append(dataset)

    return train, validation


class PitchAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, val_sets, local_average=False):
        super().__init__()
        self.val_sets = [(audio, to_weighted_average_cents(pitch)) for audio, pitch in val_sets]
        self.local_average = local_average
        self.to_cents = local_average and to_local_average_cents or to_weighted_average_cents
        self.prefix = local_average and 'local-average-' or 'default-'
        for filename in ["mae.tsv", "rpa.tsv", "rca.tsv"]:
            with open(log_path(self.prefix + filename), "w") as f:
                f.write('\t'.join(validation_set_names) + '\n')

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        names = list(validation_set_names)
        print(file=sys.stderr)

        mae_list = []
        rpa_list = []
        rca_list = []

        # print("Epoch {}, validation accuracies (local_average = {})".format(epoch + 1, self.local_average), file=sys.stderr)
        for audio, true_cents in self.val_sets:
            predicted = self.model.predict(audio)
            predicted_cents = self.to_cents(predicted)
            diff = np.abs(true_cents - predicted_cents)
            mae = np.mean(diff[np.isfinite(diff)])
            rpa, rca = accuracies(true_cents, predicted_cents)
            nans = np.mean(np.isnan(diff))
            # print("{}: MAE = {}, RPA = {}, RCA = {}, nans = {}".format(names.pop(0), mae, rpa, rca, nans), file=sys.stderr)
            print(f"{names.pop(0)}: MAE = {mae}, RPA = {rpa}, RCA = {rca}, nans = {nans}", file=sys.stderr)
            mae_list.append(mae)
            rpa_list.append(rpa)
            rca_list.append(rca)

        with open(log_path(self.prefix + "mae.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % mae for mae in mae_list]) + '\n')
        with open(log_path(self.prefix + "rpa.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rpa for rpa in rpa_list]) + '\n')
        with open(log_path(self.prefix + "rca.tsv"), "a") as f:
            f.write('\t'.join(['%.6f' % rca for rca in rca_list]) + '\n')

        print(file=sys.stderr)


def main():
    train_set, val_sets = prepare_datasets(dataset_names)
    val_data = Dataset.concat([Dataset(*val_set) for val_set in val_sets]).collect()

    model: keras.Model = build_model()
    # model.summary()

    callbacks = get_default_callbacks() + [PitchAccuracyCallback(val_sets, local_average=True)]
    # model.fit_generator(iter(train_set),
    #                     steps_per_epoch=options['steps_per_epoch'],
    #                     epochs=options['epochs'],
    #                     callbacks=callbacks,
    #                     validation_data=val_data)

    model.fit(train_set.tensorflow(),
              steps_per_epoch=options['steps_per_epoch'],
              epochs=options['epochs'],
              callbacks=callbacks,
              validation_data=val_data)


if __name__ == "__main__":
    main()
