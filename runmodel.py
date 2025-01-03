import os
import sys
import argparse

import gzip
import imageio
import numpy as np
from tensorflow.keras import models # noqa
import matplotlib.cm
from numpy.lib.stride_tricks import as_strided
from resampy import resample
from scipy.io import wavfile
from mir_eval.melody import (
    raw_pitch_accuracy,
    raw_chroma_accuracy,
    hz2cents,
    voicing_measures,
    overall_accuracy
)

from data_handlers import to_local_average_cents, to_viterbi_cents


parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help='path to the HDF5 file that contains the Keras model')
parser.add_argument('input_path',
                    help='path that contains .wav or .npy.gz files to run the model on')
parser.add_argument('output_path', nargs='?', default=None,
                    help='path to save the prediction and salience results (default: same as input_path)')
parser.add_argument('--save-numpy', action='store_true',
                    help='save the salience representation to .npy file as well')
parser.add_argument('--truth-path', default=None,
                    help='path to the corresponding .csv or .npy.gz files that contains the ground-truth annotations')
parser.add_argument('--viterbi', action='store_true',
                    help='run Viterbi decoding for finding the center frequencies')
args = parser.parse_args()

if args.output_path is None:
    args.output_path = args.input_path

def wav_stream(wav_files):
    for file in wav_files:
        srate, wav_data = wavfile.read(os.path.join(args.input_path, file))
        if len(wav_data.shape) == 2:
            wav_data = wav_data.mean(axis=1)
        if srate != 16000:
            wav_data = resample(wav_data, srate, 16000)
            srate = 16000
        hop_length = int(srate / 100)
        n_frames = 1 + int((len(wav_data) - 1024) / hop_length)
        frames = as_strided(wav_data, shape=(1024, n_frames),
                            strides=(wav_data.itemsize, hop_length * wav_data.itemsize))
        frames = frames.transpose().astype(np.float32)
        yield file, frames


def npygz_stream(npygz_files):
    for file in npygz_files:
        with gzip.open(os.path.join(args.input_path, file)) as f:
            yield file, np.load(f).transpose()

def report_accuracy(filename, label_truth, estimated, conf):
    for acc_file, tau in zip(accuracy_files, np.arange(0.40, 0.95, 0.05)):
        ref_voicing = np.array(label_truth != 0)
        est_voicing = np.array(conf > tau)

        rpa = raw_pitch_accuracy(ref_voicing, label_truth, est_voicing, estimated)
        rca = raw_chroma_accuracy(ref_voicing, label_truth, est_voicing, estimated)
        recall, false_alarm = voicing_measures(ref_voicing, est_voicing)
        oa = overall_accuracy(ref_voicing, label_truth, est_voicing, estimated)

        with open(acc_file, "a") as f:
            print(f"{filename},{rpa},{rca},{recall},{false_alarm},{oa}", file=f)

files = [file for file in os.listdir(args.input_path) if file.lower().endswith('.wav')]
files.sort()
num_files = len(files)
if num_files > 0:
    print(num_files, "wav files found")
    stream = wav_stream(files)
else:
    files = [file for file in os.listdir(args.input_path) if file.lower().endswith('.npy.gz')]
    files.sort()
    num_files = len(files)
    if num_files > 0:
        print(num_files, ".npy.gz files found")
        stream = npygz_stream(files)
    else:
        raise ValueError("No .wav or .npy.gz files found in ", args.input_path)

model = models.load_model(args.model)
model.summary()

inferno = matplotlib.cm.get_cmap('inferno')
viridis = matplotlib.cm.get_cmap('viridis')
jet = matplotlib.cm.get_cmap('jet')

accuracy_files = [os.path.join(args.output_path, 'accuracies-%.2f.csv' % f) for f in np.arange(0.40, 0.95, 0.05)]
for accuracy_file in accuracy_files:
    with open(accuracy_file, "w") as f:
        print("NAME,RPA,RCA,VR,VFA,OA", file=f)

for name, data in stream:
    print('processing', name, 'of shape', data.shape)
    data -= np.mean(data, axis=1)[:, np.newaxis]
    data /= np.std(data, axis=1)[:, np.newaxis]
    predictions = model.predict(data, verbose=True)
    if args.viterbi:
        cents = to_viterbi_cents(predictions)
    else:
        cents = to_local_average_cents(predictions)
    confidence = np.max(predictions, axis=1)
    hertz = 10.0 * 2 ** (cents / 1200.0)
    timestamps = 0.01 * np.array(range(hertz.shape[0]))
    result = np.vstack([timestamps, hertz, confidence]).transpose()
    result_file = os.path.join(args.output_path, name + '.f0.csv')
    np.savetxt(result_file, result, fmt='%.6f', delimiter=',', header='time,frequency,confidence')

    if args.save_numpy:
        salience_file = os.path.join(args.output_path, name + '.salience.npy')
        np.save(salience_file, predictions)

    predictions = np.flip(predictions, axis=1)  # to draw the low pitches in the bottom

    figure_file = str(os.path.join(args.output_path, name + '.salience.png'))
    image = inferno(predictions.transpose())
    image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
    image[-20:-10, :, :] = viridis(confidence)[np.newaxis, :, :]
    image[-10:, :, :] = viridis((confidence > 0.5).astype(float))[np.newaxis, :, :]
    imageio.imwrite(figure_file, 255 * image)

    if args.truth_path:
        basename = name.replace('.npy.gz', '')
        csv_path = os.path.join(args.truth_path, basename + '.csv')
        npygz_path = os.path.join(args.truth_path, basename + '.npy.gz')
        if os.path.isfile(csv_path):
            truth = np.loadtxt(csv_path)
        elif os.path.isfile(npygz_path):
            with gzip.open(npygz_path) as f:
                truth = np.load(f)
        else:
            print('truth file for {} not found'.format(name), file=sys.stderr)
        truth = hz2cents(truth)
        report_accuracy(name, truth, cents, confidence)

        image = imageio.v2.imread(figure_file, pilmode='RGB')
        image = np.pad(image, [(20, 0), (0, 0), (0, 0)], mode='constant')

        for i in range(image.shape[1]):
            if truth[i] < 1:
                continue  # no-voice
            image[:20, i, :] = 255 * np.array(jet(int(abs(truth[i] - cents[i]))))[:3]

        imageio.imwrite(figure_file, image)