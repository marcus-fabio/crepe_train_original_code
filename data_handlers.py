import os

import numpy as np
from hmmlearn import hmm
from random import Random
from mir_eval.melody import hz2cents
from scipy.stats import norm

from flazy import Dataset
from transforms import normalize, add_noise, pitch_shift

classifier_lowest_hz = 31.70
classifier_lowest_cent = hz2cents(np.array([classifier_lowest_hz]))[0]
classifier_cents_per_bin = 20
classifier_octaves = 6
classifier_total_bins = int((1200 / classifier_cents_per_bin) * classifier_octaves)
classifier_cents = np.linspace(0, (classifier_total_bins - 1) * classifier_cents_per_bin,
                               classifier_total_bins) + classifier_lowest_cent
classifier_cents_2d = np.expand_dims(classifier_cents, axis=1)
classifier_norm_stdev = 25
classifier_pdf_normalizer = norm.pdf(0)


def to_classifier_label(pitch):
    """
    Converts pitch labels in cents, to a vector representing the classification label
    Uses the normal distribution centered at the pitch and the standard deviation of 25 cents,
    normalized so that the exact prediction has the value 1.0.
    :param pitch: a number or numpy array of shape (1, )
    pitch values in cents, as returned by hz2cents with base_frequency = 10 (default)
    :return: ndarray
    """
    result = norm.pdf((classifier_cents - pitch) / classifier_norm_stdev).astype(np.float32)
    result /= classifier_pdf_normalizer
    return result


def to_weighted_average_cents(label):
    if label.ndim == 1:
        product_sum = np.sum(label * classifier_cents)
        weight_sum = np.sum(label)
        return product_sum / weight_sum
    if label.ndim == 2:
        product_sum = np.dot(label, classifier_cents)
        weight_sum = np.sum(label, axis=1)
        return product_sum / weight_sum
    raise Exception("label should be either 1d or 2d ndarray")


def to_local_average_cents(salience, center=None):
    """find the weighted average cents near the argmax bin"""

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience):
    """Find the Viterbi path using a transition prior that induces pitch continuity"""

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the others
    self_emission = 0.1
    emission = np.eye(360) * self_emission + np.ones(shape=(360, 360)) * ((1 - self_emission) / 359)

    # fix the model parameters because we are not optimizing the model
    # noinspection PyTypeChecker
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_ = starting
    model.transmat_ = transition
    model.emissionprob_ = emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])


def f0_to_target_vector(f0, vec_size=486, f_min=30., f_max=1000., return_freqs=False):
    """
    convert from target f0 value to target vector of vecSize pitch classes (corresponding to the values in cents_mapping) that is used as output by the CREPE model
    Unlike the original CREPE model, the first class corresponds to a frequency of 0 (for unvoiced segments).
    If the frequency is 0, all values are 0, except for the 1st value that is = 1.
    For all other cases, the values are gaussian blurred around the target_pitch class, with a maximum value of 1
    :param f0: target f0 value
    :return: target vector of vecSize pitch classes (regularly spaced in cents, from fmin to fmax)
    @param return_freqs:
    @param f_max:
    @param f_min:
    @param f0:
    @param vec_size:
    """

    fmin_cents = freq2cents(f_min)
    fmax_cents = freq2cents(f_max)
    mapping_cents = np.linspace(fmin_cents, fmax_cents, vec_size)

    # get the idx corresponding to the closest pitch
    f0_cents = freq2cents(f0[0])  # f0[0] manual adjustment to get value from ndarray

    # if isinstance(f0, np.ndarray):
    #     f0_cents = f0_cents[:, np.newaxis]

    # gaussian-blur the vector auround the taget pitch idx as stated in the paper :
    # my equation, same result but comment out to use original equation (optimized by norm.pdf function)
    # sigma = 25
    # target_vec = np.exp(-((mapping_cents - f0_cents) ** 2) / (2 * (sigma ** 2)))

    target_vec = norm.pdf((mapping_cents - f0_cents) / classifier_norm_stdev).astype(np.float32)
    target_vec /= classifier_pdf_normalizer

    if return_freqs:
        return target_vec, mapping_cents
    else:
        return target_vec


def to_local_average_cents_fcn(salience, center=None, f_min=30., f_max=1000., vec_size=486):
    """
    find the weighted average cents near the argmax bin in output pitch class vector

    :param salience: output vector of salience for each pitch class
    :param fmin: minimum output frequency (corresponding to the 1st pitch class in output vector)
    :param fmax: maximum output frequency (corresponding to the last pitch class in output vector)
    :param vecSize: number of pitch classes in output vector
    :return: predicted pitch in cents
    @param vec_size:
    @param f_max:
    @param f_min:
    @param salience:
    @param center:
    """

    if not hasattr(to_local_average_cents_fcn, 'mapping'):
        # the bin number-to-cents mapping
        fmin_cents = freq2cents(f_min)
        fmax_cents = freq2cents(f_max)
        to_local_average_cents_fcn.mapping = np.linspace(fmin_cents, fmax_cents, vec_size) # cents values corresponding to the bins of the output vector

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience)) # index of maximum value in output vector
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents_fcn.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents_fcn(salience[i, :]) for i in range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def freq2cents(f0, f_ref=10.):
    """
    Convert a given frequency into its corresponding cents value, according to given reference frequency f_ref
    :param f0: f0 value (in Hz)
    :param f_ref: reference frequency for conversion to cents (in Hz)
    :return: value in cents
    """
    c = 1200 * np.log2(f0/f_ref)
    return c


def train_dataset(names, train_path, batch_size=32, loop=True, augment=True) -> Dataset:
    if len(names) == 0:
        raise ValueError("dataset names required")

    paths = [os.path.join(train_path, name) for name in names]

    datasets = [Dataset.read.tfrecord(path, compression='gzip') for path in paths]
    datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

    if loop:
        datasets = [dataset.repeat() for dataset in datasets]

    result = Dataset.roundrobin(datasets)
    result = result.starmap(normalize)

    if augment:
        result = result.starmap(add_noise)
        result = result.starmap(pitch_shift)

    result = result.map(lambda x: (x[0], f0_to_target_vector(x[1])))

    if batch_size:
        result = result.batch(batch_size)

    return result


def validation_dataset(names, test_path: str, seed=None, take=None) -> Dataset:
    if len(names) == 0:
        raise ValueError("dataset names required")

    paths = [os.path.join(test_path, name) for name in names]

    all_datasets = []

    for path in paths:
        files = [os.path.join(path, file) for file in os.listdir(path)]

        if seed:
            files = Random(seed).sample(files, len(files))

        datasets = [Dataset.read.tfrecord(file, compression='gzip') for file in files]
        datasets = [dataset.select_tuple('audio', 'pitch') for dataset in datasets]

        if seed:
            datasets = [dataset.shuffle(seed=seed) for dataset in datasets]
        if take:
            datasets = [dataset.take(take) for dataset in datasets]

        all_datasets.append(Dataset.concat(datasets))

    result = Dataset.roundrobin(all_datasets)
    result = result.starmap(normalize)
    result = result.map(lambda x: (x[0], f0_to_target_vector(x[1])))

    return result
