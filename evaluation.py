import numpy as np
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy


def accuracies(true_cents, predicted_cents, cent_tolerence=50):
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    rpa = raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerence)
    rca = raw_chroma_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerence)
    return rpa, rca

def raw_multipitch_accuracy(ref_cent, est_cent, cent_tolerance=50):
    """
    Computes Raw Pitch Accuracy (RPA) for multi-pitch estimation.

    :param ref_cent: numpy array of reference frequencies arrays => np.array([[2000.03,550.56],[1000.20, 3654.12], ...])
    :param est_cent: numpy array of estimated frequencies arrays => np.array([[2000.03,550.56],[1000.20, 3654.12], ...])
    :param cent_tolerance: acceptable tolerance between reference and estimated frequencies
    :return: rpa as a float
    """
    nonzero_freqs = np.logical_and(est_cent != 0.0, ref_cent != 0.0)

    if np.sum(nonzero_freqs) == 0:
        return 0.0

    # est_cent array is sorted by model, but ref_cents must be sorted to match est_cents array
    freq_diff_cents = np.abs(np.sort(ref_cent) - est_cent)[nonzero_freqs]
    correct_frequencies = freq_diff_cents < cent_tolerance
    voicing = ref_cent != 0.0
    rpa = np.sum(voicing[nonzero_freqs] * correct_frequencies) / np.sum(voicing)

    return rpa