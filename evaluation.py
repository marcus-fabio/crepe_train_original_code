import numpy as np
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy
from mir_eval.multipitch import (
    compute_num_true_positives,
    compute_err_score,
    compute_num_freqs,
    compute_accuracy,
)


def accuracies(true_cents, predicted_cents, cent_tolerance=50):
    assert true_cents.shape == predicted_cents.shape

    voicing = np.ones(true_cents.shape)
    rpa = raw_pitch_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    rca = raw_chroma_accuracy(voicing, true_cents, voicing, predicted_cents, cent_tolerance)
    return rpa, rca


def raw_pitches_accuracy(ref_cents, est_cents, cent_tolerance=50) -> float:
    """
    Computes Raw Pitches Accuracy (RPA) for multi pitches estimation.

    :param ref_cents: reference frequencies in cents => np.array([[2000.03, 550.56,...],[1000.20, 3654.12,...], ...])
    :param est_cents: estimated frequencies in cents => np.array([[2000.03, 550.56,...],[1000.20, 3654.12,...], ...])
    :param cent_tolerance: acceptable tolerance between reference and estimated frequencies
    :return: rpa
    """

    # remove frames containing only zero frequencies in reference or estimation
    nonzero_freqs = np.logical_and(np.any(ref_cents, axis=1) != 0.0, np.any(est_cents, axis=1) != 0.0)

    if np.sum(nonzero_freqs) == 0:
        return 0.0

    # keep only non zeros frequencies in frames
    ref_cent_nz = [cent[cent != 0.0] for cent in ref_cents[nonzero_freqs]]
    est_cent_nz = [cent[cent != 0.0] for cent in est_cents[nonzero_freqs]]

    # calculate number of true positives, references and estimation for each frame
    true_positives = compute_num_true_positives(ref_cent_nz, est_cent_nz, cent_tolerance)
    n_ref = compute_num_freqs(ref_cent_nz)
    n_est = compute_num_freqs(est_cent_nz)

    # calculate rpa using recall (recall and rpa have same value when keep only non zeros frequencies )
    _, rpa, _ = compute_accuracy(true_positives, n_ref, n_est)
    # errors = compute_err_score(true_positives, n_ref, n_est)

    # rpa calculated without mir_eval functions (it works, same result)
    # total_true_positives = np.sum(true_positives)
    # total_ref = np.sum(ref_cent_nz != 0.0)
    # rpa = total_true_positives / total_ref

    return rpa

def evaluate_rpa_and_errors(ref_cents, est_cents, cent_tolerance=50) -> tuple:
    """
    Computes Raw Pitches Accuracy (RPA) for multi pitches estimation.

    :param ref_cents: reference frequencies in cents => np.array([[2000.03, 550.56,...],[1000.20, 3654.12,...], ...])
    :param est_cents: estimated frequencies in cents => np.array([[2000.03, 550.56,...],[1000.20, 3654.12,...], ...])
    :param cent_tolerance: acceptable tolerance between reference and estimated frequencies
    :return: rpa
    """

    # remove frames containing only zero frequencies in reference or estimation
    nonzero_freqs = np.logical_and(np.any(ref_cents, axis=1) != 0.0, np.any(est_cents, axis=1) != 0.0)

    if np.sum(nonzero_freqs) == 0:
        return 0.0, 0.0

    # keep only non zeros frequencies in frames
    ref_cent_nz = [cent[cent != 0.0] for cent in ref_cents[nonzero_freqs]]
    est_cent_nz = [cent[cent != 0.0] for cent in est_cents[nonzero_freqs]]

    # calculate number of true positives, references and estimation for each frame
    true_positives = compute_num_true_positives(ref_cent_nz, est_cent_nz, cent_tolerance)
    n_ref = compute_num_freqs(ref_cent_nz)
    n_est = compute_num_freqs(est_cent_nz)

    # calculate rpa using recall (recall and rpa have same value when keep only non zeros frequencies )
    _, rpa, _ = compute_accuracy(true_positives, n_ref, n_est)
    substitution_error, miss_error, false_alarm, total_error = compute_err_score(true_positives, n_ref, n_est)

    # rpa calculated without mir_eval functions (it works, same result)
    # total_true_positives = np.sum(true_positives)
    # total_ref = np.sum(ref_cent_nz != 0.0)
    # rpa = total_true_positives / total_ref

    return rpa, substitution_error, miss_error, false_alarm, total_error
