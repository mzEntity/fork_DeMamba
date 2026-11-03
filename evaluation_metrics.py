"""Video-level binary metrics; fake=1 and real=0."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def detection_metrics(labels, fake_scores, max_fpr=0.05):
    """Evaluate the loosest empirical ROC threshold with FPR <= max_fpr.

    Predict fake when score >= threshold. No interpolation or splitting tied
    scores is used, so the achieved FPR can be below the requested rate.
    AUC and the operating-point metrics are NaN for single-class inputs.
    """
    labels = np.asarray(labels)
    scores = np.asarray(fake_scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not labels.size:
        raise ValueError('labels and fake_scores must be nonempty matching 1-D arrays')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('labels must be 0 (real) or 1 (fake)')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('fake_scores must be finite probabilities in [0, 1]')
    if not 0 <= max_fpr < 1:
        raise ValueError('max_fpr must be in [0, 1)')
    result = dict.fromkeys(('auc', 'acc_at_fpr', 'threshold', 'fpr', 'tpr'), float('nan'))
    if np.unique(labels).size < 2:
        return result

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    point = np.flatnonzero(fpr <= max_fpr)[-1]
    threshold = float(thresholds[point])
    predictions = scores >= threshold
    result.update(
        auc=float(roc_auc_score(labels, scores)),
        acc_at_fpr=float(accuracy_score(labels, predictions)),
        threshold=threshold, fpr=float(fpr[point]), tpr=float(tpr[point]),
    )
    return result
