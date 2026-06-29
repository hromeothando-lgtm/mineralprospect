"""
Evaluation metrics for FractalProspect-Net v4.
Primary metric: AUC-PR (correct for class imbalance).
Secondary: AUC-ROC, F1 at Youden threshold, SPE curve.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)


def youden_threshold(y_true, y_prob):
    """
    Maximum Youden Index threshold from ROC curve.
    J = Sensitivity + Specificity - 1
    Selects threshold that maximises J — not arbitrary.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    idx = np.argmax(J)
    return thresholds[idx], J[idx]


def compute_all_metrics(y_true, y_prob):
    """
    Returns dict of all publication metrics.
    Primary: AUC-PR. Secondary: AUC-ROC, F1@Youden, SPE.
    """
    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    thresh, youden_j = youden_threshold(y_true, y_prob)
    y_pred  = (y_prob >= thresh).astype(int)
    f1      = f1_score(y_true, y_pred, zero_division=0)

    return {
        'auc_pr'   : auc_pr,
        'auc_roc'  : auc_roc,
        'f1_youden': f1,
        'threshold': thresh,
        'youden_j' : youden_j,
    }


def spatial_prediction_efficiency(y_true, y_prob, n_points=100):
    """
    SPE curve: prospective area fraction vs deposit capture rate.
    Exploration industry standard validation metric.
    Returns arrays for plotting.
    """
    thresholds = np.linspace(y_prob.max(), y_prob.min(), n_points)
    area_fracs, capture_rates = [], []
    for t in thresholds:
        pred_pos = (y_prob >= t)
        area_frac    = pred_pos.mean()
        capture_rate = y_true[pred_pos].sum() / max(y_true.sum(), 1)
        area_fracs.append(area_frac)
        capture_rates.append(capture_rate)
    return np.array(area_fracs), np.array(capture_rates)
