"""Prediction metrics. Undefined AUROCs remain missing, not imputed."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             balanced_accuracy_score, roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss)


def choose_thresholds(y, probabilities):
    result = []
    for j in range(y.shape[1]):
        choices = np.linspace(0.05, 0.95, 91)
        scores = [f1_score(y[:, j], probabilities[:, j] >= t, zero_division=0) for t in choices]
        result.append(float(choices[np.argmax(scores)]))
    return np.asarray(result)


def safe_macro_auc(y, proba, multilabel):
    try:
        if multilabel:
            values = [roc_auc_score(y[:, j], proba[:, j]) for j in range(y.shape[1])
                      if np.unique(y[:, j]).size == 2]
            return float(np.mean(values)) if values else None
        values = [roc_auc_score((y == j).astype(int), proba[:, j]) for j in range(proba.shape[1])
                  if np.unique(y == j).size == 2]
        return float(np.mean(values)) if values else None
    except ValueError:
        return None


def evaluate(y, proba, multilabel=False, thresholds=None):
    y = np.asarray(y)
    proba = np.asarray(proba)
    if multilabel:
        threshold = np.full(y.shape[1], .5) if thresholds is None else np.asarray(thresholds)
        pred = (proba >= threshold).astype(int)
        ap = [average_precision_score(y[:, j], proba[:, j]) for j in range(y.shape[1])
              if len(np.unique(y[:, j])) > 1]
        return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
                'macro_auroc': safe_macro_auc(y, proba, True),
                'macro_auprc': float(np.mean(ap)) if ap else None,
                'label_accuracy': float(np.mean(pred == y)),
                'exact_match_accuracy': float(np.mean(np.all(pred == y, axis=1))),
                'brier': float(np.mean((proba-y)**2))}
    pred = proba.argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
            'macro_precision': float(precision_score(y, pred, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y, pred, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_auroc': safe_macro_auc(y, proba, False),
            'brier': float(np.mean((proba-np.eye(proba.shape[1])[y])**2))}
