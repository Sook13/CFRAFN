"""Evaluation utilities for binary classification models."""

from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


def g_mean(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute geometric mean of sensitivity and specificity."""
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1, 1] / cm[1, :].sum()
    tnr = cm[0, 0] / cm[0, :].sum()
    return np.sqrt(tpr * tnr)


def evaluate_model(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_pred_prob: Iterable[float],
) -> Dict[str, float | np.ndarray]:
    """Calculate common binary-classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall_arr, precision_arr)

    gmean = g_mean(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    fp = cm[0][1]

    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "acc": acc,
        "ppv": precision,
        "sensitivity": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "aupr": aupr,
        "gmean": gmean,
        "kappa": kappa,
        "npv": npv,
        "specificity": specificity,
        "fpr": fpr,
        "tpr": tpr,
    }


def calculate_mean_std_metrics(
    metrics_list: List[Dict[str, float | np.ndarray]],
) -> Dict[str, float]:
    """Aggregate metric mean and standard deviation across repeated runs."""
    metrics_summary: Dict[str, float] = {}
    for metric in metrics_list[0].keys():
        if metric not in ["fpr", "tpr"]:
            values = [metric_item[metric] for metric_item in metrics_list]
            metrics_summary[f"{metric}_mean"] = float(np.mean(values))
            metrics_summary[f"{metric}_std"] = float(np.std(values))
    return metrics_summary


def plot_roc_curve(metrics_list: List[Dict[str, float | np.ndarray]]) -> float:
    """Plot mean ROC curve and return AUC of the mean curve."""
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []

    for metrics in metrics_list:
        fpr, tpr = metrics["fpr"], metrics["tpr"]
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr_all.append(interp_tpr)

    mean_tpr = np.mean(interp_tpr_all, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color="b", lw=2, label=f"Model (AUC = {mean_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.show()
    return float(mean_auc)


def overall_evaluate_plot(
    metrics_list: List[Dict[str, float | np.ndarray]],
) -> Dict[str, float]:
    """Plot ROC summary and return aggregate metrics."""
    plot_roc_curve(metrics_list)
    return calculate_mean_std_metrics(metrics_list)


def calculate_aupr(y_true: Iterable[int], y_pred_prob: Iterable[float]) -> float:
    """Calculate area under the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    return auc(recall, precision)


def calculate_npv(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Calculate negative predictive value."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    return tn / (tn + fn) if (tn + fn) > 0 else 0


def calculate_specificity(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Calculate specificity."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0]
    fp = cm[0][1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def bootstrap_ci(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_pred_prob: Iterable[float],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    use_proba: bool = False,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float, List[float]]:
    """Estimate BCa-like confidence intervals using bootstrap sampling."""
    np.random.seed(42)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_pred_prob_arr = np.array(y_pred_prob)

    n_samples = len(y_true_arr)
    metric_values: List[float] = []
    original_metric = (
        metric_func(y_true_arr, y_pred_prob_arr) if use_proba else metric_func(y_true_arr, y_pred_arr)
    )

    for _ in tqdm(range(n_bootstrap)):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true_arr[indices]
        y_pred_boot = y_pred_arr[indices]
        y_pred_prob_boot = y_pred_prob_arr[indices]

        metric_boot = (
            metric_func(y_true_boot, y_pred_prob_boot)
            if use_proba
            else metric_func(y_true_boot, y_pred_boot)
        )
        metric_values.append(float(metric_boot))

    z0 = np.sum(np.array(metric_values) < original_metric) / n_bootstrap
    z_alpha = norm.ppf(1 - alpha / 2)
    lower_percentile = 100 * norm.cdf(2 * z0 - z_alpha)
    upper_percentile = 100 * norm.cdf(2 * z0 + z_alpha)

    ci_lower = np.percentile(metric_values, lower_percentile)
    ci_upper = np.percentile(metric_values, upper_percentile)
    return float(original_metric), float(ci_lower), float(ci_upper), metric_values


def evaluate_model_ci(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_pred_prob: Iterable[float],
) -> Dict[str, Dict[str, float | Tuple[float, float] | List[float]]]:
    """Compute core metrics together with bootstrap confidence intervals."""
    metrics: Dict[str, Tuple[Callable[[np.ndarray, np.ndarray], float], bool]] = {
        "acc": (accuracy_score, False),
        "ppv": (lambda y, pred: precision_score(y, pred, zero_division=0), False),
        "sensitivity": (recall_score, False),
        "specificity": (calculate_specificity, False),
        "f1": (f1_score, False),
        "roc_auc": (roc_auc_score, True),
        "gmean": (g_mean, False),
        "kappa": (cohen_kappa_score, False),
        "npv": (calculate_npv, False),
    }

    results: Dict[str, Dict[str, float | Tuple[float, float] | List[float]]] = {}
    for name, (func, use_proba) in metrics.items():
        original_metric, ci_lower, ci_upper, metric_values = bootstrap_ci(
            y_true,
            y_pred,
            y_pred_prob,
            func,
            use_proba=use_proba,
        )
        results[name] = {
            "value": original_metric,
            "95% CI": (ci_lower, ci_upper),
            "all value": metric_values,
        }

    return results


# Backward-compatible alias for existing external calls.
evaluate_model_CI = evaluate_model_ci
