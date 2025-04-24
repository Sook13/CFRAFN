# -*- coding:utf-8 -*-
# @FileName  :evaluate.py
# @Time      :2024/11/20 16:37:45
# @Author    :mizzle

from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,matthews_corrcoef,precision_score,recall_score
from sklearn.metrics import f1_score,auc,precision_recall_curve,accuracy_score,roc_curve,cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

#计算几何均值
def g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1, 1] / cm[1, :].sum()
    tnr = cm[0, 0] / cm[0, :].sum()
    gmean = np.sqrt(tpr * tnr)
    return gmean

def evaluate_model(y_true, y_pred, y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # auc_value = roc_auc_score(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall_arr, precision_arr)
    
    gmean = g_mean(y_true,y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]

    npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # NPV
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    evaluate_dic={
        "acc": acc, "ppv": precision, "sensitivity": recall, "f1": f1,
        "roc_auc": roc_auc, "mcc": mcc, "aupr": aupr, "gmean": gmean,
        "kappa": kappa, "npv": npv, "specificity":specificity, "fpr": fpr, "tpr": tpr
    }

    return evaluate_dic

#计算每个指标的均值和标准差
def calculate_mean_std_metrics(metrics_list):
    # 剔除fpr和tpr，只统计其余指标的平均值和标准差
    metrics_summary = {}
    for metric in metrics_list[0].keys():
        if metric not in ['fpr', 'tpr']: #fpr和tpr不计算均值和标准差
            values = [d[metric] for d in metrics_list]
            metrics_summary[f"{metric}_mean"] = np.mean(values)
            metrics_summary[f"{metric}_std"] = np.std(values)

#绘制ROC曲线
def plot_roc_curve(metrics_list):
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []
    
    for metrics in metrics_list:
        fpr, tpr = metrics["fpr"], metrics["tpr"]
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # 确保起点为0
        interp_tpr_all.append(interp_tpr)
    
    mean_tpr = np.mean(interp_tpr_all, axis=0)
    mean_tpr[-1] = 1.0  # 确保终点为1
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f"Model (AUC = {mean_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()

def overall_evaluate_plot(metrics_list):
    mean_auc = plot_roc_curve(metrics_list)
    metrics_summary = calculate_mean_std_metrics(metrics_list)
    # print(metrics_summary)
    return metrics_summary


def calculate_aupr(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    return auc(recall, precision)

def calculate_npv(y_true, y_pred):
    # Confusion matrix for NPV
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    return npv

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return specificity

    
# 计算给定评价指标的置信区间
def bootstrap_ci(y_true, y_pred, y_pred_prob, metric_func, use_proba=False, n_bootstrap=1000, alpha=0.05):
    """
    为给定的评价指标计算置信区间
    """
    np.random.seed(42)
    n_samples = len(y_true)
    metric_values = []
    if use_proba:
        original_metric = metric_func(y_true, y_pred_prob)  # 根据函数计算原始指标使用概率值
    else:
        original_metric = metric_func(y_true, y_pred)  # 根据函数计算原始指标使用预测标签
    
    y_true = np.array(y_true) 
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)
    # Bootstrap采样
    for _ in tqdm(range(n_bootstrap)):
        indices = np.random.choice(n_samples, n_samples, replace=True) #
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_pred_prob_boot = y_pred_prob[indices]
        if use_proba:
            metric_boot = metric_func(y_true_boot, y_pred_prob_boot)
        else:
            metric_boot = metric_func(y_true_boot, y_pred_boot)
        metric_values.append(metric_boot)
    
    # 偏倚校正的百分位数法
    z0 = np.sum(np.array(metric_values) < original_metric) / n_bootstrap  # 偏差因子
    z_alpha = norm.ppf(1 - alpha/2)  # 标准正态分布的分位数
    
    # 计算校正后的分位数
    lower_percentile = 100 * norm.cdf(2 * z0 - z_alpha)
    upper_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    
    ci_lower = np.percentile(metric_values, lower_percentile)
    ci_upper = np.percentile(metric_values, upper_percentile)
    
    return original_metric, ci_lower, ci_upper, metric_values

# 95%置信CI 
def evaluate_model_CI(y_true, y_pred, y_pred_prob):
    metrics = {
        'acc': (accuracy_score, False),
        'ppv': (lambda y, y_pred: precision_score(y, y_pred, zero_division=0), False),
        'sensitivity': (recall_score, False),
        'specificity': (calculate_specificity, False),
        'f1': (f1_score, False),
        'roc_auc': (lambda y, y_pred_prob: roc_auc_score(y, y_pred_prob), True),
        'gmean': (g_mean, False),
        'kappa': (cohen_kappa_score, False),
        'npv': (calculate_npv, False)
    }
    
    results = {}
    for name, (func, use_proba) in metrics.items():
        original_metric, ci_lower, ci_upper, metric_values = bootstrap_ci(y_true, y_pred, y_pred_prob, func, use_proba=use_proba)
        results[name] = {
            'value': original_metric,
            '95% CI': (ci_lower, ci_upper),
            'all value': metric_values
        }
    
    return results