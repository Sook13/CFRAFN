import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(labels, preds, fold, save_dir):
    """
    绘制混淆矩阵并保存为图片。
    - cm : 计算混淆矩阵的值
    - annot : 是否显示每个单元格的数据值
    - cm_percent : 是否显示每个单元格的数据百分比
    - fold : 第几折，-1表示全部
    """
    cm = confusion_matrix(labels, preds)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = cm.astype('float') * 100 / cm_sum

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Non-depressed", "Depressed"], yticklabels=["Non-depressed", "Depressed"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if fold == -1:
        plt.title("Confusion Matrix - Overall")
        save_path = os.path.join(save_dir, "confusion_matrix_overall.png")
    else:
        plt.title(f"Confusion Matrix - Fold {fold + 1}")
        save_path = os.path.join(save_dir, f"confusion_matrix_fold_{fold + 1}.png")
    plt.savefig(save_path)
    plt.close()

# 绘制 ROC 曲线
def plot_roc_curve(roc_data):
    """
    绘制 ROC 曲线并保存为图片。
    - roc_data : 包含模型名称和每折 ROC 值的字典
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, model_data in roc_data.items():
        tprs = []
        aucs = []
        base_fpr = np.linspace(0, 1, 101)

        for fold_data in model_data:
            if 'fpr' in fold_data and 'tpr' in fold_data and 'auc' in fold_data:
                fpr = fold_data['fpr']
                tpr = fold_data['tpr']
                tprs.append(np.interp(base_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(fold_data['auc'])
            else:
                print(f"Warning: Missing 'fpr', 'tpr', or 'auc' in {model_name} data. Skipping this fold.")

        if tprs and aucs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)  # 使用每折的 AUC 值的平均值
            plt.plot(base_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (5-Fold CV Average)')
    plt.legend()
    plt.grid(True)
    plt.show()