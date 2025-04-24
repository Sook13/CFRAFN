import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import gmean
from sklearn.model_selection import StratifiedKFold
import torch
from sklearn.tree import DecisionTreeClassifier
from torchvggish import vggish, vggish_input
import matplotlib.pyplot as plt
from tools.evaluate import evaluate_model
from tools.common import setup_seed
import os
import warnings
import joblib
import seaborn as sns
import swanlab


warnings.filterwarnings("ignore")
# 0.99不错
with open("../result/01preprocess/eGe_feature_cumul1.txt",'r')as f:
    eGe_feature = [line.strip() for line in f]

def read_data(save_path):

    X_train_eGe_list_path = os.path.join(save_path,"X_train_eGe_list.joblib")
    X_test_eGe_list_path = os.path.join(save_path,"X_test_eGe_list.joblib")
    X_train_VGGish_list_path = os.path.join(save_path,"X_train_VGGish_list.joblib")
    X_test_VGGish_list_path = os.path.join(save_path,"X_test_VGGish_list.joblib")
    y_train_list_path = os.path.join(save_path,"y_train_list.joblib")
    y_test_list_path = os.path.join(save_path,"y_test_list.joblib")

    X_train_eGe_list = joblib.load(X_train_eGe_list_path)
    X_test_eGe_list = joblib.load(X_test_eGe_list_path)
    X_train_VGGish_list = joblib.load(X_train_VGGish_list_path) 
    X_test_VGGish_list = joblib.load(X_test_VGGish_list_path)
    y_train_list = joblib.load(y_train_list_path)
    y_test_list = joblib.load(y_test_list_path)

    return X_train_eGe_list,X_test_eGe_list,X_train_VGGish_list,X_test_VGGish_list,y_train_list,y_test_list


# 保存比对结果到excel
def save_results_to_excel(results, filename, suffix):
    df = pd.DataFrame(results)

    cols = ['Model','f1_mean','f1_std','roc_auc_mean','roc_auc_std','npv_mean','npv_std',
            'aupr_mean','aupr_std','gmean_mean','gmean_std','kappa_mean','kappa_std','mcc_mean','mcc_std',
            'acc_mean','acc_std','ppv_mean','ppv_std','sensitivity_mean','sensitivity_std','specificity_mean','specificity_std']
    df = df[cols]
    df.to_excel(filename+f"{suffix}.xlsx", index=False)

    # 合并mean和std，结果为mean±std
    results = pd.DataFrame()
    results['Model'] = df['Model']
    for col in df.columns:
        if '_mean' in col:
            metric = col.split('_mean')[0]  # 获取基本指标名称
            mean_col = col
            std_col = metric + '_std'
            
            # 创建新的列格式 "指标名: mean +/- std"
            results[metric] = df.apply(
                lambda row: f"{row[mean_col]:.3f}±{row[std_col]:.3f}", 
                axis=1
            )

    results.to_excel(filename+f"{suffix}_MeanStd50.xlsx", index=False)


def train_and_evaluate_model(model_name, model, color, suffix):
    # print(f"Training {model_name}...")

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []  # 用于存储插值后的TPR
    # 存储每个fold的评价结果
    metrics_all = {
        "acc": [], "ppv": [], "sensitivity": [], "f1": [],
        "roc_auc": [], "mcc": [], "aupr": [], "gmean": [],
        "npv": [], "kappa": [], "specificity": [],
    }
    auc_list = []

    save_path = f"../data/enhance{suffix}/"
    X_train_eGe_list,X_test_eGe_list,X_train_VGGish_list,X_test_VGGish_list,y_train_list,y_test_list = read_data(save_path)

    for i in range(len(X_train_eGe_list)):
        
        # 获取预先提取的特征
        X_train_eGe, X_test_eGe = X_train_eGe_list[i], X_test_eGe_list[i]
        X_train_VGGish, X_test_VGGish = X_train_VGGish_list[i], X_test_VGGish_list[i]
        y_train, y_test = y_train_list[i], y_test_list[i]

        # minmax_transfer = MinMaxScaler()
        # X_train_eGe_array = minmax_transfer.fit_transform(X_train_eGe)
        # X_test_eGe_array = minmax_transfer.fit_transform(X_test_eGe)
        # X_train_VGGish_array = minmax_transfer.fit_transform(X_train_VGGish)
        # X_test_VGGish_array = minmax_transfer.fit_transform(X_test_VGGish)

        # X_train_eGe = pd.DataFrame(X_train_eGe_array, columns=X_train_eGe.columns)
        # X_test_eGe = pd.DataFrame(X_test_eGe_array, columns=X_test_eGe.columns)
        # X_train_VGGish = pd.DataFrame(X_train_VGGish_array, columns=X_train_VGGish.columns)
        # X_test_VGGish = pd.DataFrame(X_test_VGGish_array, columns=X_test_VGGish.columns)
        
        # 拼接特征
        X_train = np.hstack((X_train_eGe[eGe_feature], X_train_VGGish))
        X_test = np.hstack((X_test_eGe[eGe_feature], X_test_VGGish))

        # random_indices = np.random.choice(X_train.shape[0], size=int(X_train.shape[0]/7), replace=False)
        # X_train = X_train[random_indices, :]
        # y_train = y_train.iloc[random_indices]

        X_train = X_train[::6, :]
        y_train = y_train.iloc[::6]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 评估
        metrics = evaluate_model(y_test, y_pred, y_pred_prob)
        # print(f"y_test: {np.array(y_test)}")
        # print(f"y_pred_prob: {np.round(y_pred_prob, 2)}")
        # print(f"y_pred: {y_pred}\n\n")
        for key in metrics_all.keys():
            metrics_all[key].append(metrics[key])

        # AUC曲线数据
        fpr, tpr = metrics["fpr"], metrics["tpr"]
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr_all.append(interp_tpr)

        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)

    # 计算每个指标的均值和标准差
    metrics_summary = {f"{metric}_mean": np.mean(values) for metric, values in metrics_all.items()}
    metrics_summary.update({f"{metric}_std": np.std(values) for metric, values in metrics_all.items()})

    # 计算平均TPR和AUC
    mean_tpr = np.mean(interp_tpr_all, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics_summary['roc_auc_mean']

    # 绘制AUC曲线
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=f"{model_name} (AUC = {mean_auc:.3f})")

    return metrics_summary


def compare_models(save_path, suffix):

    # models = {
    # "LR": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42),
    # "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=2),
    # "svm": SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
    # "NB": GaussianNB(var_smoothing=1e-9),
    # "DT": DecisionTreeClassifier(criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, random_state=42),
    # "RF": RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True, random_state=42),
    # "Bagging": BaggingClassifier(n_estimators=10, max_samples=1.0, max_features=1.0, random_state=42),
    # "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0,random_state=42 ),
    # "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1,max_depth=3,subsample=1.0,colsample_bytree=1.0,random_state=42,eval_metric='logloss'),
    # "LightGBM": LGBMClassifier(n_estimators=100,learning_rate=0.1,max_depth=-1,num_leaves=31,verbosity=-1,random_state=42),
    # }

    models = {
    "LR": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "svm": SVC(probability=True),
    "NB": GaussianNB(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "Bagging": BaggingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(verbosity=-1),
    }

    setup_seed(42)
    # colors = sns.color_palette("tab10", len(models))
    colors = sns.hls_palette(11,l=.5, s=.7)
    sns.set_style("ticks")
    results = []
    plt.figure(figsize=(10, 8))

    for model_info, color in zip(models.items(), colors):
        metrics_summary = train_and_evaluate_model(model_info[0],model_info[1], color, suffix)
        swanlab.log({f"{model_info[0]}/F1": f"{metrics_summary['f1_mean']:.4f}"})
        swanlab.log({f"{model_info[0]}/AUC": f"{metrics_summary['roc_auc_mean']:.4f}"})
        swanlab.log({f"{model_info[0]}/ACC": f"{metrics_summary['acc_mean']:.4f}"})
        print(f"{model_info[0]}:\tF1: {metrics_summary['f1_mean']:.4f}\tAUC: {metrics_summary['roc_auc_mean']:.4f}\tACC: {metrics_summary['acc_mean']:.4f}")
       
        metrics_summary["Model"] = model_info[0]
        results.append(metrics_summary)

    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_visible(False)  # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['bottom'].set_linewidth(2)  # 加粗X轴
    ax.spines['left'].set_linewidth(2)  # 加粗Y轴
    # 加粗坐标轴刻度
    ax.xaxis.set_tick_params(width=2)  # X轴刻度加粗
    ax.yaxis.set_tick_params(width=2)  # Y轴刻度加粗
    # 设置刻度标签的字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)  # 参考线
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold',c='k')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold',c='k')
    plt.tick_params(axis='x', colors='k')
    plt.tick_params(axis='y', colors='k')
    plt.tick_params(axis='x', labelcolor='k')
    plt.tick_params(axis='y', labelcolor='k')
    # plt.title(f'AUC under feature sets', fontsize=16, fontweight='bold',c='k')
    plt.legend(loc="lower right", prop={'weight':'bold', 'size':12}, labelcolor='black', edgecolor='black', framealpha=1, frameon=True, title='Models', title_fontproperties={'weight':'bold', 'size':13})
    plt.savefig(os.path.join(save_path,f"compare_ROC{suffix}.tif"),dpi=300,bbox_inches="tight")
    plt.savefig(os.path.join(save_path,f"compare_ROC{suffix}.pdf"),dpi=600,bbox_inches="tight")

    save_results_to_excel(results,os.path.join(save_path,"compareModel"), suffix)


def main():
    plt.rcParams['font.family'] = 'Arial'
    save_path = "../result/03compare"
    suffix = "_10" # suffix must be "" or "_10"
    assert suffix == "" or "_10", 'suffix must be empty or _10'
    compare_models(save_path=save_path, suffix=suffix)


if __name__ == "__main__":
    run = swanlab.init(
        project="machine-learning-compared",
        experiment_name= "单次五折 进行数据增强 不做标准化",
        description = "",
        mode='disabled', # 调试模式
    )
    main()
    
