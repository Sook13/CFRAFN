import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tools.common import setup_seed
from tqdm import tqdm
import os
import joblib
from torchvggish import vggish, vggish_input
from joblib import Parallel, delayed, dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from tools.evaluate import evaluate_model, plot_roc_curve, overall_evaluate_plot, calculate_mean_std_metrics
from tools.common import setup_seed,init_logger
from tools.utils import get_eGe_matrix,get_vggish_features,get_best_para_from_optuna
from tools.model import CFRAFN
from joblib import Parallel, delayed
import random
import optuna
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

setup_seed(42)
LOGFILE = f"../result/depression_optuna_f1.log"
logger,file_handler = init_logger(LOGFILE)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("../data/group_control.csv")
shuffled_df = df.sample(frac=1).reset_index(drop=True)

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

with open("../result/01preprocess/eGe_feature_cumul1.txt",'r')as f:
    eGe_feature = [line.strip() for line in f]

feature_weights = pd.read_csv('../result/01preprocess/03sorted_feature_importance.csv',index_col=0)
feature_weights = feature_weights.squeeze()
feature_weights = feature_weights[eGe_feature]

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(shuffled_df['class']), y=shuffled_df['class'])
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)


def train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                   sample_weight=0.5, epochs=50, patience=10, epoch_lst=None, save_flag=False):
    setup_seed(42)
    best_f1 = 0.0  # 用于记录最佳 F1 分数
    no_improvement_count = 0  # 记录未提升的轮数

    # 将元组中的特征分别提取出来
    X_train_eGe, X_train_VGGish = X_train
    X_test_eGe, X_test_VGGish = X_test

    # 确保输入是 NumPy 数组
    X_train_eGe = np.asarray(X_train_eGe)
    X_train_VGGish = np.asarray(X_train_VGGish)
    X_test_eGe = np.asarray(X_test_eGe)
    X_test_VGGish = np.asarray(X_test_VGGish)

    # 将特征转换为张量
    X_train_eGe = torch.tensor(X_train_eGe, dtype=torch.float32).to(device)
    X_train_VGGish = torch.tensor(X_train_VGGish, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_test_eGe = torch.tensor(X_test_eGe, dtype=torch.float32).to(device)
    X_test_VGGish = torch.tensor(X_test_VGGish, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    groups = [list(range(i, min(i+6, len(X_train_eGe)))) for i in range(0, len(X_train_eGe), 6)]
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        selected_indices = [random.choices(group, weights=[sample_weight] + [(1-sample_weight)/(len(group)-1)]*(len(group)-1), k=1)[0] for group in groups]

        # 获取选中的数据
        X_train_eGe_select = X_train_eGe[selected_indices]
        X_train_VGGish_select = X_train_VGGish[selected_indices]
        y_train_select = y_train[selected_indices]

        outputs = model(X_train_eGe_select, X_train_VGGish_select).squeeze()  # 分别传入 eGeMAPS 和 VGGish 特征
        loss = criterion(outputs, y_train_select)
        loss.backward()
        optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_eGe, X_test_VGGish).squeeze()  # 分别传入 eGeMAPS 和 VGGish 特征
            predictions_prob = val_outputs # 预测概率       
            y_pred = (predictions_prob >= 0.5).int().cpu().numpy() # 预测标签
            y_pred_prob = predictions_prob.cpu().numpy()         
            evaluta_dic = evaluate_model(y_test.cpu().numpy(), y_pred, y_pred_prob) 
        
        epoch_lst[f'epoch_{epoch}'].append(evaluta_dic)
        
        # # 早停检测
        # if evaluta_dic['f1'] > best_f1:
        #     best_f1 = evaluta_dic['f1']
        #     no_improvement_count = 0  # 重置计数器
        #     if save_flag:
        #         torch.save(model.state_dict(), "../result/02optuna/best_EgvAtt_model.pth")  # 保存当前最佳模型
        # else:
        #     no_improvement_count += 1

        # # 如果没有提升的轮数超过阈值，停止训练
        # if no_improvement_count >= patience:
        #     # print(f"Early stopping triggered. Best F1: {best_f1:.4f}")
        #     break

    return epoch_lst


def train_and_evaluate_model(model_param=None):
    setup_seed(42)
    # train param
    # model_param = {'lr': 0.00010673603218681096, 'conv_out_channels': 63, 'transformed_feature_dim': 200, 'cls_dim': 214, 'sample_weight': 0.03480728505243403}
    # lr = model_param['lr']  # 2e-4
    # conv_out_channels = model_param['conv_out_channels']
    # transformed_feature_dim = model_param['transformed_feature_dim']
    # cls_dim = model_param['cls_dim']
    # sample_weight = model_param['sample_weight']

    lr = 1e-3  # 2e-4
    conv_out_channels = 64
    transformed_feature_dim = 256
    cls_dim = 128
    sample_weight = 0.4
    
    weight_decay = 2e-4
    expansion = 2
    dropout = 0.1
    resblock_kernel_size = 5

    total_epoch = 2000
    patience = 200

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []  # 用于存储插值后的TPR

    metrics_all = {
        "acc": [], "ppv": [], "sensitivity": [], "f1": [],
        "roc_auc": [], "mcc": [], "aupr": [], "gmean": [],
        "npv": [], "kappa": [], "specificity": [],
    }
    auc_list = []
    data_path = "../data/enhance/"
    epoch_lst = {f"epoch_{i}":[] for i in range(total_epoch)}
    X_train_eGe_list,X_test_eGe_list,X_train_VGGish_list,X_test_VGGish_list,y_train_list,y_test_list = read_data(data_path)
    # print(f"data item len： {len(X_train_eGe_list)}")
    for i in tqdm(range(len(X_train_eGe_list))):
        # print(f"第{i+1}轮......")
        
        # 获取预先提取的特征
        X_train_eGe = X_train_eGe_list[i]
        X_test_eGe = X_test_eGe_list[i]
        X_train_VGGish = X_train_VGGish_list[i]
        X_test_VGGish = X_test_VGGish_list[i]
        y_train = y_train_list[i]
        y_test = y_test_list[i]

        X_train = (X_train_eGe[eGe_feature], X_train_VGGish)
        X_test = (X_test_eGe[eGe_feature], X_test_VGGish)
        
        model = CFRAFN(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, 
                        feature_weights=feature_weights, expansion=expansion, dropout=dropout,
                        conv_out_channels=conv_out_channels,
                        transformed_feature_dim=transformed_feature_dim, 
                        resblock_kernel_size=resblock_kernel_size, cls_dim=cls_dim).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        #训练并评估模型
        metric_epoch_dic = train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                        sample_weight=sample_weight, epochs=total_epoch, patience=patience, epoch_lst=epoch_lst)

    average_metrics = {}
    standard_metrics = {} ## 用于存储每个 epoch 的标准差
    
    for epoch, metric_list in metric_epoch_dic.items():
        temp_metrics = defaultdict(list)
        # 收集所有重复实验的指标值
        for metric_dict in metric_list:
            for key in ["acc", "f1", "roc_auc", "ppv", "npv", "sensitivity", "specificity", "mcc", "aupr", "gmean", "kappa"]:
                temp_metrics[key].append(metric_dict[key])

    #    "fpr": fpr, "tpr": tpr

        # 计算平均值并保存
        average_metrics[epoch] = {
            key: np.mean(values) 
            for key, values in temp_metrics.items()
        }

        # 计算标准差并保存
        standard_metrics[epoch] = {
            key: np.std(values)
            for key, values in temp_metrics.items()
        }

    # 找到 f1 最大的 epoch 及其所有指标
    best_epoch, best_metrics = max(
        average_metrics.items(),
        key=lambda item: item[1]["f1"]
    )
    # max_f1 = max(metrics["f1"] for metrics in average_metrics.values())
    # print(max_f1)

    print(f"最大 f1 值: {best_metrics['f1']}")
    print(f"对应 epoch: {best_epoch}")
    print("该 epoch 的所有平均指标:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("该 epoch 的所有标准差:")
    for metric, value in standard_metrics[best_epoch].items():
        print(f"{metric}: {value:.4f}")

    # 创建 metrics_summary 字典
    metrics_summary = {
    "best_metrics": {k: best_metrics[k] for k in ["acc", "f1", "roc_auc", "ppv", "npv", "sensitivity", "specificity", "mcc", "aupr", "gmean", "kappa"]},
    "best_metrics_std": {k: standard_metrics[best_epoch][k] for k in ["acc", "f1", "roc_auc", "ppv", "npv", "sensitivity", "specificity", "mcc", "aupr", "gmean", "kappa"]}
}
    
    return metrics_summary



def objective(trial):
    setup_seed(42)
    # train param
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)  # 2e-4
    conv_out_channels = trial.suggest_int('conv_out_channels', 32, 256)
    transformed_feature_dim = trial.suggest_int('transformed_feature_dim', 64, 512)
    cls_dim = trial.suggest_int('cls_dim', 64, 256)
    sample_weight = trial.suggest_float('sample_weight', 0.01, 0.99)

    weight_decay = 2e-4
    expansion = 2
    dropout = 0.1
    resblock_kernel_size = 5

    total_epoch = 2000
    patience = 200

    data_path = "../data/enhance/"
    X_train_eGe_list,X_test_eGe_list,X_train_VGGish_list,X_test_VGGish_list,y_train_list,y_test_list = read_data(data_path)
    f1_lst = []
    # print(f"data item len： {len(X_train_eGe_list)}")
    epoch_lst = {f"epoch_{i}":[] for i in range(total_epoch)}
    
    for i in range(len(X_train_eGe_list)):
        X_train_eGe = X_train_eGe_list[i]
        X_test_eGe = X_test_eGe_list[i]
        X_train_VGGish = X_train_VGGish_list[i]
        X_test_VGGish = X_test_VGGish_list[i]
        y_train = y_train_list[i]
        y_test = y_test_list[i]

        X_train = (X_train_eGe[eGe_feature], X_train_VGGish)
        X_test = (X_test_eGe[eGe_feature], X_test_VGGish)

        assert feature_weights.index.tolist() == X_train[0].columns.tolist(), "feature_weights order must be equal to the order of data columns"
        setup_seed(42)
        model = CFRAFN(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, 
                        feature_weights=feature_weights, expansion=expansion, dropout=dropout,
                        conv_out_channels=conv_out_channels,
                        transformed_feature_dim=transformed_feature_dim, 
                        resblock_kernel_size=resblock_kernel_size, cls_dim=cls_dim).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        #训练并评估模型
        metric_epoch_dic = train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                        sample_weight=sample_weight, epochs=total_epoch, patience=patience, epoch_lst=epoch_lst)
        # f1_lst.append(f1)
    average_metrics = {}
    for epoch, metric_list in metric_epoch_dic.items():
        temp_metrics = defaultdict(list)
        # 收集所有重复实验的指标值
        for metric_dict in metric_list:
            for key in ["acc", "f1", "roc_auc"]:
                temp_metrics[key].append(metric_dict[key])

        # 计算平均值并保存
        average_metrics[epoch] = {
            key: np.mean(values) 
            for key, values in temp_metrics.items()
        }

    max_f1 = max(metrics["roc_auc"] for metrics in average_metrics.values())
    best_epoch, best_metrics = max(
        average_metrics.items(),
        key=lambda item: item[1]["f1"]
    )

    return best_metrics['roc_auc']

# print("\nInitialized parameters:")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Shape: {param.shape} | Values (first 5): {param.data.flatten()[:5]}")

import pandas as pd

def save_metrics_summary_to_excel(metrics_summary, filename):
    # best_epoch = metrics_summary["best_epoch"]
    best_metrics = metrics_summary["best_metrics"]
    best_metrics_std = metrics_summary["best_metrics_std"]

    # 创建一个 DataFrame 来存储 mean 和 std
    df = pd.DataFrame()
    for metric in best_metrics:
        df[f"{metric}_mean"] = [best_metrics[metric]]
        df[f"{metric}_std"] = [best_metrics_std[metric]]

    # 合并 mean 和 std 列
    results = pd.DataFrame()

    for col in df.columns:
        if "_mean" in col:
            metric = col.split("_mean")[0]
            mean_col =col            
            std_col = f"{metric}_std"

            # 创建新的列格式 "指标名: mean +/- std"
            results[metric] = df.apply(
                lambda row: f"{row[mean_col]:.3f}±{row[std_col]:.3f}", 
                axis=1
            )

    # 保存带有 mean 和 std 列的 DataFrame
    df.to_excel(f"{filename}.xlsx", index=False)

    # 保存带有合并 mean 和 std 的 DataFrame
    results.to_excel(f"{filename}_MeanStd.xlsx", index=False)


def main(mode="train"):
    setup_seed(42)

    # depression: 0.86
    storage_name = "postgresql://postgres:123...@127.0.0.1/depression"
    # study_name = "EgvAttNet"
    study_name = 'EgvAttNet_cross_validation_auc'

    def optimize(n_trials):
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        study.optimize(objective, n_trials=n_trials)

    if mode == "train":
        try:
            optuna.load_study(study_name=study_name, storage=storage_name)
            optuna.delete_study(study_name=study_name, storage=storage_name)
            print(f"Deleted existing study: {study_name}")
        except KeyError:
            print(f"Study {study_name} does not exist, creating new one.")
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
        # study = optuna.load_study(study_name=study_name, storage=storage_name)
        # study.optimize(objective, n_trials=5, n_jobs=5, show_progress_bar=True)
        Parallel(n_jobs=10)([delayed(optimize)(50) for _ in range(20)])

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value}")
        file_handler.close()
    elif mode == "test":
        # best_params = get_best_para_from_optuna(study_name=study_name, storage_name=storage_name)
        metrics_summary = train_and_evaluate_model()
        # 调用示例
        save_metrics_summary_to_excel(metrics_summary, "metrics_summary_test")
    else:
        NotImplementedError


if __name__=="__main__":
    setup_seed(42)
    import argparse
    parser = argparse.ArgumentParser(description="Run the model with optional tuning.")
    
    # 2. 添加 --tuning 参数（类型为 int，默认 0）
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Enable tuning mode (train or test)."
    )
    args = parser.parse_args()
    # main(mode=args.mode)
    main(mode="test")



