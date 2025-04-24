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
from tools.utils import get_eGe_matrix,get_vggish_features
from tools.model import EGV_AttNet
from joblib import Parallel, delayed
import optuna
import warnings
warnings.filterwarnings('ignore')

setup_seed(42)
LOGFILE = f"../result/02optuna/depression_optuna_f1.log"
logger,file_handler = init_logger(LOGFILE)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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

with open("../result/01preprocess/eGe_feature_cumul0.99.txt",'r')as f:
    eGe_feature = [line.strip() for line in f]

feature_weights = pd.read_csv('../result/01preprocess/03sorted_feature_importance.csv',index_col=0)
feature_weights = feature_weights.squeeze()
feature_weights = feature_weights[eGe_feature]

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(shuffled_df['class']), y=shuffled_df['class'])
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)


def train_evaluate(model, criterion, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs=50, patience=10, save_flag=False):
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
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    X_test_eGe = torch.tensor(X_test_eGe, dtype=torch.float32).to(device)
    X_test_VGGish = torch.tensor(X_test_VGGish, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_eGe, X_train_VGGish)  # 分别传入 eGeMAPS 和 VGGish 特征
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_eGe, X_test_VGGish)  # 分别传入 eGeMAPS 和 VGGish 特征
            predictions_prob = val_outputs[:, 1]  # 预测概率          
            y_pred = (predictions_prob >= 0.5).int().cpu().numpy() # 预测标签
            y_pred_prob = predictions_prob.cpu().numpy()         
            evaluta_dic = evaluate_model(y_test.cpu().numpy(), y_pred, y_pred_prob)    
        
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
    torch.save(model.state_dict(), "../result/02optuna/best_EgvAtt_model.pth")  # 保存当前最佳模型
    return evaluta_dic['roc_auc'], evaluta_dic['f1']


def train_and_evaluate_model(model_param):
    setup_seed(42)
    print(f"Training EGVAttNet...")
    lr = model_param['lr']
    # lr = 2e-3
    transformed_feature_dim = model_param['transformed_feature_dim']
    # transformed_feature_dim = 64
    weight_decay = model_param['weight_decay']
    # weight_decay = 1.5e-4
    T_0_value = 14
    T_mult_value = 1
    # epoch = model_param['epoch']
    epoch = 200
    # patience = model_param['patience']
    patience = 30
    conv_out_channels = model_param['conv_out_channels']
    # conv_out_channels = 256
    eca_kernel_size = 3
    resblock_kernel_size = 3
    num_conv_layers = 1
    class_weight_flag = True

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []  # 用于存储插值后的TPR

    metrics_all = {
        "acc": [], "ppv": [], "sensitivity": [], "f1": [],
        "roc_auc": [], "mcc": [], "aupr": [], "gmean": [],
        "npv": [], "kappa": [], "specificity": [],
    }
    auc_list = []

    data_path = "../data/enhance_10/"
    X_train_eGe_list,X_test_eGe_list,X_train_VGGish_list,X_test_VGGish_list,y_train_list,y_test_list = read_data(data_path)
    # print(f"data item len： {len(X_train_eGe_list)}")
    for i in range(len(X_train_eGe_list)):
        
        # 获取预先提取的特征
        X_train_eGe = X_train_eGe_list[i]
        X_test_eGe = X_test_eGe_list[i]
        X_train_VGGish = X_train_VGGish_list[i]
        X_test_VGGish = X_test_VGGish_list[i]
        y_train = y_train_list[i]
        y_test = y_test_list[i]

        X_train = (X_train_eGe[eGe_feature], X_train_VGGish)
        X_test = (X_test_eGe[eGe_feature], X_test_VGGish)
        
        model = EGV_AttNet(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, num_classes=2, feature_weights=feature_weights, 
                        num_conv_layers=num_conv_layers, conv_out_channels=conv_out_channels,
                        eca_kernel_size=eca_kernel_size, transformed_feature_dim=transformed_feature_dim, resblock_kernel_size=resblock_kernel_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0_value, T_mult=T_mult_value)
        if class_weight_flag:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        #训练并评估模型
        train_evaluate(model, criterion, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs=epoch, patience=patience, save_flag=True)

        model.load_state_dict(torch.load("../result/02optuna/best_EgvAtt_model.pth"))
        model.eval()
        with torch.no_grad():

            X_test_eGe = np.asarray(X_test_eGe[eGe_feature])
            X_test_VGGish = np.asarray(X_test_VGGish)

            # 将特征转换为张量
            X_test_eGe = torch.tensor(X_test_eGe, dtype=torch.float32).to(device)
            X_test_VGGish = torch.tensor(X_test_VGGish, dtype=torch.float32).to(device)
            y_test = torch.tensor(y_test, dtype=torch.long).to(device)

            outputs = model(X_test_eGe, X_test_VGGish)
            predictions_prob = outputs[:, 1]
            y_pred = (predictions_prob >= 0.5).int().cpu().numpy()
            y_pred_prob = predictions_prob.cpu().numpy()         

        # 评估
        metrics = evaluate_model(y_test.cpu().numpy(), y_pred, y_pred_prob)
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

    return metrics_summary



def objective(trial):
    setup_seed(42)
    # train param
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    # lr = 2e-3
    transformed_feature_dim = trial.suggest_int('transformed_feature_dim', 50, 256)
    # transformed_feature_dim = 64
    weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-4, log=True)
    # weight_decay = 1.5e-4
    T_0_value = 14  # CosineAnnealingWarmRestarts中的 T_0
    T_mult_value = 1
    # epoch = trial.suggest_int("epoch", 50, 300, log=True)
    epoch = 200
    # patience = trial.suggest_int("patience", 30, 110, log=True)
    patience = 30

    conv_out_channels = trial.suggest_int('conv_out_channels', 64, 256)
    # conv_out_channels = 256
    # eca_kernel_size = trial.suggest_int('eca_kernel_size', 1, 7)
    eca_kernel_size = 3
    # resblock_kernel_size = trial.suggest_int('resblock_kernel_size', 1, 7)
    resblock_kernel_size = 3

    # model param
    num_conv_layers = 1
    # num_conv_layers = trial.suggest_int('num_conv_layers', 1, 2)

    # extra param
    class_weight_flag = True

    data_path = "../data/enhance/"
    X_train_eGe_list,X_test_eGe_list,X_train_VGGish_list,X_test_VGGish_list,y_train_list,y_test_list = read_data(data_path)
    f1_lst = []
    auc_lst = []
    # print(f"data item len： {len(X_train_eGe_list)}")
    
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
        model = EGV_AttNet(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, num_classes=2, feature_weights=feature_weights, 
                           num_conv_layers=num_conv_layers, conv_out_channels=conv_out_channels,
                           eca_kernel_size=eca_kernel_size, transformed_feature_dim=transformed_feature_dim, resblock_kernel_size=resblock_kernel_size).to(device)

        # for name, param in model.named_parameters():
        #     if name == "eGeMAPS_transform.weight":
        #         print(f"Layer: {name} | Shape: {param.shape} | Values (first 5): {param.data.flatten()[:5]}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0_value, T_mult=T_mult_value)
        if class_weight_flag:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        #训练并评估模型
        auc,f1 = train_evaluate(model, criterion, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs=epoch, patience=patience)
        f1_lst.append(f1)
        auc_lst.append(auc)

    return np.mean(auc_lst),np.mean(f1_lst)

# print("\nInitialized parameters:")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Shape: {param.shape} | Values (first 5): {param.data.flatten()[:5]}")

def get_best_para_from_optuna(study_name, storage_name):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_trials[2].params
    # best_params = study.best_params
    print(study_name,best_params)
    # print(f"best value: {study.best_value}")
    print(f"best value: {study.best_trials[2].values}")
    return best_params

def main(mode="train"):
    setup_seed(42)

    # depression: 0.86
    storage_name = "postgresql://postgres:123...@127.0.0.1/depression_multi"
    study_name = "EgvAttNet"

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
        study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["maximize", "maximize"])
        # study = optuna.load_study(study_name=study_name, storage=storage_name)
        # study.optimize(objective, n_trials=5, n_jobs=5, show_progress_bar=True)
        Parallel(n_jobs=10)([delayed(optimize)(50) for _ in range(10)])

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value}")
        file_handler.close()
    elif mode == "test":
        best_params = get_best_para_from_optuna(study_name=study_name, storage_name=storage_name)
        metrics_summary = train_and_evaluate_model(best_params)
        print(metrics_summary)
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
    main(mode=args.mode)



