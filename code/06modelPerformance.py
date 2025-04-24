import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import joblib
from torchvggish import vggish, vggish_input
from joblib import Parallel, delayed, dump
from sklearn.preprocessing import StandardScaler
from tools.evaluate import evaluate_model, plot_roc_curve, overall_evaluate_plot, calculate_mean_std_metrics
from tools.common import setup_seed,init_logger
from tools.utils import get_eGe_matrix,get_vggish_features,get_best_para_from_optuna
from tools.model import EGV_AttNet
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import confusion_matrix
import seaborn as sns
import swanlab
from swanlab.plugin.notification import WXWorkCallback
import random
from collections import defaultdict

setup_seed(42)

LOGFILE = f"../result/02optuna/depression_model_performance622.log"
logger,file_handler = init_logger(LOGFILE)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("../data/group_control.csv")
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = shuffled_df['name']
y = shuffled_df['class']
self_folder = "../data/group_control/"

def read_data(save_path):
    X_train_eGe_path = os.path.join(save_path,"X_train_eGe.joblib")
    X_test_eGe_path = os.path.join(save_path,"X_test_eGe.joblib")
    X_val_eGe_path = os.path.join(save_path,"X_val_eGe.joblib")
    X_train_VGGish_path = os.path.join(save_path,"X_train_VGGish.joblib")
    X_test_VGGish_path = os.path.join(save_path,"X_test_VGGish.joblib")
    X_val_VGGish_path = os.path.join(save_path,"X_val_VGGish.joblib")
    y_train_path = os.path.join(save_path,"y_train.joblib")
    y_test_path = os.path.join(save_path,"y_test.joblib")
    y_val_path = os.path.join(save_path,"y_val.joblib")

    X_train_eGe = joblib.load(X_train_eGe_path)
    X_test_eGe = joblib.load(X_test_eGe_path)
    X_val_eGe = joblib.load(X_val_eGe_path)
    X_train_VGGish = joblib.load(X_train_VGGish_path) 
    X_test_VGGish = joblib.load(X_test_VGGish_path)
    X_val_VGGish = joblib.load(X_val_VGGish_path)
    y_train = joblib.load(y_train_path)
    y_test = joblib.load(y_test_path)
    y_val = joblib.load(y_val_path)
    return X_train_eGe,X_test_eGe,X_val_eGe,X_train_VGGish,X_test_VGGish,X_val_VGGish,y_train,y_test,y_val


with open("../result/01preprocess/eGe_feature_cumul0.99.txt",'r')as f:
    eGe_feature = [line.strip() for line in f]

feature_weights = pd.read_csv('../result/01preprocess/03sorted_feature_importance.csv',index_col=0)
feature_weights = feature_weights.squeeze()
feature_weights = feature_weights[eGe_feature]

labels = shuffled_df['class']
classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
print(labels.sum())
# class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
class_weights = torch.tensor(np.array([100,1]), dtype=torch.float, device=device)

def train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                   sample_weight=0.5, epochs=50, patience=10, save_flag=False):
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
        # # 初始化选中索引列表（保留所有原始样本）
        # selected_indices = list(range(0, num_samples, 6))  # 所有组的第1个样本（原始数据）

        # # 为每组随机选1个增强样本
        # for group_start in range(0, num_samples, 6):
        #     # 随机从该组的增强样本（1-5）中选1个
        #     augment_index = random.randint(group_start + 1, group_start + 5)
        #     selected_indices.append(augment_index)
        
        selected_indices = [random.choices(group, weights=[sample_weight] + [(1-sample_weight)/(len(group)-1)]*(len(group)-1), k=1)[0] for group in groups]

        # 获取选中的数据
        X_train_eGe_select = X_train_eGe[selected_indices]
        X_train_VGGish_select = X_train_VGGish[selected_indices]
        y_train_select = y_train[selected_indices]

        
        # 随机取

        # outputs = model(X_train_eGe, X_train_VGGish).squeeze()  # 分别传入 eGeMAPS 和 VGGish 特征
        outputs = model(X_train_eGe_select, X_train_VGGish_select).squeeze() 
        # loss = criterion(outputs, y_train)
        loss = criterion(outputs, y_train_select)
        loss.backward()
        optimizer.step()
        swanlab.log({"train/loss": loss}, step=epoch)

        # 评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_eGe, X_test_VGGish).squeeze()  # 分别传入 eGeMAPS 和 VGGish 特征
            loss_test = criterion(val_outputs, y_test) 
            predictions_prob = val_outputs # 预测概率       
            y_pred = (predictions_prob >= 0.5).int().cpu().numpy() # 预测标签
            y_pred_prob = predictions_prob.cpu().numpy()         
            evaluta_dic = evaluate_model(y_test.cpu().numpy(), y_pred, y_pred_prob)
            swanlab.log({"test/loss": loss_test})
            swanlab.log({"test/f1": evaluta_dic['f1']})
            swanlab.log({"test/auc": evaluta_dic['roc_auc']})
            swanlab.log({"test/acc": evaluta_dic['acc']})


        # print(f"{epoch} ==> loss: {loss:.4f}    f1; {evaluta_dic['f1']:.4f}")
        
        # 早停检测
        if evaluta_dic['f1'] > best_f1:
            best_f1 = evaluta_dic['f1']
            best_auc = evaluta_dic['roc_auc']
            no_improvement_count = 0  # 重置计数器
            swanlab.log({"test/early_stop_epoch": epoch})
            swanlab.log({"test/early_stop_f1": best_f1})
            swanlab.log({"test/early_stop_auc": best_auc})
            if save_flag:
                torch.save(model.state_dict(), "../result/04modelPerformance/best_EgvAtt_model.pth")  # 保存当前最佳模型
        else:
            no_improvement_count += 1

        # 如果没有提升的轮数超过阈值，停止训练
        if no_improvement_count >= patience:
            # print(f"Early stopping triggered. Best F1: {best_f1:.4f}")
            break
    # torch.save(model.state_dict(), "../result/04modelPerformance/best_EgvAtt_model.pth")
    return best_f1

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 平衡类别权重
        self.gamma = gamma  # 聚焦难样本的参数

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # 模型对正确分类的置信度
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()




def evaluate_test_model(model_param=None):
    setup_seed(42)
    # lr = model_param['lr']
    # transformed_feature_dim = model_param['transformed_feature_dim']
    # weight_decay = model_param['weight_decay']
    # conv_out_channels = model_param['conv_out_channels']
    # epoch = model_param['epoch']
    # patience = model_param['patience']
    lr = model_param['lr']
    sample_weight = model_param['sample_weight']
    conv_out_channels = model_param['conv_out_channels']
    transformed_feature_dim = model_param['transformed_feature_dim']
    cls_dim = model_param['cls_dim']
    weight_decay = 2e-4
    expansion = 2
    dropout = 0.1
    resblock_kernel_size = 5
    epoch = 2000
    patience = 200


    # swanlab.config = {
    #     "lr": lr,
    #     "transformed_feature_dim": transformed_feature_dim,
    #     "weight_decay": weight_decay,
    #     "epoch": epoch,
    #     "patience": patience,
    #     "conv_out_channels": conv_out_channels,
    #     "eca_kernel_size": eca_kernel_size,
    #     "resblock_kernel_size": resblock_kernel_size,
    # }

    data_path = "../data/modelBasicPerformance/dropvoice/"  # 没有做归一并且去除了音量增强
    X_train_eGe,X_test_eGe,X_val_eGe,X_train_VGGish,X_test_VGGish,X_val_VGGish,y_train,y_test,y_val = read_data(data_path)

    X_train = (X_train_eGe[eGe_feature], X_train_VGGish)
    X_test = (X_test_eGe[eGe_feature], X_test_VGGish)

    assert feature_weights.index.tolist() == X_train[0].columns.tolist(), "feature_weights order must be equal to the order of data columns"
    
    model = EGV_AttNet(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, 
                       feature_weights=feature_weights, expansion=expansion, dropout=dropout,
                       conv_out_channels=conv_out_channels,
                       transformed_feature_dim=transformed_feature_dim, 
                       resblock_kernel_size=resblock_kernel_size, cls_dim=cls_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # if criterion == "bce":
    #     criterion = nn.BCELoss()
    # elif criterion == 'focal':
    #     focal_alpha = model_param['focal_alpha']
    #     criterion = FocalLoss(alpha=focal_alpha)
    # else:
    #     NotImplementedError
        
    #训练并评估模型
    f1 = train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                        sample_weight=sample_weight, epochs=epoch, patience=patience, 
                        save_flag=True)
    
    # print(f"best f1: {f1}")
    def _plot_confusion_matrix(y_true, y_pred, classes, dataset_name, cmap=plt.cm.Blues):
        plt.rcParams['font.family'] = 'Arial'
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        plt.savefig(f'../result/04modelPerformance/01confusion_matrix_{dataset_name}.tif', dpi=300, bbox_inches="tight")
        plt.savefig(f'../result/04modelPerformance/01confusion_matrix_{dataset_name}.pdf', dpi=300, bbox_inches="tight")
        
        # plt.show()

    def _get_evaluate_dic(eGe, vgg, y_true, dataset_name, save_path):
        outputs = model(eGe, vgg)
        predictions_prob = outputs
        y_pred = (predictions_prob >= 0.5).int().cpu().numpy()
        y_pred_prob = predictions_prob.cpu().numpy()         
        evaluate_dic = evaluate_model(y_true.cpu().numpy(), y_pred, y_pred_prob)
        print(f"{dataset_name} dataset:\tF1: {evaluate_dic['f1']:.4f}\tAUC: {evaluate_dic['roc_auc']:.4f}\tACC: {evaluate_dic['acc']:.4f}")
        _plot_confusion_matrix(y_true.cpu().numpy(), y_pred, 2, dataset_name)

        del evaluate_dic['fpr']
        del evaluate_dic['tpr']
        df = pd.DataFrame(evaluate_dic, index=[0])
        
        cols = ['f1','roc_auc','aupr','gmean','kappa','mcc','acc','npv','ppv','sensitivity','specificity']
        df = df[cols]
        df.to_excel(os.path.join(save_path, f"{dataset_name}_evaluate.xlsx"), index=False)



    model.load_state_dict(torch.load("../result/04modelPerformance/best_EgvAtt_model.pth"))
    model.eval()
    with torch.no_grad():

        X_test_eGe = np.asarray(X_test_eGe[eGe_feature])
        X_val_eGe = np.asarray(X_val_eGe[eGe_feature])

        X_test_VGGish = np.asarray(X_test_VGGish)
        X_val_VGGish = np.asarray(X_val_VGGish)
        # print(f"y_test: {y_test}")
        # print(f"y_val: {y_val}")
        # 将特征转换为张量
        X_test_eGe = torch.tensor(X_test_eGe, dtype=torch.float32).to(device)
        X_test_VGGish = torch.tensor(X_test_VGGish, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        X_val_eGe = torch.tensor(X_val_eGe, dtype=torch.float32).to(device)
        X_val_VGGish = torch.tensor(X_val_VGGish, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        save_path = '../result/04modelPerformance/'
        _get_evaluate_dic(X_test_eGe, X_test_VGGish, y_test, "val", save_path)
        _get_evaluate_dic(X_val_eGe, X_val_VGGish, y_val, "test", save_path)


def objective(trial):
    setup_seed(42)
    # train param

    # weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-4, log=True)
    # expansion = trial.suggest_categorical('expansion', [1, 2, 4, 8])
    # dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # resblock_kernel_size = trial.suggest_categorical('resblock_kernel_size', [3, 5, 7])
    # criterion = trial.suggest_categorical('criterion', ['bce', 'focal'])
    # scheduler_flag = trial.suggest_categorical('scheduler_flag', [True, False])

    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)  # 2e-4
    conv_out_channels = trial.suggest_int('conv_out_channels', 32, 256)
    transformed_feature_dim = trial.suggest_int('transformed_feature_dim', 64, 512)
    cls_dim = trial.suggest_int('cls_dim', 64, 256)
    sample_weight = trial.suggest_float('sample_weight', 0.01, 0.99)
    
    weight_decay = 2e-4
    expansion = 2
    dropout = 0.1
    resblock_kernel_size = 5

    epoch = 2000
    patience = 200

    data_path = "../data/modelBasicPerformance/dropvoice/"
    X_train_eGe,X_test_eGe,X_val_eGe,X_train_VGGish,X_test_VGGish,X_val_VGGish,y_train,y_test,y_val = read_data(data_path)

    X_train = (X_train_eGe[eGe_feature], X_train_VGGish)
    X_test = (X_test_eGe[eGe_feature], X_test_VGGish)

    assert feature_weights.index.tolist() == X_train[0].columns.tolist(), "feature_weights order must be equal to the order of data columns"
    
    model = EGV_AttNet(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, 
                       feature_weights=feature_weights, expansion=expansion, dropout=dropout,
                       conv_out_channels=conv_out_channels,
                       transformed_feature_dim=transformed_feature_dim, 
                       resblock_kernel_size=resblock_kernel_size, cls_dim=cls_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # if criterion == "bce":
    #     criterion = nn.BCELoss()
    # elif criterion == 'focal':
    #     focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.9)
    #     criterion = FocalLoss(alpha=focal_alpha)
    # else:
    #     NotImplementedError
        
    #训练并评估模型
    f1 = train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                        sample_weight=sample_weight, epochs=epoch, patience=patience)

    return f1



def main(mode="train"):

    setup_seed(42)
    storage_name = "postgresql://postgres:123...@127.0.0.1/depression"
    # EgvAttNet
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
        study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=optuna.samplers.TPESampler(seed=42), direction='maximize')
        # study = optuna.load_study(study_name=study_name, storage=storage_name)
        # study.optimize(objective, n_trials=10, n_jobs=10, show_progress_bar=True)
        Parallel(n_jobs=10)([delayed(optimize)(30) for _ in range(200)])

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value}")
        file_handler.close()
    elif mode == "test":
        best_params = get_best_para_from_optuna(study_name=study_name, storage_name=storage_name)
        evaluate_test_model(best_params)
        # evaluate_test_model()
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

    # # wxwork_callback = WXWorkCallback(
    #     webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=15a5c31b-8d22-433c-9e11-f815793ac58b",
    # )

    run = swanlab.init(
        project="depression",
        experiment_name="ATT 随机加权 optuna 6000",
        description="不进行标准化且去除音量增强。改为二元交叉熵损失BCEloss  设置小学习率和大epoch",
        mode='disabled', # 调试模式
        # callbacks=[wxwork_callback]
    )

    main(mode=args.mode)

# python .\06modelPerformance.py --mode train

