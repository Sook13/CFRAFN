import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
from tools.evaluate import evaluate_model
from tools.common import setup_seed
from tools.utils import get_best_para_from_optuna
# from tools.model_simplecombine import CFRAFN
# from tools.model_vgg_resnet import CFRAFN
# from tools.model_eGe_resnet import CFRAFN
from tools.model_resnet import CFRAFN
# from tools.model import CFRAFN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc
import seaborn as sns
import random

setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
feature_weights = feature_weights.squeeze()[eGe_feature]

def train_evaluate(model, criterion, optimizer, X_train, y_train, X_test, y_test, 
                   sample_weight=0.5, epochs=50, patience=10, save_flag=False):
    setup_seed(42)
    best_f1 = 0.0
    no_improvement_count = 0

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
        selected_indices = [random.choices(group, weights=[sample_weight] + 
                                           [(1-sample_weight)/(len(group)-1)]*(len(group)-1), k=1)[0] for group in groups]

        # 获取选中的数据
        X_train_eGe_select = X_train_eGe[selected_indices]
        X_train_VGGish_select = X_train_VGGish[selected_indices]
        y_train_select = y_train[selected_indices]

        outputs = model(X_train_eGe_select, X_train_VGGish_select).squeeze()
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
        
        # 早停检测
        if evaluta_dic['f1'] > best_f1:
            best_f1 = evaluta_dic['f1']
            no_improvement_count = 0  # 重置计数器
            if save_flag:
                torch.save(model.state_dict(), "../result/04modelPerformance/best_CFRAFN622.pth")  # 保存当前最佳模型
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            break

    return best_f1

def evaluate_test_model(model_param=None):
    setup_seed(42)
    model_param =  {'lr': 0.00012365438684898807, 'conv_out_channels': 240, 'transformed_feature_dim': 461, 'cls_dim': 156, 'sample_weight': 0.9}

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
    patience = 100

    data_path = "../data/modelBasicPerformance/"
    X_train_eGe,X_test_eGe,X_val_eGe,X_train_VGGish,X_test_VGGish,X_val_VGGish,y_train,y_test,y_val = read_data(data_path)

    X_train = (X_train_eGe[eGe_feature], X_train_VGGish)
    # X_test = (X_test_eGe[eGe_feature], X_test_VGGish)
    X_val = (X_val_eGe[eGe_feature], X_val_VGGish)

    assert feature_weights.index.tolist() == X_train[0].columns.tolist(), "feature_weights order must be equal to the order of data columns"
    
    model = CFRAFN(input_dim_eGeMAPS=len(feature_weights), input_dim_VGGish=128, 
                       feature_weights=feature_weights, expansion=expansion, dropout=dropout,
                       conv_out_channels=conv_out_channels,
                       transformed_feature_dim=transformed_feature_dim, 
                       resblock_kernel_size=resblock_kernel_size, cls_dim=cls_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
        
    #训练并评估模型
    train_evaluate(model, criterion, optimizer, X_train, y_train, X_val, y_val, 
                   sample_weight=sample_weight, epochs=epoch, patience=patience, 
                   save_flag=True)
    
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
    
    def _plot_roc_curve(results):
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr_all = []
        plt.figure(figsize=(10, 8))

        colors = sns.color_palette("Set1",2)
        
        n = 0
        for evaluate_dic in results:
            fpr, tpr = evaluate_dic["fpr"], evaluate_dic["tpr"]
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # 确保起点为0
            interp_tpr_all.append(interp_tpr)
            
            mean_tpr = np.mean(interp_tpr_all, axis=0)
            mean_tpr[-1] = 1.0  # 确保终点为1
            # mean_auc = auc(mean_fpr, mean_tpr)
            
            plt.plot(mean_fpr, mean_tpr, color=colors[n], lw=2, label=f"{evaluate_dic['Dataset']} Dataset (AUC = {evaluate_dic['roc_auc']:.3f})")
            n+=1

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
        plt.legend(loc="lower right", prop={'weight':'bold', 'size':12}, labelcolor='black', edgecolor='black')
        # plt.legend(loc="lower right")
        plt.savefig(os.path.join("../result/04modelPerformance/experiment1_model_performance.tif"),dpi=300,bbox_inches="tight")
        plt.savefig(os.path.join("../result/04modelPerformance/experiment1_model_performance.pdf"),dpi=600,bbox_inches="tight")
    
    def _save_results_to_excel(results, filename):
        for evaluate_dic in results:
            del evaluate_dic['fpr']
            del evaluate_dic['tpr']

        df = pd.DataFrame(results)
        cols = ['Dataset', 'f1','roc_auc','aupr','gmean','kappa','mcc','acc','npv','ppv','sensitivity','specificity']
        df = df[cols]
        df.to_excel(filename+".xlsx", index=False)

    def _get_evaluate_dic(eGe, vgg, y_true, dataset_name):
        ### metrics指标 ###
        outputs = model(eGe, vgg)
        predictions_prob = outputs
        y_pred = (predictions_prob >= 0.5).int().cpu().numpy()
        y_pred_prob = predictions_prob.cpu().numpy()         
        # 评估
        evaluate_dic = evaluate_model(y_true.cpu().numpy(), y_pred, y_pred_prob) 
    
        print(f"{dataset_name} dataset:\tF1: {evaluate_dic['f1']:.4f}\tAUC: {evaluate_dic['roc_auc']:.4f}\tACC: {evaluate_dic['acc']:.4f}")
        _plot_confusion_matrix(y_true.cpu().numpy(), y_pred, 2, dataset_name)
        evaluate_dic['Dataset'] = dataset_name
        return evaluate_dic

    model.load_state_dict(torch.load("../result/04modelPerformance/best_CFRAFN622.pth"))
    model.eval()
    with torch.no_grad():

        X_test_eGe = np.asarray(X_test_eGe[eGe_feature])
        X_val_eGe = np.asarray(X_val_eGe[eGe_feature])

        X_test_VGGish = np.asarray(X_test_VGGish)
        X_val_VGGish = np.asarray(X_val_VGGish)

        # 将特征转换为张量
        X_test_eGe = torch.tensor(X_test_eGe, dtype=torch.float32).to(device)
        X_test_VGGish = torch.tensor(X_test_VGGish, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        X_val_eGe = torch.tensor(X_val_eGe, dtype=torch.float32).to(device)
        X_val_VGGish = torch.tensor(X_val_VGGish, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        save_path = '../result/04modelPerformance/'
        evaluate_dic_val = _get_evaluate_dic(X_val_eGe, X_val_VGGish, y_val, "Validation")
        evaluate_dic_test = _get_evaluate_dic(X_test_eGe, X_test_VGGish, y_test, "Test")

        final_result = [evaluate_dic_val, evaluate_dic_test]
        _plot_roc_curve(final_result)
        _save_results_to_excel(final_result,os.path.join(save_path,"CFRAFN_performance"))

def main():
    setup_seed(42)
    # storage_name = "postgresql://Sy:qwe123@127.0.0.1:5432/depression"
    # study_name = "EgvAttNet"
    # best_params = get_best_para_from_optuna(study_name=study_name, storage_name=storage_name)
    evaluate_test_model()

if __name__=="__main__":
    main()