import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import gmean
from dataprocess import get_feature_paths_and_labels, AudioFeaturesDataset, pad_collate
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from utils import setup_seed
# 收集数据
def collect_data(data_loader, device):
    all_X = []
    all_y = []
    max_length = 128 

    for eGeMAPS_feature, VGGish_feature, label in data_loader:
        eGeMAPS_feature, VGGish_feature, label = eGeMAPS_feature.to(device), VGGish_feature.to(device), label.to(device)
        for e, v, l in zip(eGeMAPS_feature, VGGish_feature, label):
            v_flat = v.flatten()
            e_flat = e.flatten()
            combined_feature = torch.cat((v_flat, e_flat))
            all_X.append(combined_feature.cpu().numpy())
            all_y.append(l.item())

    all_X_padded = []
    for x in all_X:
        if len(x) > max_length:
            x_padded = x[:max_length]
        else:
            x_padded = np.pad(x, (0, max_length - len(x)), mode='constant')
        all_X_padded.append(x_padded)
    all_X = np.array(all_X_padded)
    all_y = np.array(all_y)

    return all_X, all_y

# 比较传统机器学习模型
def compare_models(config):
    setup_seed(42)
    eGeMAPS_features_paths, VGGish_features_paths, labels = get_feature_paths_and_labels(config["features_save_dir"])

    skf = StratifiedKFold(n_splits=config["num_folds"], shuffle=True)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "Bagging": BaggingClassifier(),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
        "XGBoost": xgb.XGBClassifier(),
        "LightGBM": LGBMClassifier(verbosity = -1),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "GBDT": GradientBoostingClassifier()
    }

    results = []
    model_roc_data = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', model)])
        roc_data = []
        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "aupr": [],
            "mcc": [],
            "balanced_acc": [],
            "kappa": [],
            "gmean": []
        }
        for train_indices, val_indices in skf.split(np.arange(len(labels)), labels):
            train_dataset = AudioFeaturesDataset(
                [eGeMAPS_features_paths[i] for i in train_indices],
                [VGGish_features_paths[i] for i in train_indices],
                [labels[i] for i in train_indices]
            )
            val_dataset = AudioFeaturesDataset(
                [eGeMAPS_features_paths[i] for i in val_indices],
                [VGGish_features_paths[i] for i in val_indices],
                [labels[i] for i in val_indices]
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

            all_X_train, all_y_train = collect_data(train_loader, config["device"])
            all_X_val, all_y_val = collect_data(val_loader, config["device"])
            pipeline.fit(all_X_train, all_y_train)

            y_pred = pipeline.predict(all_X_val)
            y_prob_positive = pipeline.predict_proba(all_X_val)[:, 1]
            
            fpr, tpr, _ = roc_curve(all_y_val, y_prob_positive)
            
            roc_auc = auc(fpr, tpr)
            roc_data.append({'fpr':fpr,'tpr':tpr,'auc':roc_auc})

        
           
            metrics["accuracy"].append(accuracy_score(all_y_val, y_pred))
            metrics["precision"].append(precision_score(all_y_val, y_pred, average='macro'))
            metrics["recall"].append(recall_score(all_y_val, y_pred, average='macro'))
            metrics["f1"].append(f1_score(all_y_val, y_pred, average='macro'))
            metrics["aupr"].append(average_precision_score(all_y_val, y_prob_positive))
            metrics["mcc"].append(matthews_corrcoef(all_y_val, y_pred))
            metrics["balanced_acc"].append(np.mean(recall_score(all_y_val, y_pred, average=None)))
            metrics["kappa"].append(cohen_kappa_score(all_y_val, y_pred))
            metrics["gmean"].append(gmean([metrics["precision"][-1], metrics["recall"][-1]]))
        
        #将当前模型的所有折的 ROC 数据存储到主字典中
        model_roc_data[name] = roc_data
        results.append({
            "Model": name,
            **{f"{metric.capitalize()} Mean": f"{np.mean(scores):.4f} ± {np.std(scores):.4f}" for metric, scores in metrics.items()}
        })
        print(f"{name} - " + ", ".join([f"{metric.capitalize()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}" for metric, scores in metrics.items()]))
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(r"E:\抑郁症3\结果\3_组间对照深度学习\2_对比实验\输出文件\comparison_results.csv", index=False, float_format='%.4f')
    print("Comparison results saved to comparison_results.csv")
    return model_roc_data
