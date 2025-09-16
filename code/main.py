import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tools.evaluate import evaluate_model
from tools.common import setup_seed
from sklearn.model_selection import train_test_split
from tools.model import CFRAFN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict
from tools.utils import get_eGe_matrix,get_vggish_features
from scipy.interpolate import interp1d
import argparse 
 
setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_csv(root_folder):
    cls_files = []
    for root, _, files in os.walk(root_folder):
        for filename in files:
            cls_files.append((filename, os.path.basename(root).split("_")[-1]))

    cls_files_df = pd.DataFrame(cls_files, columns=['name', 'class'])
    cls_files_df.to_csv(os.path.join(os.path.dirname(root_folder),os.path.basename(root_folder)+".csv"),index=False,encoding="utf-8-sig")

root_folder = '../data/demo_data'

get_data_csv(root_folder)

with open("../result/01preprocess/eGe_feature_cumul1.txt",'r')as f:
    eGe_feature = [line.strip() for line in f]

feature_weights = pd.read_csv('../result/01preprocess/sorted_feature_importance.csv',index_col=0)
feature_weights = feature_weights.squeeze()[eGe_feature]

def read_data(data_path):
    df = pd.read_csv("../data/demo_data.csv")
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    X = shuffled_df['name']
    y = shuffled_df['class']
    self_folder = data_path
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
    
    train_index = X.index[X.isin(X_train)].tolist()
    val_index = X.index[X.isin(X_val)].tolist()
    test_index = X.index[X.isin(X_test)].tolist()

    X_train_eGe, y_train, _, _ = get_eGe_matrix(train_index, self_folder, X, y, n_jobs=-1)
    X_val_eGe, y_val, _, _ = get_eGe_matrix(val_index, self_folder, X, y, n_jobs=-1)
    X_test_eGe, y_test, _, _ = get_eGe_matrix(test_index, self_folder, X, y, n_jobs=-1)

    X_train_VGGish, _, _, _ = get_vggish_features(train_index, self_folder, X, y, n_jobs=-1)
    X_val_VGGish, _, _, _ = get_vggish_features(val_index, self_folder, X, y, n_jobs=-1)
    X_test_VGGish, _, _, _ = get_vggish_features(test_index, self_folder, X, y, n_jobs=-1)
    
    data = {
        'X_train_eGe': X_train_eGe,
        'X_test_eGe': X_test_eGe,
        'X_val_eGe': X_val_eGe,
        'X_train_VGGish': X_train_VGGish,
        'X_test_VGGish': X_test_VGGish,
        'X_val_VGGish': X_val_VGGish,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val
    }
    
    return (
        data['X_train_eGe'], data['X_test_eGe'], data['X_val_eGe'],
        data['X_train_VGGish'], data['X_test_VGGish'], data['X_val_VGGish'],
        data['y_train'], data['y_test'], data['y_val']
    )
 
def init_model(params):
    """Initialize the CFRAFN model with given parameters"""
    return CFRAFN(
        input_dim_eGeMAPS=len(feature_weights),
        input_dim_VGGish=128,
        feature_weights=feature_weights,
        **{k: params[k] for k in ['expansion', 'dropout', 'conv_out_channels',
                                 'transformed_feature_dim', 'resblock_kernel_size', 'cls_dim']}
    ).to(device)

def prepare_datasets(
    X_train_eGe: pd.DataFrame,
    X_val_eGe: pd.DataFrame,
    X_test_eGe: pd.DataFrame,
    X_train_VGGish: pd.DataFrame,
    X_val_VGGish: pd.DataFrame,
    X_test_VGGish: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict[str, Tuple]:
    """Prepare datasets for training, validation and testing"""
    def _prepare(X_eGe, X_VGGish, y):
        X_eGe = np.asarray(X_eGe)
        X_VGGish = np.asarray(X_VGGish)
        y = np.asarray(y)

        return torch.tensor(X_eGe, dtype=torch.float32).to(device), \
               torch.tensor(X_VGGish, dtype=torch.float32).to(device), \
               torch.tensor(y, dtype=torch.float32).to(device)
    
    return {
        'train': _prepare(X_train_eGe, X_train_VGGish, y_train),
        'val': _prepare(X_val_eGe, X_val_VGGish, y_val),
        'test': _prepare(X_test_eGe, X_test_VGGish, y_test)
    }

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enhance_feature_vector(feature_vector, sampling_rate=None, random_seed=None):  
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    if isinstance(feature_vector, torch.Tensor):
        features = feature_vector.cpu().numpy()
    else:
        features = feature_vector.copy()
    
    original = features.copy()
    
    noises = features + 0.05 * np.random.randn(*features.shape)
    
    pitch_shift = 2  
    pitches = np.roll(features, pitch_shift)
    if pitch_shift > 0:
        pitches[:pitch_shift] = 0
    else:
        pitches[pitch_shift:] = 0
    
    orig_indices = np.arange(len(features))
    stretch_factor = random.choice([0.5, 1.5, 2.0])
    new_indices = np.linspace(0, len(features)-1, int(len(features)*stretch_factor))
    interp_fn = interp1d(orig_indices, features, kind='linear', fill_value='extrapolate')
    stretched = interp_fn(new_indices)
    
    if len(stretched) > len(features):
        stretches = stretched[:len(features)]
    else:
        stretches = np.pad(stretched, (0, len(features)-len(stretched)), mode='constant')
    
    mid_index = len(features) // 2
    cut1 = np.concatenate([features[:mid_index], np.zeros(len(features) - mid_index)])
    cut2 = np.concatenate([np.zeros(mid_index), features[mid_index:]])
    
    enhanced = [original, noises, pitches, stretches, cut1, cut2]
    return [torch.from_numpy(x).float().to(feature_vector.device) if isinstance(feature_vector, torch.Tensor) else x for x in enhanced]

def apply_feature_augmentation(X_train_eGe, selected_indices=None, sampling_rate=None): 
    augmented_data = X_train_eGe.clone()
    selected_indices = range(len(X_train_eGe)) if selected_indices is None else selected_indices
    
    for idx in selected_indices:
        features = X_train_eGe[idx]
        
        enhanced_versions = enhance_feature_vector(features, sampling_rate)
        
        chosen = random.choice(enhanced_versions)
        
        min_len = min(len(features), len(chosen))
        augmented_data[idx, :min_len] = chosen[:min_len]
    
    return augmented_data


def train_model(model, criterion, optimizer, train_data, val_data, params, save_flag=False):
    """Train the model with early stopping"""
    setup_seed(42)
    best_f1 = 0.0
    no_improvement_count = 0

    X_train_eGe, X_train_VGGish, y_train = train_data
    X_val_eGe, X_val_VGGish, y_val = val_data    

    model.train()
    for _ in range(params['epochs']):
        optimizer.zero_grad()
        X_train_eGe_augmented = apply_feature_augmentation(X_train_eGe, sampling_rate=12000)
        X_train_VGGish_augmented = apply_feature_augmentation(X_train_VGGish, sampling_rate=12000)

        outputs = model(X_train_eGe_augmented, X_train_VGGish_augmented).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_eGe, X_val_VGGish).squeeze()      
            val_pred = (val_outputs >= 0.5).int() 
            val_f1 = evaluate_model(y_val.cpu().numpy(), val_pred.cpu().numpy(), val_outputs.cpu().numpy())['f1']
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improvement_count = 0  
            if save_flag:
                torch.save(model.state_dict(), os.path.join(params['save_path'], params['model_save_name']))  
        else:
            no_improvement_count += 1

        if no_improvement_count >= params['patience']:
            break

@dataclass
class EvaluationResults:
    """Class to store evaluation results"""
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: Dict[str, float]
    fpr: np.ndarray
    tpr: np.ndarray

def evaluate_dataset(model, dataset, dataset_type: str) -> EvaluationResults:
    """Evaluate model performance on a dataset"""
    X_eGe, X_VGGish, y_true = dataset
    model.eval()
    with torch.no_grad():
        outputs = model(X_eGe, X_VGGish).squeeze()
        y_prob = outputs.cpu().numpy()
        y_pred = (outputs >= 0.5).int().cpu().numpy()
        y_true = y_true.cpu().numpy()
        
        metrics = evaluate_model(y_true, y_pred, y_prob)
        fpr, tpr = metrics["fpr"], metrics["tpr"]
        
        return EvaluationResults(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
            fpr=fpr,
            tpr=tpr
        )

def update_metrics(metrics_store: Dict, results: EvaluationResults):
    for metric, value in results.metrics.items():
        if metric not in ['fpr','tpr']:
            metrics_store[metric].append(value)

def update_predictions(pred_store: Dict, results: EvaluationResults):
    pred_store['true'].append(results.y_true)
    pred_store['pred'].append(results.y_pred)

def update_roc_data(roc_store: Dict, results: EvaluationResults):
    roc_store['fpr'].append(results.fpr)
    roc_store['tpr'].append(results.tpr)

def plot_confusion_matrix(y_true, y_pred, dataset_name: str, save_path: str):

    plt.rcParams['font.family'] = 'Arial'
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True, vmin=0, vmax=1, linewidths=0.5, annot_kws={'size':14})
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold',c='k')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold',c='k')
    ax.xaxis.set_tick_params(width=2, labelsize=12)
    ax.yaxis.set_tick_params(width=2, labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{dataset_name}.tif'), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{dataset_name}.pdf'), dpi=300, bbox_inches="tight")
    plt.close()

def save_results(metrics: Dict, predictions: Dict, roc_data: Dict, save_path: str):
    """Save evaluation results and plots"""
    metrics_df = pd.DataFrame({
        'Dataset': ['Validation', 'Test'],
        **{f'{k}_mean': [np.mean(metrics['val'][k]), np.mean(metrics['test'][k])] for k in metrics['val']},
        **{f'{k}_std': [np.std(metrics['val'][k]), np.std(metrics['test'][k])] for k in metrics['val']}
    })
    metrics_df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)

    results = pd.DataFrame()
    results['Dataset'] = metrics_df['Dataset']
    for col in metrics_df.columns:
        if '_mean' in col:
            metric = col.split('_mean')[0]
            mean_col = col
            std_col = metric + '_std'
            results[metric] = metrics_df.apply(
                lambda row: f"{row[mean_col]:.3f}±{row[std_col]:.3f}", 
                axis=1
            )
    col = ['Dataset', 'f1', 'roc_auc', 'acc', 'aupr', 'kappa', 'gmean', 'mcc', 'ppv', 'npv', 'sensitivity', 'specificity']
    results = results[col]
    results.to_csv(os.path.join(save_path, 'metrics_meanstd.csv'), index=False, encoding='utf-8-sig')
    print(results.to_string())

    for dataset in predictions:
        plot_confusion_matrix(
            np.concatenate(predictions[dataset]['true']),
            np.concatenate(predictions[dataset]['pred']),
            dataset,
            save_path
        )

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set1",2)
    plt.rcParams['font.family'] = 'Arial'
    dataset_labels = {"val": "Validation", "test": "Test"}
    for i, dataset in enumerate(roc_data):
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(roc_data[dataset]['fpr'], roc_data[dataset]['tpr'])], axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, color=colors[i], label=f'{dataset_labels[dataset]} (AUC = {np.mean(metrics[dataset]["roc_auc"]):.3f})')

    ax = plt.gca()  
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)  # Reference line
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold',c='k')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold',c='k')
    plt.tick_params(axis='x', colors='k')
    plt.tick_params(axis='y', colors='k')
    plt.tick_params(axis='x', labelcolor='k')
    plt.tick_params(axis='y', labelcolor='k')
    plt.legend(loc="lower right", prop={'weight':'bold', 'size':12})
    plt.savefig(os.path.join(save_path, 'roc_curve.tif'),dpi=300,bbox_inches="tight")
    plt.savefig(os.path.join(save_path, 'roc_curve.pdf'),dpi=300,bbox_inches="tight")

    plt.close()

def evaluate_test_model(model_param=None):
    """Main function to evaluate the model"""
    setup_seed(42)
    
    args = parse_args()

    params = {
        'expansion': 2,
        'lr': 5e-4,
        'conv_out_channels': 128,
        'transformed_feature_dim': 256, 
        'cls_dim': 128,
        'weight_decay': 2e-4,
        'sample_weight': 0.5,
        'dropout': 0.1,
        'resblock_kernel_size': 3,
        'epochs': 2000,
        'patience': 100,
        'data_path': args.data_path,
        'save_path': args.save_path,
        'model_save_name': args.model_save_name
    }
    os.makedirs(params['save_path'], exist_ok=True)
    (X_train_eGe_list,X_test_eGe_list,X_val_eGe_list,X_train_VGGish_list,
    X_test_VGGish_list,X_val_VGGish_list,y_train_list,y_test_list,y_val_list) = read_data(params['data_path'])

    metrics = {'val': defaultdict(list), 'test': defaultdict(list)}
    predictions = {'val': {'true': [], 'pred': []}, 
                   'test': {'true': [], 'pred': []}}
    roc_data = {'val': {'fpr': [], 'tpr': []}, 
                'test': {'fpr': [], 'tpr': []}}
    
    model = init_model(params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.BCELoss()

    datasets = prepare_datasets(
        X_train_eGe_list, X_val_eGe_list, X_test_eGe_list,
        X_train_VGGish_list, X_val_VGGish_list, X_test_VGGish_list,
        y_train_list, y_val_list, y_test_list
    )

    train_model(model, criterion, optimizer, datasets['train'], datasets['val'], params, save_flag=True)
    model.load_state_dict(torch.load(os.path.join(params['save_path'], params['model_save_name'])))
    
    for dataset_type in ['val', 'test']:
        results = evaluate_dataset(model, datasets[dataset_type], dataset_type)
        update_metrics(metrics[dataset_type], results)
        update_predictions(predictions[dataset_type], results)
        update_roc_data(roc_data[dataset_type], results)

    save_results(metrics, predictions, roc_data, params['save_path'])

def parse_args():
    parser = argparse.ArgumentParser(description='CFRAFN Model Training and Evaluation')
    
    parser.add_argument('--data_path', type=str, default='../data/demo_data',
                       help='Path to the input data directory')
    parser.add_argument('--save_path', type=str, default='../result/02demo_result/',
                       help='Path to save the results and model')
    parser.add_argument('--model_save_name', type=str, default='best_CFRAFN.pth',
                       help='Filename for saving the best model')
    
    return parser.parse_args()

def main():
    setup_seed(42)
    evaluate_test_model()

if __name__=="__main__":
    main()
