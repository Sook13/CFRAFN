import pandas as pd
import numpy as np
import opensmile
import os
import warnings
import librosa
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tools.common import setup_seed
from torchvggish import vggish, vggish_input
import torch
import optuna

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_vggish_features(file_path, sampling_rate=None):
    model = vggish()
    input_batch = vggish_input.waveform_to_examples(file_path, sampling_rate) 
    with torch.no_grad():
        model.eval()
        features_vggish = model(input_batch) 
    return features_vggish 


def enhance_data(file, random_seed=42):
    setup_seed(random_seed)
    data, sampling_rate = librosa.load(file, sr=None) 
    noises = data + 0.05 * np.random.randn(len(data)) 
    pitches = librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=2) 
    stretches = librosa.effects.time_stretch(data, rate=2)
    mid_index = len(data) // 2
    cut1 = data[:mid_index]
    cut2 = data[mid_index:]

    group = [data, noises, pitches, stretches, cut1, cut2]
    return group, sampling_rate

def process_group(group, sampling_rate, label, index):
    enhance = []
    num = 0
    for g in group:
        num += 1
        features_eGeMAPS = smile.process_signal(g, sampling_rate)
        
        features_eGeMAPS['label'] = label
        features_eGeMAPS['index'] = str(index)+f".{num}"
        enhance.append(features_eGeMAPS)
    return enhance

def process_vggish_group(group, sampling_rate, label, index):
    enhance = []
    num = 0
    for g in group:
        num += 1
        features_vggish = extract_vggish_features(g, sampling_rate)
        features_vggish = features_vggish.mean(dim=0, keepdim=True) 
        df_vggish = pd.DataFrame(features_vggish.numpy(), columns=[f'feature_{i}' for i in range(features_vggish.shape[1])])
        df_vggish['label'] = label
        df_vggish['index'] = str(index)+f".{num}"
        enhance.append(df_vggish)
    return enhance

def get_eGe_matrix(data_index, folder, X, y, train=False, n_jobs=-1):    
    features_df = []
    def process_file(index):
        filename = X[index]
        label = y[index]
        found = False
        for root, dirs, files in os.walk(folder):
            if filename in files:
                full_path = os.path.join(root, filename)
                if train:
                    group, sampling_rate = enhance_data(full_path)
                    features_eGeMAPS_lst = process_group(group, sampling_rate, label, index)
                    return features_eGeMAPS_lst
                else:
                    features_eGeMAPS = smile.process_file(full_path)
                    features_eGeMAPS['label'] = label
                    features_eGeMAPS['index'] = str(index)
                    return [features_eGeMAPS]
        if not found:
            print(f"file {filename} lost")
        return []
    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(index) for index in data_index)
    features_df = [item for sublist in results for item in sublist] 
    features_df = pd.concat(features_df).reset_index(drop=True)
    
    # features_df = features_df.drop(columns=drop_lst)

    features = features_df.drop(columns=['label','index'])
    df_features = pd.DataFrame(features, columns=features_df.columns.tolist()[:-2])
    df_features['label'] = features_df['label']
    df_features['index'] = features_df['index']

    x_df = df_features.drop(columns=['label','index'])
    y_df = df_features['label']
    return x_df, y_df, x_df.columns, df_features


def get_vggish_features(data_index, folder, X, y, train=False, n_jobs=-1):
    features_df = []
    def process_file(index):
        filename = X[index]
        label = y[index]
        found = False
        for root, dirs, files in os.walk(folder):
            if filename in files:
                full_path = os.path.join(root, filename)
                if train:
                    group, sampling_rate = enhance_data(full_path)
                    features_vggish = process_vggish_group(group, sampling_rate, label, index)
                    return features_vggish
                else: 
                    model = vggish()
                    input_batch = vggish_input.wavfile_to_examples(full_path)
                    with torch.no_grad():
                        model.eval()
                        features_vggish = model(input_batch)
                        features_vggish = features_vggish.mean(dim=0, keepdim=True)
                        df_vggish = pd.DataFrame(features_vggish.numpy(), columns=[f'feature_{i}'for i in range(features_vggish.shape[1])] )
                        df_vggish['label'] = label
                        df_vggish['index'] = str(index)
                    return [df_vggish]
        if not found:
            print(f"File {filename} not found!")
        return []
    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(index) for index in data_index)
    features_df = [item for sublist in results for item in sublist]
    features_df = pd.concat(features_df).reset_index(drop=True)

    features = features_df.drop(columns=['label','index'])
    df_features = pd.DataFrame(features, columns=features_df.columns.tolist()[:-2])
    df_features['label'] = features_df['label']
    df_features['index'] = features_df['index']

    x_df = df_features.drop(columns=['label','index'])
    y_df = df_features['label']
    return x_df, y_df, x_df.columns, df_features

def get_best_para_from_optuna(study_name, storage_name):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_params
    print(study_name,best_params)
    print(f"best value: {study.best_value}")
    return best_params