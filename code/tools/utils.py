# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2025/04/08 21:28:19
# @Author    :mizzle
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

# 去除方差小于等于0.01的特征
with open("../result/01preprocess/varLessthan0.01.txt",'r')as f:
    drop_lst = [line.strip() for line in f]


# 创建一个Smile对象，配置为使用eGeMAPS特征集
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_vggish_features(file_path, sampling_rate=None):
    model = vggish()
    input_batch = vggish_input.waveform_to_examples(file_path, sampling_rate) # input_batch size = torch.Size([45, 1, 96, 64])
    with torch.no_grad():
        model.eval()
        features_vggish = model(input_batch) # features_vggish=torch.Size([45, 128])
    return features_vggish 


def enhance_data(file, random_seed=42):
    setup_seed(random_seed)
    data, sampling_rate = librosa.load(file, sr=None) 
    noises = data + 0.05 * np.random.randn(len(data)) # 向音频数据中添加不同强度的随机噪声
    pitches = librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=2) # 改变音频的音高
    stretches = librosa.effects.time_stretch(data, rate=2) # 改变音频的播放速度
    volumes = data * 2 # 调整音频的音量
    # 将音频数据从中间切割为两部分
    mid_index = len(data) // 2
    cut1 = data[:mid_index]
    cut2 = data[mid_index:]

    group = [data, noises, pitches, stretches, volumes, cut1, cut2]
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
        features_vggish = features_vggish.mean(dim=0, keepdim=True) # 平均池化
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
            print(f"文件 {filename} 未找到")
        return []
    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(index) for index in data_index)
    features_df = [item for sublist in results for item in sublist] # flatten the list of lists
    features_df = pd.concat(features_df).reset_index(drop=True)
    
    # 去除方差为0的列
    features_df = features_df.drop(columns=drop_lst)

    # 最大最小归一化
    features = features_df.drop(columns=['label','index'])
    # minmax_transfer = MinMaxScaler()
    # features = minmax_transfer.fit_transform(features)

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
    features_df = [item for sublist in results for item in sublist] # flatten the list of lists
    features_df = pd.concat(features_df).reset_index(drop=True)

    # 最大最小归一化
    features = features_df.drop(columns=['label','index'])
    # minmax_transfer = MinMaxScaler()
    # features = minmax_transfer.fit_transform(features)

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