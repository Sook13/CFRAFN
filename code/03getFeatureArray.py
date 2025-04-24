import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tools.common import setup_seed
from tqdm import tqdm
from joblib import dump
from tools.common import setup_seed
from tools.utils import get_eGe_matrix,get_vggish_features
import os
setup_seed(42)
fold = 5
df = pd.read_csv("../data/group_control.csv")
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = shuffled_df['name']
y = shuffled_df['class']
self_folder = "../data/group_control/"

skf = StratifiedKFold(n_splits=fold, shuffle=True)

# 提前提取所有特征
X_train_eGe_list = []
X_test_eGe_list = []
X_train_VGGish_list = []
X_test_VGGish_list = []
y_train_list = []
y_test_list = []
train_features_list = []
test_features_list = []
for _ in tqdm(range(10)):
    for train_index, test_index in skf.split(X, y):
        print(test_index)
        # 获取 eGeMAPS 特征
        X_train_eGe, y_train, train_features, _ = get_eGe_matrix(train_index, self_folder, X, y, train=True, n_jobs=-1)
        X_test_eGe, y_test, test_features, _ = get_eGe_matrix(test_index, self_folder, X, y, n_jobs=-1)

        # 获取 VGGish 特征
        X_train_VGGish, _, _, _ = get_vggish_features(train_index, self_folder, X, y, train=True, n_jobs=-1)
        X_test_VGGish, _, _, _ = get_vggish_features(test_index, self_folder, X, y, n_jobs=-1)

        #保存特征和标签
        X_train_eGe_list.append(X_train_eGe)
        X_test_eGe_list.append(X_test_eGe)
        X_train_VGGish_list.append(X_train_VGGish)
        X_test_VGGish_list.append(X_test_VGGish)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        train_features_list.append(train_features)
        test_features_list.append(test_features)

save_path = "../data/enhance_10/"

X_train_eGe_list_path = os.path.join(save_path,"X_train_eGe_list.joblib")
X_test_eGe_list_path = os.path.join(save_path,"X_test_eGe_list.joblib")
X_train_VGGish_list_path = os.path.join(save_path,"X_train_VGGish_list.joblib")
X_test_VGGish_list_path = os.path.join(save_path,"X_test_VGGish_list.joblib")
y_train_list_path = os.path.join(save_path,"y_train_list.joblib")
y_test_list_path = os.path.join(save_path,"y_test_list.joblib")
train_features_list_path = os.path.join(save_path,"train_features_list.joblib")
test_features_list_path = os.path.join(save_path,"test_features_list.joblib")

# 使用joblib保存增强数据
dump(X_train_eGe_list, X_train_eGe_list_path)
dump(X_test_eGe_list, X_test_eGe_list_path)
dump(X_train_VGGish_list, X_train_VGGish_list_path)
dump(X_test_VGGish_list, X_test_VGGish_list_path)
dump(y_train_list, y_train_list_path)
dump(y_test_list, y_test_list_path)
dump(train_features_list, train_features_list_path) 
dump(test_features_list, test_features_list_path)