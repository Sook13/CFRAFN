# 将数据集划分为6:2:2

import pandas as pd
from sklearn.model_selection import train_test_split
from tools.common import setup_seed
from tqdm import tqdm
from joblib import dump
from tools.common import setup_seed
from tools.utils import get_eGe_matrix,get_vggish_features
import os

setup_seed(120)
df = pd.read_csv("../data/group_control.csv")
shuffled_df = df.sample(frac=1).reset_index(drop=True)

X = shuffled_df['name']
y = shuffled_df['class']
self_folder = "../data/group_control/"

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Get indices for each set
train_index = X.index[X.isin(X_train)].tolist()
val_index = X.index[X.isin(X_val)].tolist()
test_index = X.index[X.isin(X_test)].tolist()

# print(f"X_train: {X_train.tolist()}")
# print("\n\n\n\n\n")
# print(f"X_val: {X_val.tolist()}")
# print("\n\n\n\n\n")
# print(f"X_test: {X_test.tolist()}")
print(f"val index: {val_index}")
print(f"test index: {test_index}")
print(f"train index: {train_index[:5]}")
# Get eGeMAPS features
X_train_eGe, y_train, train_features, train_df = get_eGe_matrix(train_index, self_folder, X, y, train=True, n_jobs=-1)
X_val_eGe, y_val, val_features, val_df = get_eGe_matrix(val_index, self_folder, X, y, n_jobs=-1)
X_test_eGe, y_test, test_features, test_df = get_eGe_matrix(test_index, self_folder, X, y, n_jobs=-1)

# Get VGGish features
X_train_VGGish, y_train_vgg, train_features_vgg, train_df_vgg = get_vggish_features(train_index, self_folder, X, y, train=True, n_jobs=-1)
X_val_VGGish, y_val_vgg, val_features_vgg, val_df_vgg = get_vggish_features(val_index, self_folder, X, y, n_jobs=-1)
X_test_VGGish, y_test_vgg, test_features_vgg, test_df_vgg = get_vggish_features(test_index, self_folder, X, y, n_jobs=-1)


save_path = "../data/modelBasicPerformance/origin/"
os.makedirs(save_path, exist_ok=True)

dump(X_train_eGe, os.path.join(save_path, "X_train_eGe.joblib"))
dump(X_val_eGe, os.path.join(save_path, "X_val_eGe.joblib"))
dump(X_test_eGe, os.path.join(save_path, "X_test_eGe.joblib"))
dump(X_train_VGGish, os.path.join(save_path, "X_train_VGGish.joblib"))
dump(X_val_VGGish, os.path.join(save_path, "X_val_VGGish.joblib"))
dump(X_test_VGGish, os.path.join(save_path, "X_test_VGGish.joblib"))
dump(y_train, os.path.join(save_path, "y_train.joblib"))
dump(y_val, os.path.join(save_path, "y_val.joblib"))
dump(y_test, os.path.join(save_path, "y_test.joblib"))
dump(train_features, os.path.join(save_path, "train_features.joblib"))
dump(val_features, os.path.join(save_path, "val_features.joblib"))
dump(test_features, os.path.join(save_path, "test_features.joblib"))
