"""
生成group_control.csv和self_control.csv，即患者的样本名和类别的对应关系表
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import opensmile
from tqdm import tqdm
import os
import warnings
from tools.common import setup_seed
import librosa
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tools.utils import get_eGe_matrix

def get_data_csv(root_folder):
    cls_files = []
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            cls_files.append((filename, os.path.basename(root).split("_")[-1]))

    cls_files_df = pd.DataFrame(cls_files, columns=['name', 'class'])
    cls_files_df.to_csv(os.path.join(os.path.dirname(root_folder),os.path.basename(root_folder)+".csv"),index=False,encoding="utf-8-sig")

root_folder = '../data/group_control'
get_data_csv(root_folder)

root_folder = '../data/self_control'
get_data_csv(root_folder)

fold = 5
df = pd.read_csv("../data/self_control.csv")
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = shuffled_df['name']
y = shuffled_df['class']
self_folder = "../data/self_control/"


# 创建一个Smile对象，配置为使用eGeMAPS特征集
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

classifiers = {
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

# # 去除方差小于等于0.01的特征
# with open("../result/01preprocess/varLessthan0.01.txt",'r')as f:
#     drop_lst = [line.strip() for line in f]
results = {}
feature_importances = {clf_name: None for clf_name in classifiers.keys()}
clf_f1 = {clf_name: [] for clf_name in classifiers.keys()}

setup_seed(42)
skf = StratifiedKFold(n_splits=fold)

for _ in tqdm(range(1)):
    for train_index, test_index in skf.split(X, y):
        print(train_index)
        print(y[train_index])
        feature_name = []
        X_train, y_train, train_features = get_eGe_matrix(train_index, self_folder, X, y, train=True, n_jobs=-1)
        X_test, y_test, test_features = get_eGe_matrix(test_index, self_folder, X, y, n_jobs=-1)
        
        feature_name.append(train_features.tolist())
        feature_name.append(test_features.tolist())
        for clf_name, clf in classifiers.items():   
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[clf_name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred)
            }
            print(f"{clf_name}  F1 Score: {results[clf_name]['F1 Score']}")  
            if feature_importances[clf_name] is None:
                feature_importances[clf_name] = np.zeros(len(train_features))

            if results[clf_name]['F1 Score'] == 0:
                feature_importances[clf_name] += np.zeros(len(train_features))
            else:
                feature_importances[clf_name] += clf.feature_importances_   
                feature_name.append(clf.feature_names_in_.tolist())
            clf_f1[clf_name].append(results[clf_name]['F1 Score'])
        
        all_same = all(lst == feature_name[0] for lst in feature_name)
        print(f"feature_named 长度：{len(feature_name)}")
        assert all(lst == feature_name[0] for lst in feature_name), f"Expected all feature name are same！"
            # print(clf_f1)

# 计算平均特征重要性
for clf_name in feature_importances.keys():
    feature_importances[clf_name] /= (10 * 5)


f1_clf_df = pd.DataFrame(clf_f1)
f1_clf_df.to_csv("../result/01preprocess/02feature_importance_f1_weighted.csv",encoding='utf-8-sig',index=False)

# 4种分类器的特征重要性表格
feature_df = pd.DataFrame(feature_importances,index=feature_name[0])
feature_df.to_csv("../result/01preprocess/02feature_importance.csv",encoding='utf-8-sig',index=True)



# 重要性加权
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# 加载特征重要性数据
feature_importances_df = pd.read_csv('../result/01preprocess/02feature_importance.csv',index_col=0)
importances_f1_weighted = pd.read_csv('../result/01preprocess/02feature_importance_f1_weighted.csv')

# 计算每个分类器的平均预测值
classifier_means = importances_f1_weighted.mean(axis=0)

# 归一化分类器的平均预测值，得到权重
classifier_weights = classifier_means / classifier_means.sum()
print(classifier_weights)
# 对每个分类器的特征重要性进行归一化（按列归一化）
normalized_feature_importances = feature_importances_df.apply(
    lambda x: x / x.sum(), axis=0
)

# 加权归一化后的特征重要性（按分类器权重加权）
weighted_feature_importances = normalized_feature_importances * classifier_weights

# 最终加权后的特征重要性（按行求和，得到每个特征的总重要性）
final_feature_importances = weighted_feature_importances.sum(axis=1) # final_feature_importances.sum() = 1

sorted_feature_importances = final_feature_importances.sort_values(ascending=False)
sorted_feature_importances.to_csv('../result/01preprocess/03sorted_feature_importance.csv',index=False)

## 重要性曲线
# 画图
x = np.arange(len(sorted_feature_importances))
y = sorted_feature_importances.values

x_new = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=1)
y_smooth = spl(x_new)

plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Arial'
plt.plot(x_new, y_smooth, color='royalblue', linewidth=2, label='Weighted Feature Importance')
plt.bar(x, y, color = 'lightgreen', label = 'Weighted Importance Score', alpha=0.6)

ax = plt.gca()  # 获取当前轴
ax.spines['top'].set_visible(False)  # 隐藏上边框
ax.spines['right'].set_visible(False)  # 隐藏右边框

plt.xlabel('Feature Index (Sorted by Importance)')
plt.ylabel('Weighted Importance Score')
plt.legend()
plt.legend(loc="upper right", prop={'size':10}, labelcolor='black')
plt.savefig('../result/01preprocess/03Weighted_Feature_Importance_curve_scaled.tif', dpi=300, bbox_inches="tight")
plt.savefig('../result/01preprocess/03Weighted_Feature_Importance_curve_scaled.pdf', dpi=300, bbox_inches="tight")
plt.show()

# ax.spines['bottom'].set_linewidth(2)  # 加粗X轴
# ax.spines['left'].set_linewidth(2)  # 加粗Y轴
# 加粗坐标轴刻度
# ax.xaxis.set_tick_params(width=2)  # X轴刻度加粗
# ax.yaxis.set_tick_params(width=2)  # Y轴刻度加粗
# 设置刻度标签的字体
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontweight('bold')

# plt.title('Smoothed and Weighted Feature Importance Across ML Algorithms')






## 相关性热图
import matplotlib.pyplot as plt
import seaborn as sns
# 标签列相关性最强的前top_n个特征热图绘制
def plot_top_features_heatmap(df, label_column='label', top_n=20):
    corr_matrix = df.corr()
    # abs_corr_with_label = abs(corr_matrix[label_column]).sort_values(ascending=False)
    # top_features = abs_corr_with_label[1:top_n + 1].index.tolist()
    # top_corr_matrix = df[top_features + [label_column]].corr()
    top_corr_matrix = corr_matrix

    plt.figure(figsize=(20, 16))
    plt.rcParams['font.family'] = 'Arial'
    sns.heatmap(top_corr_matrix, cmap='coolwarm',
                cbar_kws={'shrink': .5},
                annot=True,
                fmt='.2f',
                linewidths=.5, square=True)
    # plt.title('Top Features Correlation with Label', fontsize=18)

    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('../result/01preprocess/03Top Feature Correlation with Label.tif', dpi=300, bbox_inches="tight")
    plt.savefig('../result/01preprocess/03Top Feature Correlation with Label.pdf', dpi=300, bbox_inches="tight")
    plt.show()

data = pd.read_csv("../result/01preprocess/01features_eGeMAPS_minmax_drop0.01var.csv")
need_col = sorted_feature_importances[:20].index.tolist()
need_col.append("label")
data = data[need_col]
plot_top_features_heatmap(data, 'label', 20)