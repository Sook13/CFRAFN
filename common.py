import logging
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import random
import torch
import optuna
import scipy.stats as st
import pandas as pd

# logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    """
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = None
    if log_file:
        file_handler = logging.FileHandler(log_file,encoding="utf-8",mode="a")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger,file_handler

# 随机种子
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # PyTorch 随机数生成器（CPU）
    torch.manual_seed(seed)
    # 如果使用 GPU，确保 CUDA 的随机性也可控
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True # 保证每次结果一样
    torch.backends.cudnn.benchmark = False     # 关闭 cuDNN 的自动优化
# 随机种子
# def set_torch_seed(seed):
#     """
#     Sets the pytorch seeds for current experiment run
#     :param seed: The seed (int)
#     :return: A random number generator to use
#     """
#     rng = np.random.RandomState(seed=seed)
#     torch_seed = rng.randint(0, 999999)
#     torch.manual_seed(seed=torch_seed)

#     return rng

def get_best_para_from_optuna(study_name,storage_name):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_params
    print(study_name,best_params)
    return best_params


class DelongTest():  
    def __init__(self, preds1, preds2, label, threshold=0.05):  
        '''  
        preds1: the output of model1  
        preds2: the output of model2  
        label : the actual label  
        '''  
        self._preds1 = preds1  
        self._preds2 = preds2  
        self._label = label  
        self.threshold = threshold  
        self.z, self.p = self._show_result()
  
    def _auc(self, X, Y) -> float:  
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])  
  
    def _kernel(self, X, Y) -> float:  
        '''Mann-Whitney statistic'''  
        return 0.5 if Y == X else int(Y < X)  
  
    def _structural_components(self, X, Y) -> list:  
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]  
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]  
        return V10, V01  
  
    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:  
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])  
  
    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):  
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** 0.5 + 1e-8)  
  
    def _group_preds_by_label(self, preds, actual) -> list:  
        X = [p for (p, a) in zip(preds, actual) if a]  
        Y = [p for (p, a) in zip(preds, actual) if not a]  
        return X, Y  
  
    def _compute_z_p(self):  
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)  
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)  
  
        V_A10, V_A01 = self._structural_components(X_A, Y_A)  
        V_B10, V_B01 = self._structural_components(X_B, Y_B)  
  
        auc_A = self._auc(X_A, Y_A)  
        auc_B = self._auc(X_B, Y_B)  
  
        # Compute entries of covariance matrix S (covar_AB = covar_BA)  
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1 / len(V_A01))  
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1 / len(V_B01))  
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1 / len(V_A01))  
  
        # Two tailed test  
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)  
        p = st.norm.sf(abs(z)) * 2  
  
        return z, p  
  
    def _show_result(self):  
        z, p = self._compute_z_p()  
        # print("good")
        # print(f"z score = {format_number(z)};\np value = {format_number(p)};")
        # if p < self.threshold:  
        #     print("There is a significant difference")  
        # else:  
        #     print("There is NO significant difference")  
        return z, p  # 返回 z 和 p 值，以便后续使用
    
def format_number(num, sig_figs=5):
    """
    自动格式化数值，根据大小切换常规/科学计数法
    
    参数:
        num: 待格式化的数值
        sig_figs: 有效数字位数（默认5位）
    
    返回:
        格式化后的字符串
    """
    if pd.isna(num):  # 处理缺失值
        return "NaN"
    
    abs_num = abs(num)
    # 定义切换阈值：极小值或极大值用科学计数法
    if abs_num < 1e-4 or abs_num > 1e6:
        return f"{num:.{sig_figs-1}e}"  # 科学计数法保留sig_figs位有效数字
    else:
        # 常规显示：保留足够小数位以保证有效数字
        return f"{num:.{max(sig_figs - len(str(int(abs_num) if abs_num !=0 else '')) -1, 0)}f}"