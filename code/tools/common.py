import logging
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import random
import torch
import optuna

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