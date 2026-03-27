"""Shared utility helpers for reproducibility, logging, statistics, and formatting."""

from pathlib import Path
import logging
import random
from typing import Iterable, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import scipy.stats as st
import torch


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(
    log_file: Optional[Path | str] = None,
    log_file_level: int = logging.NOTSET,
) -> Tuple[logging.Logger, Optional[logging.Handler]]:
    """Initialize the root logger with console and optional file handlers."""
    if isinstance(log_file, Path):
        log_file = str(log_file)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.handlers = [console_handler]

    file_handler: Optional[logging.Handler] = None
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, file_handler


def get_best_para_from_optuna(study_name: str, storage_name: str) -> dict:
    """Load an Optuna study and return its best parameter set."""
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_params
    print(study_name, best_params)
    return best_params


class DelongTest:
    """Perform DeLong's test for comparing two ROC AUC scores."""

    def __init__(
        self,
        preds1: Iterable[float],
        preds2: Iterable[float],
        label: Iterable[int],
        threshold: float = 0.05,
    ) -> None:
        """Store predictions, labels, and compute the test statistics."""
        self._preds1 = list(preds1)
        self._preds2 = list(preds2)
        self._label = list(label)
        self.threshold = threshold
        self.z, self.p = self._show_result()

    def _auc(self, positive_scores: list[float], negative_scores: list[float]) -> float:
        return (
            1
            / (len(positive_scores) * len(negative_scores))
            * sum(
                self._kernel(pos_score, neg_score)
                for pos_score in positive_scores
                for neg_score in negative_scores
            )
        )

    def _kernel(self, positive_score: float, negative_score: float) -> float:
        """Compute the Mann-Whitney statistic for a score pair."""
        return 0.5 if negative_score == positive_score else int(negative_score < positive_score)

    def _structural_components(
        self,
        positive_scores: list[float],
        negative_scores: list[float],
    ) -> Tuple[list[float], list[float]]:
        v10 = [
            sum(self._kernel(pos_score, neg_score) for neg_score in negative_scores)
            / len(negative_scores)
            for pos_score in positive_scores
        ]
        v01 = [
            sum(self._kernel(pos_score, neg_score) for pos_score in positive_scores)
            / len(positive_scores)
            for neg_score in negative_scores
        ]
        return v10, v01

    def _get_s_entry(
        self,
        values_a: list[float],
        values_b: list[float],
        auc_a: float,
        auc_b: float,
    ) -> float:
        return sum(
            (value_a - auc_a) * (value_b - auc_b)
            for value_a, value_b in zip(values_a, values_b)
        ) / (len(values_a) - 1)

    def _z_score(
        self,
        variance_a: float,
        variance_b: float,
        covariance_ab: float,
        auc_a: float,
        auc_b: float,
    ) -> float:
        denominator = (variance_a + variance_b - 2 * covariance_ab) ** 0.5 + 1e-8
        return (auc_a - auc_b) / denominator

    def _group_preds_by_label(
        self,
        preds: list[float],
        actual: list[int],
    ) -> Tuple[list[float], list[float]]:
        positive_scores = [pred for pred, label in zip(preds, actual) if label]
        negative_scores = [pred for pred, label in zip(preds, actual) if not label]
        return positive_scores, negative_scores

    def _compute_z_p(self) -> Tuple[float, float]:
        positive_a, negative_a = self._group_preds_by_label(self._preds1, self._label)
        positive_b, negative_b = self._group_preds_by_label(self._preds2, self._label)

        v_a10, v_a01 = self._structural_components(positive_a, negative_a)
        v_b10, v_b01 = self._structural_components(positive_b, negative_b)

        auc_a = self._auc(positive_a, negative_a)
        auc_b = self._auc(positive_b, negative_b)

        variance_a = (
            self._get_s_entry(v_a10, v_a10, auc_a, auc_a) / len(v_a10)
            + self._get_s_entry(v_a01, v_a01, auc_a, auc_a) / len(v_a01)
        )
        variance_b = (
            self._get_s_entry(v_b10, v_b10, auc_b, auc_b) / len(v_b10)
            + self._get_s_entry(v_b01, v_b01, auc_b, auc_b) / len(v_b01)
        )
        covariance_ab = (
            self._get_s_entry(v_a10, v_b10, auc_a, auc_b) / len(v_a10)
            + self._get_s_entry(v_a01, v_b01, auc_a, auc_b) / len(v_a01)
        )

        z_score = self._z_score(variance_a, variance_b, covariance_ab, auc_a, auc_b)
        p_value = st.norm.sf(abs(z_score)) * 2
        return z_score, p_value

    def _show_result(self) -> Tuple[float, float]:
        return self._compute_z_p()


def format_number(num: float, sig_figs: int = 5) -> str:
    """Format a number with adaptive precision for tables and reports."""
    if pd.isna(num):
        return "NaN"

    abs_num = abs(num)
    if abs_num < 1e-4 or abs_num > 1e6:
        return f"{num:.{sig_figs - 1}e}"

    integer_digits = len(str(int(abs_num))) if abs_num != 0 else 0
    decimal_places = max(sig_figs - integer_digits - 1, 0)
    return f"{num:.{decimal_places}f}"
