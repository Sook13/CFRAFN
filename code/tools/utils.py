"""Utility functions for feature extraction and data preparation."""

import os
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
import opensmile
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed
from torchvggish import vggish, vggish_input

from tools.common import setup_seed


warnings.filterwarnings("ignore")

SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def extract_vggish_features(
    waveform_or_path,
    sampling_rate: int | None = None,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Extract VGGish features from either waveform data or file path."""
    if model is None:
        model = vggish()
        model.eval()

    input_batch = vggish_input.waveform_to_examples(waveform_or_path, sampling_rate)
    with torch.no_grad():
        features_vggish = model(input_batch)
    return features_vggish


def enhance_data(file_path: str | Path, random_seed: int = 42) -> tuple[list[np.ndarray], int]:
    """Generate simple audio augmentations for training."""
    setup_seed(random_seed)
    data, sampling_rate = librosa.load(file_path, sr=None)

    noise_aug = data + 0.05 * np.random.randn(len(data))
    pitch_aug = librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=2)
    stretch_aug = librosa.effects.time_stretch(data, rate=2)

    mid_index = len(data) // 2
    cut1 = data[:mid_index]
    cut2 = data[mid_index:]

    group = [data, noise_aug, pitch_aug, stretch_aug, cut1, cut2]
    return group, sampling_rate


def process_group(
    group: Iterable[np.ndarray],
    sampling_rate: int,
    label,
    index: int,
) -> list[pd.DataFrame]:
    """Extract eGeMAPS features for each augmented sample."""
    enhanced_features: list[pd.DataFrame] = []
    for aug_idx, sample in enumerate(group, start=1):
        features_egemaps = SMILE.process_signal(sample, sampling_rate)
        features_egemaps["label"] = label
        features_egemaps["index"] = f"{index}.{aug_idx}"
        enhanced_features.append(features_egemaps)
    return enhanced_features


def process_vggish_group(
    group: Iterable[np.ndarray],
    sampling_rate: int,
    label,
    index: int,
    model: torch.nn.Module | None = None,
) -> list[pd.DataFrame]:
    """Extract VGGish features for each augmented sample."""
    enhanced_features: list[pd.DataFrame] = []
    for aug_idx, sample in enumerate(group, start=1):
        features_vggish = extract_vggish_features(sample, sampling_rate, model=model)
        features_vggish = features_vggish.mean(dim=0, keepdim=True)
        df_vggish = pd.DataFrame(
            features_vggish.numpy(),
            columns=[f"feature_{i}" for i in range(features_vggish.shape[1])],
        )
        df_vggish["label"] = label
        df_vggish["index"] = f"{index}.{aug_idx}"
        enhanced_features.append(df_vggish)
    return enhanced_features


def get_eGe_matrix(
    data_index: Sequence[int],
    folder: str | Path,
    X: Sequence[str],
    y: Sequence,
    train: bool = False,
    n_jobs: int = -1,
):
    """Build eGeMAPS feature matrix and metadata."""

    def process_file(index: int) -> list[pd.DataFrame]:
        filename = X[index]
        label = y[index]
        for root, _, files in os.walk(folder):
            if filename in files:
                full_path = os.path.join(root, filename)
                if train:
                    group, sampling_rate = enhance_data(full_path)
                    return process_group(group, sampling_rate, label, index)

                data, sampling_rate = librosa.load(full_path, sr=None)
                features_egemaps = SMILE.process_signal(data, sampling_rate)
                features_egemaps["label"] = label
                features_egemaps["index"] = str(index)
                return [features_egemaps]
        return []

    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(index) for index in data_index)
    features_df = [item for sublist in results for item in sublist]
    features_df = pd.concat(features_df).reset_index(drop=True)

    features = features_df.drop(columns=["label", "index"])
    df_features = pd.DataFrame(features, columns=features_df.columns.tolist()[:-2])
    df_features["label"] = features_df["label"]
    df_features["index"] = features_df["index"]

    x_df = df_features.drop(columns=["label", "index"])
    y_df = df_features["label"]
    return x_df, y_df, x_df.columns, df_features


def get_vggish_features(
    data_index: Sequence[int],
    folder: str | Path,
    X: Sequence[str],
    y: Sequence,
    train: bool = False,
    n_jobs: int = -1,
):
    """Build VGGish feature matrix and metadata."""

    def process_file(index: int) -> list[pd.DataFrame]:
        filename = X[index]
        label = y[index]
        model = vggish()
        model.eval()

        for root, _, files in os.walk(folder):
            if filename in files:
                full_path = os.path.join(root, filename)
                try:
                    if train:
                        group, sampling_rate = enhance_data(full_path)
                        return process_vggish_group(group, sampling_rate, label, index, model=model)

                    data, sampling_rate = librosa.load(full_path, sr=None)
                    features_vggish = extract_vggish_features(data, sampling_rate, model=model)
                    features_vggish = features_vggish.mean(dim=0, keepdim=True)
                    df_vggish = pd.DataFrame(
                        features_vggish.numpy(),
                        columns=[f"feature_{i}" for i in range(features_vggish.shape[1])],
                    )
                    df_vggish["label"] = label
                    df_vggish["index"] = str(index)
                    return [df_vggish]
                except Exception as exc:
                    print(f"Error processing {filename}: {exc}")
                    return []
        return []

    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(index) for index in data_index)
    features_df = [item for sublist in results for item in sublist]
    features_df = pd.concat(features_df).reset_index(drop=True)

    meta_cols = ["label", "index"]
    feature_cols = [column for column in features_df.columns if column not in meta_cols]
    x_df = features_df[feature_cols]
    y_df = features_df["label"]
    df_features = pd.concat([x_df, features_df[meta_cols]], axis=1)
    return x_df, y_df, x_df.columns, df_features


def get_best_para_from_optuna(study_name: str, storage_name: str) -> dict:
    """Load Optuna study and print the best parameter set."""
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_params = study.best_params
    print(study_name, best_params)
    print(f"best value: {study.best_value}")
    return best_params
