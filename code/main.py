"""Training and evaluation entry point for the CFRAFN model."""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tools.common import setup_seed
from tools.evaluate import evaluate_model
from tools.model import CFRAFN
from tools.utils import get_eGe_matrix, get_vggish_features


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "demo_data"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "result" / "demo_result"
DEFAULT_EGE_FEATURE_PATH = PROJECT_ROOT / "result" / "preprocess" / "eGe_feature_cumul1.txt"
DEFAULT_FEATURE_WEIGHT_PATH = PROJECT_ROOT / "result" / "preprocess" / "sorted_feature_importance.csv"


def to_abs_path(path_value: str) -> Path:
    """Resolve absolute path from user input."""
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return raw_path
    return (CODE_DIR / raw_path).resolve()


def get_data_csv(root_folder: Path) -> Path:
    """Create a CSV mapping filenames to class labels."""
    cls_files = []
    for root, _, files in os.walk(root_folder):
        for filename in files:
            cls_files.append((filename, os.path.basename(root).split("_")[-1]))

    cls_files_df = pd.DataFrame(cls_files, columns=["name", "class"])
    output_csv = root_folder.parent / f"{root_folder.name}.csv"
    cls_files_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv


def load_feature_weights(ege_feature_file: Path, feature_weight_file: Path) -> pd.Series:
    """Load selected eGeMAPS features and aligned feature weights."""
    with open(ege_feature_file, "r", encoding="utf-8") as file_obj:
        ege_feature_names = [line.strip() for line in file_obj]

    feature_weights = pd.read_csv(feature_weight_file, index_col=0)
    return feature_weights.squeeze()[ege_feature_names]


def read_data(data_path: Path):
    """Read data, split train/validation/test, and extract multimodal features."""
    csv_path = data_path.parent / f"{data_path.name}.csv"
    if not csv_path.exists():
        csv_path = get_data_csv(data_path)

    df = pd.read_csv(csv_path)
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = shuffled_df["name"]
    y = shuffled_df["class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    train_index = X.index[X.isin(X_train)].tolist()
    val_index = X.index[X.isin(X_val)].tolist()
    test_index = X.index[X.isin(X_test)].tolist()

    X_train_ege, y_train, _, _ = get_eGe_matrix(train_index, data_path, X, y, train=True, n_jobs=-1)
    X_val_ege, y_val, _, _ = get_eGe_matrix(val_index, data_path, X, y, n_jobs=-1)
    X_test_ege, y_test, _, _ = get_eGe_matrix(test_index, data_path, X, y, n_jobs=-1)

    X_train_vggish, _, _, _ = get_vggish_features(train_index, data_path, X, y, train=True, n_jobs=-1)
    X_val_vggish, _, _, _ = get_vggish_features(val_index, data_path, X, y, n_jobs=-1)
    X_test_vggish, _, _, _ = get_vggish_features(test_index, data_path, X, y, n_jobs=-1)

    return (
        X_train_ege,
        X_test_ege,
        X_val_ege,
        X_train_vggish,
        X_test_vggish,
        X_val_vggish,
        y_train,
        y_test,
        y_val,
    )


def init_model(params: Dict, feature_weights: pd.Series) -> CFRAFN:
    """Initialize CFRAFN with configured hyperparameters."""
    return CFRAFN(
        input_dim_eGeMAPS=len(feature_weights),
        input_dim_VGGish=128,
        feature_weights=feature_weights,
        **{
            key: params[key]
            for key in [
                "expansion",
                "dropout",
                "conv_out_channels",
                "transformed_feature_dim",
                "resblock_kernel_size",
                "cls_dim",
            ]
        },
    ).to(DEVICE)


def prepare_datasets(
    X_train_ege: pd.DataFrame,
    X_val_ege: pd.DataFrame,
    X_test_ege: pd.DataFrame,
    X_train_vggish: pd.DataFrame,
    X_val_vggish: pd.DataFrame,
    X_test_vggish: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert pandas objects to tensors on target device."""

    def _prepare(X_ege, X_vggish, y_labels):
        return (
            torch.tensor(np.asarray(X_ege), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.asarray(X_vggish), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.asarray(y_labels), dtype=torch.float32).to(DEVICE),
        )

    return {
        "train": _prepare(X_train_ege, X_train_vggish, y_train),
        "val": _prepare(X_val_ege, X_val_vggish, y_val),
        "test": _prepare(X_test_ege, X_test_vggish, y_test),
    }


def enhance_feature_vector(feature_vector: torch.Tensor) -> list[torch.Tensor]:
    """Create simple feature-space augmentations for a single vector."""
    features = feature_vector.detach().cpu().numpy()
    original = features.copy()
    noise_aug = features + 0.05 * np.random.randn(*features.shape)

    pitch_shift = 2
    pitch_aug = np.roll(features, pitch_shift)
    pitch_aug[:pitch_shift] = 0

    stretch_factor = np.random.choice([0.5, 1.5, 2.0])
    new_len = int(len(features) * stretch_factor)
    resized = np.interp(
        np.linspace(0, len(features) - 1, new_len),
        np.arange(len(features)),
        features,
    )
    if len(resized) > len(features):
        stretch_aug = resized[: len(features)]
    else:
        stretch_aug = np.pad(resized, (0, len(features) - len(resized)), mode="constant")

    mid_index = len(features) // 2
    cut1 = np.concatenate([features[:mid_index], np.zeros(len(features) - mid_index)])
    cut2 = np.concatenate([np.zeros(mid_index), features[mid_index:]])

    enhanced = [original, noise_aug, pitch_aug, stretch_aug, cut1, cut2]
    return [torch.from_numpy(item).float().to(feature_vector.device) for item in enhanced]


def apply_feature_augmentation(X_train_tensor: torch.Tensor) -> torch.Tensor:
    """Apply random augmentation to each training sample."""
    augmented_data = X_train_tensor.clone()
    for idx in range(len(X_train_tensor)):
        features = X_train_tensor[idx]
        enhanced_versions = enhance_feature_vector(features)
        chosen = enhanced_versions[np.random.randint(len(enhanced_versions))]
        min_len = min(len(features), len(chosen))
        augmented_data[idx, :min_len] = chosen[:min_len]
    return augmented_data


def train_model(model, criterion, optimizer, train_data, val_data, params, save_flag: bool = False) -> None:
    """Train model with early stopping based on validation F1."""
    setup_seed(42)
    best_f1 = 0.0
    no_improvement_count = 0

    X_train_ege, X_train_vggish, y_train = train_data
    X_val_ege, X_val_vggish, y_val = val_data

    for _ in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()

        X_train_ege_aug = apply_feature_augmentation(X_train_ege)
        X_train_vggish_aug = apply_feature_augmentation(X_train_vggish)
        outputs = model(X_train_ege_aug, X_train_vggish_aug).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_ege, X_val_vggish).squeeze()
            val_pred = (val_outputs >= 0.5).int()
            val_f1 = evaluate_model(
                y_val.cpu().numpy(),
                val_pred.cpu().numpy(),
                val_outputs.cpu().numpy(),
            )["f1"]

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improvement_count = 0
            if save_flag:
                model_file = Path(params["save_path"]) / params["model_save_name"]
                torch.save(model.state_dict(), model_file)
        else:
            no_improvement_count += 1

        if no_improvement_count >= params["patience"]:
            break


@dataclass
class EvaluationResults:
    """Container for dataset evaluation outputs."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: Dict[str, float]
    fpr: np.ndarray
    tpr: np.ndarray


def evaluate_dataset(model, dataset) -> EvaluationResults:
    """Evaluate model performance on one dataset split."""
    X_ege, X_vggish, y_true = dataset
    model.eval()
    with torch.no_grad():
        outputs = model(X_ege, X_vggish).squeeze()
        y_prob = outputs.cpu().numpy()
        y_pred = (outputs >= 0.5).int().cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        metrics = evaluate_model(y_true_np, y_pred, y_prob)
        return EvaluationResults(
            y_true=y_true_np,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
            fpr=metrics["fpr"],
            tpr=metrics["tpr"],
        )


def update_metrics(metrics_store: Dict, results: EvaluationResults) -> None:
    for metric, value in results.metrics.items():
        if metric not in ["fpr", "tpr"]:
            metrics_store[metric].append(value)


def update_predictions(pred_store: Dict, results: EvaluationResults) -> None:
    pred_store["true"].append(results.y_true)
    pred_store["pred"].append(results.y_pred)


def update_roc_data(roc_store: Dict, results: EvaluationResults) -> None:
    roc_store["fpr"].append(results.fpr)
    roc_store["tpr"].append(results.tpr)


def plot_confusion_matrix(y_true, y_pred, dataset_name: str, save_path: Path) -> None:
    """Plot and save normalized confusion matrix."""
    plt.rcParams["font.family"] = "Arial"
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        cbar=True,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold", c="k")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold", c="k")
    ax.xaxis.set_tick_params(width=2, labelsize=12)
    ax.yaxis.set_tick_params(width=2, labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(save_path / f"confusion_matrix_{dataset_name}.tif", dpi=300, bbox_inches="tight")
    plt.savefig(save_path / f"confusion_matrix_{dataset_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(metrics: Dict, predictions: Dict, roc_data: Dict, save_path: Path) -> None:
    """Save evaluation metrics, confusion matrices, and ROC curves."""
    metrics_df = pd.DataFrame(
        {
            "Dataset": ["Validation", "Test"],
            **{f"{k}_mean": [np.mean(metrics["val"][k]), np.mean(metrics["test"][k])] for k in metrics["val"]},
            **{f"{k}_std": [np.std(metrics["val"][k]), np.std(metrics["test"][k])] for k in metrics["val"]},
        }
    )
    metrics_df.to_csv(save_path / "metrics.csv", index=False)

    results = pd.DataFrame()
    results["Dataset"] = metrics_df["Dataset"]
    for col in metrics_df.columns:
        if "_mean" in col:
            metric = col.split("_mean")[0]
            mean_col = col
            std_col = f"{metric}_std"
            results[metric] = metrics_df.apply(
                lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}",
                axis=1,
            )

    ordered_cols = [
        "Dataset",
        "f1",
        "roc_auc",
        "acc",
        "aupr",
        "kappa",
        "gmean",
        "mcc",
        "ppv",
        "npv",
        "sensitivity",
        "specificity",
    ]
    results = results[ordered_cols]
    results.to_csv(save_path / "metrics_meanstd.csv", index=False, encoding="utf-8-sig")
    print(results.to_string())

    for dataset in predictions:
        plot_confusion_matrix(
            np.concatenate(predictions[dataset]["true"]),
            np.concatenate(predictions[dataset]["pred"]),
            dataset,
            save_path,
        )

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set1", 2)
    plt.rcParams["font.family"] = "Arial"
    dataset_labels = {"val": "Validation", "test": "Test"}

    for i, dataset in enumerate(roc_data):
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(
            [
                np.interp(mean_fpr, fpr, tpr)
                for fpr, tpr in zip(roc_data[dataset]["fpr"], roc_data[dataset]["tpr"])
            ],
            axis=0,
        )
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=colors[i],
            label=f'{dataset_labels[dataset]} (AUC = {np.mean(metrics[dataset]["roc_auc"]):.3f})',
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)
    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold", c="k")
    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold", c="k")
    plt.tick_params(axis="x", colors="k")
    plt.tick_params(axis="y", colors="k")
    plt.tick_params(axis="x", labelcolor="k")
    plt.tick_params(axis="y", labelcolor="k")
    plt.legend(loc="lower right", prop={"weight": "bold", "size": 12})
    plt.savefig(save_path / "roc_curve.tif", dpi=300, bbox_inches="tight")
    plt.savefig(save_path / "roc_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_test_model(args: argparse.Namespace) -> None:
    """Train and evaluate model using parsed command-line arguments."""
    setup_seed(42)

    data_path = to_abs_path(args.data_path)
    save_path = to_abs_path(args.save_path)
    ege_feature_path = to_abs_path(args.ege_feature_path)
    feature_weight_path = to_abs_path(args.feature_weight_path)

    feature_weights = load_feature_weights(ege_feature_path, feature_weight_path)
    save_path.mkdir(parents=True, exist_ok=True)

    params = {
        "expansion": 2,
        "lr": 5e-4,
        "conv_out_channels": 128,
        "transformed_feature_dim": 256,
        "cls_dim": 128,
        "weight_decay": 2e-4,
        "sample_weight": 0.5,
        "dropout": 0.1,
        "resblock_kernel_size": 3,
        "epochs": 1000,
        "patience": 100,
        "data_path": str(data_path),
        "save_path": str(save_path),
        "model_save_name": args.model_save_name,
    }

    (
        X_train_ege,
        X_test_ege,
        X_val_ege,
        X_train_vggish,
        X_test_vggish,
        X_val_vggish,
        y_train,
        y_test,
        y_val,
    ) = read_data(data_path)

    metrics = {"val": defaultdict(list), "test": defaultdict(list)}
    predictions = {"val": {"true": [], "pred": []}, "test": {"true": [], "pred": []}}
    roc_data = {"val": {"fpr": [], "tpr": []}, "test": {"fpr": [], "tpr": []}}

    model = init_model(params, feature_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.BCELoss()

    datasets = prepare_datasets(
        X_train_ege,
        X_val_ege,
        X_test_ege,
        X_train_vggish,
        X_val_vggish,
        X_test_vggish,
        y_train,
        y_val,
        y_test,
    )

    train_model(model, criterion, optimizer, datasets["train"], datasets["val"], params, save_flag=True)
    model_file = save_path / args.model_save_name
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))

    for dataset_type in ["val", "test"]:
        results = evaluate_dataset(model, datasets[dataset_type])
        update_metrics(metrics[dataset_type], results)
        update_predictions(predictions[dataset_type], results)
        update_roc_data(roc_data[dataset_type], results)

    save_results(metrics, predictions, roc_data, save_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CFRAFN model training and evaluation")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the input data directory",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_SAVE_PATH),
        help="Path to save model and evaluation results",
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="best_CFRAFN622.pth",
        help="Filename for the best model checkpoint",
    )
    parser.add_argument(
        "--ege_feature_path",
        type=str,
        default=str(DEFAULT_EGE_FEATURE_PATH),
        help="Path to selected eGeMAPS feature list text file",
    )
    parser.add_argument(
        "--feature_weight_path",
        type=str,
        default=str(DEFAULT_FEATURE_WEIGHT_PATH),
        help="Path to sorted feature-importance CSV file",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    evaluate_test_model(args)


if __name__ == "__main__":
    main()
