"""Model definitions for multimodal depression detection experiments."""

import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicResBlock(nn.Module):
    """Basic 1D residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class EgeFeatureBlock(nn.Module):
    """MLP block with residual connection for eGeMAPS features."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class VggFeatureBlock(nn.Module):
    """VGG-style convolutional block with residual shortcut."""

    def __init__(self, dim: int, out_dim: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_dim),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv1d(dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv(x)
        out = out + identity
        out = self.relu(out)
        return out.squeeze(-1)


class FeatureFusionWithAttention(nn.Module):
    """Attention-based fusion between eGeMAPS and VGGish embeddings."""

    def __init__(
        self,
        input_dim_eGeMAPS: int,
        input_dim_VGGish: int,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        combined_dim = input_dim_eGeMAPS + input_dim_VGGish
        self.attention_layer = nn.Linear(combined_dim, 1)
        self.temperature = temperature

    def forward(self, x_eGeMAPS: torch.Tensor, x_VGGish: torch.Tensor) -> torch.Tensor:
        combined_features = torch.cat([x_eGeMAPS, x_VGGish], dim=1)
        logit = self.attention_layer(combined_features)
        alpha = torch.sigmoid(logit / self.temperature)
        return alpha * x_eGeMAPS + (1 - alpha) * x_VGGish


class CFRAFN(nn.Module):
    """Cross-feature representation attention fusion network."""

    def __init__(
        self,
        input_dim_eGeMAPS: int,
        input_dim_VGGish: int,
        feature_weights,
        expansion: int = 4,
        dropout: float = 0.1,
        conv_out_channels: int = 64,
        transformed_feature_dim: int = 128,
        resblock_kernel_size: int = 3,
        cls_dim: int = 128,
    ) -> None:
        super().__init__()

        self.feature_weights = torch.tensor(feature_weights.values, dtype=torch.float32).to(DEVICE)
        self.layer_norm = nn.LayerNorm(normalized_shape=transformed_feature_dim)

        self.eGeMAPS_transform = nn.Linear(input_dim_eGeMAPS, transformed_feature_dim)
        self.eGeMAPS_net = nn.Sequential(
            EgeFeatureBlock(transformed_feature_dim, expansion, dropout),
            nn.Linear(transformed_feature_dim, transformed_feature_dim // 2),
        )

        self.VGGish_transform = nn.Linear(input_dim_VGGish, transformed_feature_dim)
        self.VGGish_net = nn.Sequential(
            VggFeatureBlock(transformed_feature_dim, conv_out_channels, resblock_kernel_size),
            nn.Linear(conv_out_channels, transformed_feature_dim // 2),
        )

        self.fusion = FeatureFusionWithAttention(
            transformed_feature_dim // 2,
            transformed_feature_dim // 2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(transformed_feature_dim // 2, cls_dim),
            nn.ReLU(),
            nn.Linear(cls_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_eGeMAPS: torch.Tensor, x_VGGish: torch.Tensor) -> torch.Tensor:
        x_eGeMAPS_weighted = x_eGeMAPS * self.feature_weights
        x_eGeMAPS_transformed = self.eGeMAPS_transform(x_eGeMAPS_weighted)
        x_eGeMAPS_transformed = self.layer_norm(x_eGeMAPS_transformed)
        x_eGeMAPS_transformed = self.eGeMAPS_net(x_eGeMAPS_transformed)

        x_VGGish = self.VGGish_transform(x_VGGish)
        x_VGGish = self.layer_norm(x_VGGish)
        x_VGGish = x_VGGish.unsqueeze(-1)
        x_VGGish_transformed = self.VGGish_net(x_VGGish)

        x = self.fusion(x_eGeMAPS_transformed, x_VGGish_transformed)
        return self.classifier(x)


class FeatureFusionFCNN(nn.Module):
    """Feature-level fusion model (FCNN_FF)."""

    def __init__(self, input_dim_eGe: int = 88, input_dim_VGGish: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        total_dim = input_dim_eGe + input_dim_VGGish

        self.layers = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_eGe, x_VGGish = x
        merged = torch.cat([x_eGe, x_VGGish], dim=1)
        return self.layers(merged).squeeze()


class LayerFusionFCNN(nn.Module):
    """Layer-level fusion model (FCNN_LF)."""

    def __init__(self, input_dim_eGe: int, input_dim_VGGish: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.eGe_layers = nn.Sequential(
            nn.Linear(input_dim_eGe, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.VGGish_layers = nn.Sequential(
            nn.Linear(input_dim_VGGish, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_eGe, x_VGGish = x
        x_eGe = self.eGe_layers(x_eGe)
        x_VGGish = self.VGGish_layers(x_VGGish)
        merged = torch.cat([x_eGe, x_VGGish], dim=1)
        return self.fusion_layers(merged).squeeze()


class DCNN(nn.Module):
    """Dual-branch 1D CNN for multimodal feature fusion."""

    def __init__(self, input_dim_eGe: int, input_dim_VGGish: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.eGe_conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.VGGish_conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        ege_flatten_size = self._calculate_flatten_size(input_dim_eGe, self.eGe_conv_layers)
        vggish_flatten_size = self._calculate_flatten_size(input_dim_VGGish, self.VGGish_conv_layers)

        self.eGe_branch = nn.Sequential(
            self.eGe_conv_layers,
            nn.Flatten(),
            nn.Linear(ege_flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.VGGish_branch = nn.Sequential(
            self.VGGish_conv_layers,
            nn.Flatten(),
            nn.Linear(vggish_flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.joint_tuning = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def _calculate_flatten_size(self, input_dim: int, conv_layers: nn.Module) -> int:
        x = torch.randn(1, 1, input_dim)
        x = conv_layers(x)
        return int(x.shape[1] * x.shape[2])

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_eGe, x_VGGish = x

        ege_features = self.eGe_branch(x_eGe.unsqueeze(1))
        vggish_features = self.VGGish_branch(x_VGGish.unsqueeze(1))
        merged_features = torch.cat([ege_features, vggish_features], dim=1)
        return self.joint_tuning(merged_features).squeeze()
