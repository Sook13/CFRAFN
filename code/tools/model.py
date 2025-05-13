import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class EgeFeatureBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return x + self.mlp(x)
    
class VggFeatureBlock(nn.Module):
    def __init__(self, dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=out_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_dim),
            )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv1d(dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        identity = self.shortcut(identity)  
        out += identity  
        out = self.relu(out)
        return out.squeeze(-1)

class FeatureFusionWithAttention(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish):
        super().__init__()
        combined_dim = input_dim_eGeMAPS + input_dim_VGGish
        self.attention_layer = nn.Linear(combined_dim, 1)
    
    def forward(self, x_eGeMAPS, x_VGGish):
        combined_features = torch.cat([x_eGeMAPS, x_VGGish], dim=1)
        attention = torch.sigmoid(self.attention_layer(combined_features))
        fused_features = attention * x_eGeMAPS + (1 - attention) * x_VGGish
        return fused_features
        
class CFRAFN(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish, feature_weights, expansion=4, dropout=0.1, conv_out_channels=64, 
                 transformed_feature_dim=128, resblock_kernel_size=3, cls_dim=128):
        super(CFRAFN, self).__init__()

        self.feature_weights = torch.tensor(feature_weights.values, dtype=torch.float32).to(device)
        self.layer_norm = nn.LayerNorm(normalized_shape=transformed_feature_dim)

        self.eGeMAPS_transform = nn.Linear(input_dim_eGeMAPS, transformed_feature_dim)
        self.eGeMAPS_net = nn.Sequential(EgeFeatureBlock(transformed_feature_dim, expansion, dropout),
                                         nn.Linear(transformed_feature_dim, transformed_feature_dim//2),
                                         )

        self.VGGish_transform = nn.Linear(input_dim_VGGish, transformed_feature_dim)
        self.VGGish_net = nn.Sequential(VggFeatureBlock(transformed_feature_dim, conv_out_channels, resblock_kernel_size),
                                        nn.Linear(conv_out_channels, transformed_feature_dim//2),
                                        )
        
        self.fusion = FeatureFusionWithAttention(transformed_feature_dim//2, transformed_feature_dim//2)
        self.classifier = nn.Sequential(nn.Linear(transformed_feature_dim//2, cls_dim),
                                        nn.ReLU(),
                                        nn.Linear(cls_dim, 1),
                                        nn.Sigmoid()
                                        )

    def forward(self, x_eGeMAPS, x_VGGish):
        x_eGeMAPS_weighted = x_eGeMAPS * self.feature_weights
        x_eGeMAPS_transformed = self.eGeMAPS_transform(x_eGeMAPS_weighted)
        x_eGeMAPS_transformed = self.layer_norm(x_eGeMAPS_transformed)
        x_eGeMAPS_transformed = self.eGeMAPS_net(x_eGeMAPS_transformed)

        x_VGGish = self.VGGish_transform(x_VGGish)
        x_VGGish = self.layer_norm(x_VGGish)
        x_VGGish = x_VGGish.unsqueeze(-1)  
        x_VGGish_transformed = self.VGGish_net(x_VGGish)

        x = self.fusion(x_eGeMAPS_transformed, x_VGGish_transformed)

        output = self.classifier(x)
        return output


class FeatureFusionFCNN(nn.Module):
    """Feature Fusion model FCNN_FF"""
    def __init__(self, input_dim_eGe=88, input_dim_VGGish=128, dropout=0.5):
        super().__init__()
        self.input_dim_eGe = input_dim_eGe
        self.input_dim_VGGish = input_dim_VGGish
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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_eGe, x_VGGish = x
        x = torch.cat([x_eGe, x_VGGish], dim=1)
        output = self.layers(x)
        return output.squeeze()    

class LayerFusionFCNN(nn.Module):
    """Layer Fusion model FCNN_LF"""
    def __init__(self, input_dim_eGe, input_dim_VGGish, dropout=0.5):
        super().__init__()
        self.input_dim_eGe = input_dim_eGe
        self.input_dim_VGGish = input_dim_VGGish
        
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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_eGe, x_VGGish = x
        x_eGe = self.eGe_layers(x_eGe)
        x_VGGish = self.VGGish_layers(x_VGGish)
        
        x = torch.cat([x_eGe, x_VGGish], dim=1)
        output = self.fusion_layers(x)
        return output.squeeze()


class DCNN(nn.Module):
    """DCNNmodel"""
    def __init__(self, input_dim_eGe, input_dim_VGGish, dropout=0.5):
        super().__init__()
        self.input_dim_eGe = input_dim_eGe
        self.input_dim_VGGish = input_dim_VGGish
        
        self.eGe_conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
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
            nn.MaxPool1d(2)
        )
        
        eGe_flatten_size = self._calculate_flatten_size(input_dim_eGe, self.eGe_conv_layers)
        VGGish_flatten_size = self._calculate_flatten_size(input_dim_VGGish, self.VGGish_conv_layers)
        
         
        self.eGe_branch = nn.Sequential(
            self.eGe_conv_layers,
            nn.Flatten(),
            nn.Linear(eGe_flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        
        self.VGGish_branch = nn.Sequential(
            self.VGGish_conv_layers,
            nn.Flatten(),
            nn.Linear(VGGish_flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.joint_tuning = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _calculate_flatten_size(self, input_dim, conv_layers):
        x = torch.randn(1, 1, input_dim)
        x = conv_layers(x)
        return x.shape[1] * x.shape[2]
    
    def forward(self, x):
        x_eGe, x_VGGish = x
        x_eGe = x_eGe.unsqueeze(1) 
        eGe_features = self.eGe_branch(x_eGe)
        
        x_VGGish = x_VGGish.unsqueeze(1)  
        VGGish_features = self.VGGish_branch(x_VGGish)
        
        merged_features = torch.cat([eGe_features, VGGish_features], dim=1)
        
        output = self.joint_tuning(merged_features)
        
        return output.squeeze()