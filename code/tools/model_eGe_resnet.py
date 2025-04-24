import torch
import torch.nn as nn

# kernel_size
# dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class ECAAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out.squeeze(-1)

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

        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(out_channels)

class FeatureFusionWithAttention(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish):
        super().__init__()
        combined_dim = input_dim_eGeMAPS + input_dim_VGGish
        # self.attention_layer = nn.Linear(combined_dim, 1)
        self.attention_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出两个权重
        )
        # self.attention_layer = nn.Linear(combined_dim, combined_dim)
    
    def forward(self, x_eGeMAPS, x_VGGish):
        # 拼接特征
        combined_features = torch.cat([x_eGeMAPS, x_VGGish], dim=1)
        # 生成注意力权重（范围[0,1]）
        # attention = torch.sigmoid(self.attention_layer(combined_features))
        # attention = torch.softmax(self.attention_layer(combined_features),dim=-1)
        weights = torch.softmax(self.attention_net(combined_features), dim=1)
        weights = weights.unsqueeze(-1)
        # 加权融合
        # fused_features = attention * x_eGeMAPS + (1 - attention) * x_VGGish
        fused_features = weights[:,0] * x_eGeMAPS + weights[:,1] * x_VGGish
        # fused_features = combined_features * attention
        # print(f"ege weight:{weights[:,0]}")
        return fused_features
    
class CFRAFN(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish, feature_weights, expansion=4, dropout=0.1, conv_out_channels=64, 
                 transformed_feature_dim=128, resblock_kernel_size=3, cls_dim=128):
        super(CFRAFN, self).__init__()

        self.feature_weights = torch.tensor(feature_weights.values, dtype=torch.float32).to(device)
        self.layer_norm = nn.LayerNorm(normalized_shape=transformed_feature_dim)
        # self.layer_norm = nn.BatchNorm1d(transformed_feature_dim)

        # Feature transformation layers
        self.eGeMAPS_transform = nn.Linear(input_dim_eGeMAPS, transformed_feature_dim)
        self.eGeMAPS_net = nn.Sequential(EgeFeatureBlock(transformed_feature_dim, expansion, dropout),
                                         nn.Linear(transformed_feature_dim, transformed_feature_dim//2),
                                        #  nn.LayerNorm(transformed_feature_dim//2),
                                         )

        self.VGGish_transform = nn.Linear(input_dim_VGGish, transformed_feature_dim)
        self.VGGish_net = nn.Sequential(nn.Linear(transformed_feature_dim, transformed_feature_dim//2),
                                        # nn.LayerNorm(transformed_feature_dim//2),
                                        )
        
        self.fusion = FeatureFusionWithAttention(transformed_feature_dim//2, transformed_feature_dim//2)
        self.classifier = nn.Sequential(nn.Linear(transformed_feature_dim//2, cls_dim),
                                        nn.ReLU(),
                                        nn.Linear(cls_dim, 1),
                                        nn.Sigmoid()
                                        )

    def forward(self, x_eGeMAPS, x_VGGish):
        # Process eGeMAPS features
        x_eGeMAPS_weighted = x_eGeMAPS * self.feature_weights
        x_eGeMAPS_transformed = self.eGeMAPS_transform(x_eGeMAPS_weighted)
        x_eGeMAPS_transformed = self.layer_norm(x_eGeMAPS_transformed)
        x_eGeMAPS_transformed = self.eGeMAPS_net(x_eGeMAPS_transformed)

        x_VGGish = self.VGGish_transform(x_VGGish)
        x_VGGish = self.layer_norm(x_VGGish)
        # x_VGGish = x_VGGish.unsqueeze(-1)  # Add channel dimension
        x_VGGish_transformed = self.VGGish_net(x_VGGish)

        x = self.fusion(x_eGeMAPS_transformed, x_VGGish_transformed)

        # x = x_eGeMAPS_transformed + x_VGGish_transformed

        # Classifier
        output = self.classifier(x)
        return output