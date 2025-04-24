import torch
import torch.nn as nn

# kernel_size
# dropout

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
    
class ResidualFeatureBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return x + self.mlp(x)

class FeatureFusionWithAttention(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish):
        super().__init__()
        combined_dim = input_dim_eGeMAPS + input_dim_VGGish
        self.attention_layer = nn.Linear(combined_dim, 1)
    
    def forward(self, x_eGeMAPS, x_VGGish):
        # 拼接特征
        combined_features = torch.cat([x_eGeMAPS, x_VGGish], dim=1)
        # 生成注意力权重（范围[0,1]）
        attention = torch.sigmoid(self.attention_layer(combined_features))
        # 加权融合
        fused_features = attention * x_eGeMAPS + (1 - attention) * x_VGGish
        return fused_features
    
class EGV_AttNet(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish, num_classes, feature_weights, num_conv_layers=3, conv_out_channels=128, 
                 eca_kernel_size=3, transformed_feature_dim=128, resblock_kernel_size=3):
        super(EGV_AttNet, self).__init__()

        self.feature_weights = torch.tensor(feature_weights.values, dtype=torch.float32).to(device)

        # Feature transformation layers
        self.eGeMAPS_transform = nn.Linear(input_dim_eGeMAPS, transformed_feature_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=transformed_feature_dim)
        self.eGeMAPS_net = nn.Sequential(ResidualFeatureBlock(transformed_feature_dim),
                                         nn.Linear(transformed_feature_dim, transformed_feature_dim//2),
                                         nn.LayerNorm(transformed_feature_dim//2),
                                         )


        self.VGGish_transform = nn.Linear(input_dim_VGGish, transformed_feature_dim)

        # 可学习的权重参数（lambda）
        self.lambda_weight = nn.Parameter(torch.tensor(0.5).to(device))  # 初始化为0.5

        # Convolutional network for VGGish features
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels=transformed_feature_dim, out_channels=conv_out_channels, kernel_size=resblock_kernel_size, padding=(resblock_kernel_size-1)//2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, padding=1))
            conv_layers.append(BasicResBlock(conv_out_channels, transformed_feature_dim))


        self.VGGish_net = nn.Sequential(nn.Sequential(*conv_layers),
                                        ECAAttention(transformed_feature_dim, kernel_size=eca_kernel_size),
                                        nn.Linear(transformed_feature_dim, transformed_feature_dim//2),
                                        nn.LayerNorm(transformed_feature_dim//2),
                                        )
        
        # Classifier
        # self.classifier = nn.Linear(self.input_dim, num_classes)
        self.fusion = FeatureFusionWithAttention(transformed_feature_dim//2, transformed_feature_dim//2)

        self.classifier = nn.Sequential(nn.Linear(transformed_feature_dim//2, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, num_classes),
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
        x_VGGish = x_VGGish.unsqueeze(-1)  # Add channel dimension
        # Process VGGish features
        x_VGGish_transformed = self.VGGish_net(x_VGGish)

        # Merge features directly on the last dimension
        # x = torch.cat((x_eGeMAPS_transformed, x_VGGish_transformed), dim=1)

        # 使用可学习的 lambda 加权拼接
        # lambda_weight = torch.sigmoid(self.lambda_weight)  # 限制在 [0, 1] 之间
        # x = lambda_weight * x_eGeMAPS_transformed + (1 - lambda_weight) * x_VGGish_transformed
        x = self.fusion(x_eGeMAPS_transformed, x_VGGish_transformed)

        # Classifier
        output = self.classifier(x)
        return output