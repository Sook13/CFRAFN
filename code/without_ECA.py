import torch
import torch.nn as nn

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
        return x * y.expand_as(x)

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
    
class EGV_AttNet_without_ECA(nn.Module):
    def __init__(self, input_dim_eGeMAPS, input_dim_VGGish, num_classes, feature_weights, num_conv_layers=2, conv_out_channels=128, transformed_feature_dim=128):
        super(EGV_AttNet_without_ECA, self).__init__()
        self.feature_weights = feature_weights

        # Feature transformation layers
        self.eGeMAPS_transform = nn.Linear(input_dim_eGeMAPS, transformed_feature_dim)
        self.VGGish_transform = nn.Linear(conv_out_channels, transformed_feature_dim)

        # Convolutional network for VGGish features
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels=input_dim_VGGish, out_channels=conv_out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, padding=1))
            conv_layers.append(BasicResBlock(conv_out_channels, conv_out_channels))
            input_dim_VGGish = conv_out_channels
        self.conv_net = nn.Sequential(*conv_layers)
        # self.eca = ECAAttention(conv_out_channels, kernel_size=3)
        self.input_dim = transformed_feature_dim * 2
        # Classifier
        self.classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, x_eGeMAPS, x_VGGish):
        # Process eGeMAPS features
        x_eGeMAPS_weighted = x_eGeMAPS * self.feature_weights
        x_eGeMAPS_transformed = self.eGeMAPS_transform(x_eGeMAPS_weighted)
        x_VGGish = x_VGGish.unsqueeze(-1)  # Add channel dimension
        # Process VGGish features
        conv_out = self.conv_net(x_VGGish)
       # conv_out = self.eca(conv_out)  # Apply ECAAttention
        conv_out = conv_out.squeeze(-1)  # Remove channel dimension
        
        x_VGGish_transformed = self.VGGish_transform(conv_out)
        
        # Merge features directly on the last dimension
        x = torch.cat((x_eGeMAPS_transformed, x_VGGish_transformed), dim=1)

        # Classifier
        output = self.classifier(x)
        return output