import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_cnn_layers):
        super(ResidualBlock, self).__init__()
        self.basic_block = BasicBlock(in_channels, out_channels, n_cnn_layers)
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        out = self.basic_block(x)
        residual = self.residual(x)
        return out + residual


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_cnn_layers):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        args = [out_channels, out_channels]
        kwargs = {"kernel_size": 3, "padding": 1}
        self.cnn_layers = nn.ModuleList(
            [nn.Conv2d(*args, **kwargs) for i in range(n_cnn_layers)]
        )
        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        for i, layer in enumerate(self.cnn_layers):
            out = F.relu(layer(out))
        out = F.relu(self.conv3(out))
        out = self.dropout(out)
        return out


class AllCNNModel(nn.Module):
    def __init__(self, block, n_blocks, n_cnn_layers, num_classes=10):
        super(AllCNNModel, self).__init__()
        base_channels = 64
        self.layer1 = self._make_layer(block, base_channels, n_blocks, n_cnn_layers)
        self.layer2 = nn.Conv2d(base_channels * 16, num_classes, 1)

    def _make_layer(self, block, base_channels, n_blocks, n_cnn_layers):
        layers = []
        in_channels = 3
        out_channels = base_channels
        for i in range(n_blocks):
            layers.append(block(in_channels, out_channels, n_cnn_layers))
            in_channels = out_channels
            out_channels = 2 * out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out.squeeze()

