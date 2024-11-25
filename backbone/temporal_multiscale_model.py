import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from backbone.feature_lateral_inhibition import LateralInhibition
from dropblock import DropBlock3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) Block.

    This block applies a 1D convolution, followed by batch normalization
    and a ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation rate for the convolution.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bnorm(self.conv(x)))


class MS_TCN(nn.Module):
    """
    Multi-Scale Temporal Convolutional Network (MS-TCN).

    A stack of temporal convolutional layers with varying kernel sizes
    to capture features at multiple temporal scales.

    Args:
        num_layers (int): Number of temporal convolutional layers.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (list): List of kernel sizes for each layer.
    """

    def __init__(self, num_layers, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MS_TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            layers.append(
                TCNBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    dilation=2**i
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LipReadModel(nn.Module):
    """
    Lip Reading Model.

    This model extracts spatial and temporal features from input videos
    using a combination of 3D convolutions, a pre-trained ResNet backbone,
    and a Multi-Scale Temporal Convolutional Network (MS-TCN).
    Lateral inhibition is applied to refine predictions.

    Args:
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_classes):
        super(LipReadModel, self).__init__()
        self.num_classes = num_classes

        self.conv3d = nn.Conv3d(
            1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)
        )
        self.bnorm3d = nn.BatchNorm3d(num_features=64)
        self.relu3d = nn.ReLU(inplace=True)
        self.mpool3d = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.dropblock3d = DropBlock3D(block_size=5, drop_prob=0.1)

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 256)

        self.ms_tcn = MS_TCN(num_layers=4, in_channels=256, out_channels=256)

        self.temporal_conv1 = nn.Conv1d(
            256, 128, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU(inplace=True)

        self.temporal_conv2 = nn.Conv1d(
            128, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bnorm2 = nn.BatchNorm1d(num_classes)
        self.dropout2 = nn.Dropout(p=0.1)

        self.lateral_inhibition = LateralInhibition(
            num_features=num_classes, k=1.0, device=device  # type: ignore
        )

        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the Lip Reading Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.relu3d(self.bnorm3d(self.conv3d(x)))
        if self.training:
            x = self.dropblock3d(x)
        x = self.mpool3d(x)

        batch_size, channels, depth, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * depth, channels, height, width)
        x = self.resnet(x)

        x = x.view(batch_size, depth, -1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.ms_tcn(x)

        x = self.dropout1(self.bnorm1(self.temporal_conv1(x)))
        x = self.relu(x)
        x = self.dropout2(self.bnorm2(self.temporal_conv2(x)))

        x = self.lateral_inhibition(x.view(x.size(0), -1, self.num_classes))
        x = x.mean(dim=1)

        return x

    def _initialize_weights(self):
        """
        Initializes weights for the model using Kaiming initialization
        for convolutional layers and normal initialization for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
