import torch
import torch.nn as nn


class LateralInhibition(nn.Module):
    """
    Lateral Inhibition Layer for Neural Networks.

    This module implements lateral inhibition, a neural mechanism where neuron activity
    inhibits the activity of neighboring neurons. This mechanism helps to emphasize important
    features by suppressing the influence of nearby activations.

    Args:
        num_features (int): Number of features (input dimension).
        k (float): Scaling factor for the inhibition strength.
        device (str): Device to run the module on ('cpu' or 'cuda').

    Attributes:
        w (torch.nn.Parameter): Weight matrix for inhibition with no self-inhibition.
        b (torch.nn.Parameter): Bias term for inhibition.
    """

    def __init__(self, num_features, k, device='cpu'):
        """
        Initializes the LateralInhibition layer with the given number of features, inhibition
        strength, and device.

        Args:
            num_features (int): Number of input features.
            k (float): Scaling factor for inhibition strength.
            device (str): Device to run the operations on, defaults to 'cpu'.
        """
        super(LateralInhibition, self).__init__()
        self.num_features = num_features
        self.k = k
        self.device = device

        self.w = nn.Parameter(torch.randn(
            num_features, num_features, device=device))
        self.b = nn.Parameter(torch.randn(num_features, device=device))

    def forward(self, x):
        """
        Forward pass for the Lateral Inhibition layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Inhibited output after applying lateral inhibition.

        The process involves:
            - Removing self-inhibition by setting the diagonal of `w` to zero.
            - Applying sigmoid scaling to adjust the inhibition strength using `k` and `b`.
            - Using matrix multiplication to apply lateral inhibition.
        """
        x = x.to(self.device)

        w_zero_diag = self.w - torch.diag(torch.diag(self.w))

        theta = torch.sigmoid(self.k * (torch.diag(self.w) + self.b))
        diag_part = torch.diag(theta)

        inhibited = (x @ diag_part) @ w_zero_diag
        return inhibited


