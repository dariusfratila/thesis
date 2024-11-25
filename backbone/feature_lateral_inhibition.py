import torch
import torch.nn as nn


class LateralInhibition(nn.Module):
    """
    Implements a Lateral Inhibition layer to suppress overlapping or redundant features
    and enhance key feature responses.

    Attributes:
        num_features (int): The number of input and output features.
        k (float): Scaling factor for the sigmoid activation applied to the diagonal weights.
        device (str): Device on which computations are performed (e.g., 'cpu' or 'cuda').
        w (nn.Parameter): Learnable weight matrix for feature inhibition.
        b (nn.Parameter): Learnable bias vector for feature inhibition.
    """

    def __init__(self, num_features, k, device='cpu'):
        """
        Initializes the Lateral Inhibition layer.

        Args:
            num_features (int): The number of input and output features.
            k (float): Scaling factor for the sigmoid activation.
            device (str): Device on which computations will be performed ('cpu' or 'cuda').
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
        Performs the forward pass with lateral inhibition.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor after applying lateral inhibition.
        """
        x = x.to(self.device)

        w_zero_diag = self.w - torch.diag(torch.diag(self.w))

        theta = torch.sigmoid(self.k * (torch.diag(self.w) + self.b))
        diag_part = torch.diag(theta)

        inhibited = (x @ diag_part) @ w_zero_diag
        return inhibited
