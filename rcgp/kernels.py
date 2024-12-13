import torch as tc
import numpy as np

class Matern32Kernel:
    """
    Matern 3/2 Kernel class.
    Computes the Matern 3/2 kernel matrix given fixed lengthscale and magnitude.
    """
    def __init__(self, lengthscale: tc.Tensor, magnitude: tc.Tensor):
        """
        Args:
            lengthscale (torch.Tensor): Scalar tensor for the lengthscale parameter.
            magnitude (torch.Tensor): Scalar tensor for the magnitude (variance) parameter.
        """
        assert lengthscale.ndim == 0 and magnitude.ndim == 0, "lengthscale and magnitude must be scalars (0-d tensors)."
        assert lengthscale > 0, "lengthscale must be positive."
        assert magnitude > 0, "magnitude must be positive."

        self.lengthscale = lengthscale  # Fixed lengthscale tensor
        self.magnitude = magnitude      # Fixed magnitude tensor

    def forward(self, x1: tc.Tensor, x2: tc.Tensor):
        """
        Compute the Matern 3/2 kernel matrix.

        Args:
            x1 (torch.Tensor): Tensor of shape (n1, d), input points 1.
            x2 (torch.Tensor): Tensor of shape (n2, d), input points 2.

        Returns:
            torch.Tensor: Kernel matrix of shape (n1, n2).
        """
        # Normalize inputs using the lengthscale
        scaled_x1 = x1 / self.lengthscale
        scaled_x2 = x2 / self.lengthscale

        # Compute pairwise Euclidean distance
        distance = tc.cdist(scaled_x1, scaled_x2, p=2) + 1e-8  # Add epsilon for numerical stability

        # Compute Matern 3/2 kernel
        sqrt3_d = np.sqrt(3) * distance
        K = (1.0 + sqrt3_d) * tc.exp(-sqrt3_d)

        # Scale the kernel matrix with magnitude (variance)
        return self.magnitude * K


class RBFKernel:
    """
    RBF Kernel (Gaussian Kernel) class.
    Computes the RBF kernel matrix given fixed lengthscale and magnitude.
    """
    def __init__(self, lengthscale: tc.Tensor, magnitude: tc.Tensor):
        """
        Args:
            lengthscale (torch.Tensor): Scalar tensor for the lengthscale parameter.
            magnitude (torch.Tensor): Scalar tensor for the magnitude (variance) parameter.
        """
        assert lengthscale.ndim == 0 and magnitude.ndim == 0, "lengthscale and magnitude must be scalars (0-d tensors)."
        assert lengthscale > 0, "lengthscale must be positive."
        assert magnitude > 0, "magnitude must be positive."

        self.lengthscale = lengthscale  # Fixed lengthscale tensor
        self.magnitude = magnitude      # Fixed magnitude tensor

    def forward(self, x1: tc.Tensor, x2: tc.Tensor):
        """
        Compute the RBF kernel matrix.

        Args:
            x1 (torch.Tensor): Tensor of shape (n1, d), input points 1.
            x2 (torch.Tensor): Tensor of shape (n2, d), input points 2.

        Returns:
            torch.Tensor: Kernel matrix of shape (n1, n2).
        """
        # Normalize inputs using the lengthscale
        scaled_x1 = x1 / self.lengthscale
        scaled_x2 = x2 / self.lengthscale

        # Compute squared pairwise Euclidean distance
        distance_sq = tc.cdist(scaled_x1, scaled_x2, p=2) ** 2

        # Compute the RBF kernel
        K = tc.exp(-0.5 * distance_sq)

        # Scale the kernel matrix with magnitude (variance)
        return self.magnitude * K