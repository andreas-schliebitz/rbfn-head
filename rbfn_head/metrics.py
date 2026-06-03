import torch


def sns(centers: torch.Tensor, sigmas: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-normalized separation (SNS) metric based on Gaussian interaction energy for RBF kernel geometry.

    Args:
        centers:    Tensor of shape (K, D)
        sigmas:     Tensor of shape (K,)
        eps:        Numerical stability constant

    Returns:
        Scalar tensor representing geometric quality
        (higher is better, max achieved near normalized distance = 1)
    """
    K, d = centers.shape

    # Pairwise Euclidean distances between centers
    dist = torch.cdist(centers, centers)

    # Scale normalization: sqrt(sigma_i^2 + sigma_j^2)
    scale = torch.sqrt(sigmas[:, None] ** 2 + sigmas[None, :] ** 2 + eps)

    # Normalized distances (d_ij)
    delta = dist / (scale * (d**0.5))

    # Remove diagonal containing self-comparisons
    mask = ~torch.eye(K, dtype=torch.bool, device=delta.device)
    delta = delta[mask]

    # Quality metric: d^2 * exp(-d^2)
    return (delta**2 * torch.exp(-(delta**2))).mean()
