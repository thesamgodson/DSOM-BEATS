import torch
import torch.nn as nn
import torch.nn.functional as F

def volatility_aware_mse(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Calculates the standard Mean Squared Error loss.
    The original volatility-aware logic was unstable for multivariate targets.
    """
    return F.mse_loss(pred, target)


def som_quantization_loss(features: torch.Tensor, prototypes: torch.Tensor, assignments: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    """
    Calculates the SOM quantization and commitment loss.
    As described in PRD section 2.3.2.

    Args:
        features (torch.Tensor): Input features from encoder, shape [B, T, D].
        prototypes (torch.Tensor): SOM prototype vectors, shape [K, D].
        assignments (torch.Tensor): Soft assignments from SOM, shape [B, T, K].
        beta (float): Weight for the commitment loss.

    Returns:
        torch.Tensor: The combined quantization and commitment loss.
    """
    # 1. Quantization Loss (pulls prototypes towards features)
    # This measures the distance between each feature and all prototypes, weighted by the assignment.
    distances = torch.cdist(features, prototypes)  # Shape: [B, T, K]
    weighted_dist = (assignments * distances).sum(dim=-1)
    quantization_loss = weighted_dist.mean()

    # 2. Commitment Loss (pulls features towards their assigned prototypes)
    # The 'assigned_prototypes' are the weighted average of all prototypes for each feature.
    assigned_prototypes = torch.matmul(assignments, prototypes) # Shape: [B, T, D]

    # The PRD specifies using features.detach() for the commitment loss.
    # This stops gradients from flowing back into the encoder from this loss term,
    # focusing the loss on updating the prototype vectors.
    commitment_loss = F.mse_loss(features.detach(), assigned_prototypes)

    return quantization_loss + beta * commitment_loss


def cluster_stability_loss(assignments_t: torch.Tensor, assignments_t_minus_1: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Penalizes rapid changes in cluster assignments between consecutive time steps.
    As described in PRD section 2.3.3.

    Note: Assumes `assignments_t` and `assignments_t_minus_1` are probability
    distributions (i.e., have had softmax applied).
    """
    if assignments_t is None or assignments_t_minus_1 is None:
        return torch.tensor(0.0, device=assignments_t.device if assignments_t is not None else 'cpu')

    # KL divergence expects log-probabilities as the first argument.
    # Clamp for numerical stability before taking the log.
    log_p_t = torch.log(assignments_t.clamp(min=1e-8))

    # The second argument should be a probability distribution.
    # Detach to prevent gradients from flowing into the previous time step's computation.
    p_t_minus_1 = assignments_t_minus_1.detach()

    # Calculate KL divergence. log_target=False means the target is not in log space.
    transition_cost = F.kl_div(log_p_t, p_t_minus_1, reduction='batchmean', log_target=False)

    return gamma * transition_cost
