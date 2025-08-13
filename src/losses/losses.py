import torch
import torch.nn as nn
import torch.nn.functional as F

def volatility_aware_mse(pred: torch.Tensor, target: torch.Tensor, volatility_window: int = 20) -> torch.Tensor:
    """
    Calculates Mean Squared Error weighted inversely by local volatility.
    As described in PRD section 2.3.1.
    The loss is only computed for the part of the sequence where volatility can be determined.
    """
    # Ensure tensors are at least 2D
    if target.dim() == 1:
        target = target.unsqueeze(0)
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)

    # If the target sequence is shorter than the window, we cannot compute
    # volatility, so we fall back to standard MSE.
    if target.shape[1] < volatility_window:
        return F.mse_loss(pred, target)

    # Estimate local volatility on the target series using a sliding window
    unfolded_target = target.unfold(dimension=1, size=volatility_window, step=1)
    volatility = torch.std(unfolded_target, dim=-1)

    # Weights are inversely proportional to volatility
    weights = 1.0 / (1.0 + volatility)

    # Slice predictions and targets to align with the calculated weights.
    # The weights correspond to the value at the end of each window.
    pred_sliced = pred[:, volatility_window - 1:]
    target_sliced = target[:, volatility_window - 1:]

    # Ensure dimensions match for broadcasting (e.g., for multivariate forecasts)
    if weights.dim() < pred_sliced.dim():
        weights = weights.unsqueeze(-1)

    # Calculate weighted MSE
    loss = weights * (pred_sliced - target_sliced) ** 2
    return loss.mean()


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
