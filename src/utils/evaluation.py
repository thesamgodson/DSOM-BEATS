import torch
import numpy as np
from sklearn.metrics import confusion_matrix

class EvaluationMetrics:
    """
    Provides a collection of static methods for evaluating time-series forecasting models,
    as specified in PRD section 5.1.
    """

    @staticmethod
    def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
        """Calculates the R-squared (coefficient of determination) score."""
        ss_res = torch.sum((true - pred) ** 2)
        ss_tot = torch.sum((true - true.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8)) # Add epsilon to avoid division by zero
        return r2.item()

    @staticmethod
    def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
        """Calculates the Mean Absolute Error."""
        return torch.abs(pred - true).mean().item()

    @staticmethod
    def smape(pred: torch.Tensor, true: torch.Tensor) -> float:
        """Calculates the Symmetric Mean Absolute Percentage Error."""
        numerator = torch.abs(pred - true)
        denominator = (torch.abs(true) + torch.abs(pred)) / 2
        # Add epsilon to handle cases where both true and pred are zero
        return (100 * torch.mean(numerator / (denominator + 1e-8))).item()

    @staticmethod
    def interval_coverage(pred_lower: torch.Tensor, pred_upper: torch.Tensor, true: torch.Tensor) -> float:
        """
        Calculates the Prediction Interval Coverage Probability (PICP).
        This measures how often the true value falls within the predicted uncertainty interval.
        """
        in_interval = (true >= pred_lower) & (true <= pred_upper)
        return in_interval.float().mean().item()

    @staticmethod
    def cluster_purity(assignments: torch.Tensor, true_labels: np.ndarray) -> float:
        """
        Calculates cluster purity, a measure of the extent to which clusters contain a single class.
        This is only useful if ground-truth regime labels are available.

        Args:
            assignments (torch.Tensor): The hard cluster assignments from the model, shape [N,].
            true_labels (np.ndarray): The ground-truth labels, shape [N,].

        Returns:
            float: The cluster purity score.
        """
        if assignments.shape != torch.from_numpy(true_labels).shape:
            # Assuming assignments are [B, T, K] or [B, K] -> convert to hard clusters [B*T] or [B]
            hard_assignments = assignments.argmax(dim=-1).flatten().cpu().numpy()
        else:
            hard_assignments = assignments.cpu().numpy()

        cm = confusion_matrix(true_labels, hard_assignments)
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)
