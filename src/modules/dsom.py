import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSOM(nn.Module):
    """
    Fully differentiable Self-Organizing Map.

    This implementation is based on the details provided in section 4.1.1 of the PRD.
    """

    def __init__(self, input_dim: int, map_size: tuple[int, int] = (10, 10), tau: float = 1.0, sigma: float = None):
        """
        Initializes the Differentiable SOM module.

        Args:
            input_dim (int): The dimensionality of the input feature vectors.
            map_size (tuple[int, int]): The dimensions (width, height) of the SOM grid.
            tau (float): The temperature parameter for the softmax function.
            sigma (float, optional): The radius of the neighborhood function. If None, it defaults to max(map_size) / 2.
        """
        super().__init__()
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.tau = tau

        # The PRD's code for DifferentiableSOM uses 'self.sigma' in a calculation
        # but does not define it in __init__. We add it here for completeness.
        # A common default value is half the map's largest dimension.
        if sigma is None:
            self.sigma = max(map_size) / 2.0
        else:
            self.sigma = sigma

        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, input_dim))

        # Precompute grid distances for the neighborhood function.
        # We register this as a buffer so it's not considered a model parameter
        # but is moved to the correct device along with the model.
        self.register_buffer('grid_distances', self._compute_grid_distances())

    def _compute_grid_distances(self) -> torch.Tensor:
        """
        Computes the pairwise Euclidean distances between all prototype coordinates on the grid.

        Returns:
            torch.Tensor: A tensor of shape [n_prototypes, n_prototypes] containing the distances.
        """
        map_w, map_h = self.map_size
        # Create a grid of coordinates for the prototypes
        xv, yv = torch.meshgrid(torch.arange(map_w), torch.arange(map_h), indexing='ij')
        grid_coords = torch.stack([xv.flatten(), yv.flatten()], dim=1).float()

        # Calculate pairwise distances between grid coordinates
        return torch.cdist(grid_coords, grid_coords)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the DSOM.

        Args:
            x (torch.Tensor): The input tensor of shape [B, T, D] (Batch, Time, Dimension) or [B, D].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - assignments (torch.Tensor): The soft assignment tensor of shape [B, T, K].
                - prototypes (torch.Tensor): The learnable prototype vectors of shape [K, D].
        """
        # Ensure input is at least 3D for consistency
        if x.dim() == 2:
            x = x.unsqueeze(1) # Treat as a sequence of length 1

        # Calculate pairwise distances between input vectors and prototypes
        # x: [B, T, D], self.prototypes: [K, D] -> distances: [B, T, K]
        distances = torch.cdist(x, self.prototypes)

        # Compute soft assignments using softmax with a temperature parameter
        assignments = F.softmax(-distances / self.tau, dim=-1)

        # The PRD's code block for DifferentiableSOM (4.1.1) includes a calculation
        # for 'weighted_assignments' using a neighborhood function, but this result
        # is not returned or utilized in other components described in the PRD.
        # The calculation is omitted here to align with the specified return signature,
        # but could be added if a topology-preserving loss is introduced.
        #
        # Example of the calculation:
        # neighborhood_weights = torch.exp(-self.grid_distances / (2 * self.sigma**2))
        # weighted_assignments = assignments @ neighborhood_weights

        return assignments, self.prototypes
