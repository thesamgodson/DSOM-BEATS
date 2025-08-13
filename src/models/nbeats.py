import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.modules.dsom import DifferentiableSOM
from src.modules.transformer import TransformerStack

# --- Feature Extractor for DSOM ---
class FeatureExtractor(nn.Module):
    """A simple feature extractor to create an embedding of the input time series."""
    def __init__(self, input_size, width):
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )

    def forward(self, x):
        return self.fc_stack(x)

# --- Core N-BEATS Implementation ---
class NBEATSBlock(nn.Module):
    """A single block of the N-BEATS architecture."""
    def __init__(self, input_size, theta_dim, hidden_units, basis_function):
        super().__init__()
        self.basis_function = basis_function
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU()
        )
        self.theta_fc = nn.Linear(hidden_units, theta_dim)

    def forward(self, x):
        hidden_output = self.fc_stack(x)
        theta = self.theta_fc(hidden_output)
        return self.basis_function(theta)

class NBEATSStack(nn.Module):
    """A stack of N-BEATS blocks for trend or seasonality."""
    def __init__(self, input_size, horizon, n_blocks, hidden_units, theta_dim, block_type):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size

        if block_type == 'trend':
            self.basis_function_generator = self.trend_basis
            self.p = theta_dim - 1
            time_vector = torch.arange(horizon, dtype=torch.float32) / horizon
            self.register_buffer('t_forecast', time_vector)
            time_vector_backcast = torch.arange(input_size, dtype=torch.float32)
            self.register_buffer('t_backcast', time_vector_backcast)
        elif block_type == 'seasonality':
            self.basis_function_generator = self.seasonality_basis
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_dim, hidden_units, self.basis_function_generator)
            for _ in range(n_blocks)
        ])

    def trend_basis(self, theta):
        backcast_basis = torch.stack([self.t_backcast**i for i in range(self.p + 1)], dim=0)
        forecast_basis = torch.stack([self.t_forecast**i for i in range(self.p + 1)], dim=0)
        backcast = torch.einsum('bp,pt->bt', theta, backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta, forecast_basis)
        return backcast, forecast

    def seasonality_basis(self, theta):
        n_harmonics = theta.shape[-1] // 2
        freq = torch.arange(1, n_harmonics + 1, device=theta.device, dtype=torch.float32)

        t_forecast = torch.arange(self.horizon, device=theta.device, dtype=torch.float32) * (2 * np.pi / self.horizon)
        forecast_cos = torch.cos(freq.unsqueeze(0) * t_forecast.unsqueeze(1))
        forecast_sin = torch.sin(freq.unsqueeze(0) * t_forecast.unsqueeze(1))
        forecast_basis = torch.cat([forecast_cos, forecast_sin], dim=1)

        t_backcast = torch.arange(self.input_size, device=theta.device, dtype=torch.float32) * (2 * np.pi / self.input_size)
        backcast_cos = torch.cos(freq.unsqueeze(0) * t_backcast.unsqueeze(1))
        backcast_sin = torch.sin(freq.unsqueeze(0) * t_backcast.unsqueeze(1))
        backcast_basis = torch.cat([backcast_cos, backcast_sin], dim=1)

        forecast = torch.einsum('bf,tf->bt', theta, forecast_basis)
        backcast = torch.einsum('bf,tf->bt', theta, backcast_basis)
        return backcast, forecast

    def forward(self, x):
        residual = x
        stack_forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            stack_forecast += forecast
        return stack_forecast

# --- Router ---
class RegimeGatedRouter(nn.Module):
    """Combines expert predictions based on SOM cluster assignments."""
    def __init__(self, n_clusters, n_experts):
        super().__init__()
        self.cluster_to_expert_map = nn.Linear(n_clusters, n_experts)

    def forward(self, expert_preds: list[torch.Tensor], som_assignments: torch.Tensor) -> torch.Tensor:
        avg_assignments = som_assignments.mean(dim=1)
        expert_weights = torch.softmax(self.cluster_to_expert_map(avg_assignments), dim=-1)

        # Stack predictions: from list of [B, H] to [B, E, H]
        stacked_preds = torch.stack(expert_preds, dim=1)

        # Reshape weights from [B, E] to [B, E, 1] for broadcasting
        # This allows multiplying with stacked_preds [B, E, H]
        weighted_preds = stacked_preds * expert_weights.unsqueeze(-1)

        # Sum across the experts dimension
        return weighted_preds.sum(dim=1)

# --- Main Model ---
class DSOM_NBEATS(nn.Module):
    """DSOM-NBEATS architecture configured from a YAML file."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature extractor for DSOM
        self.feature_extractor = FeatureExtractor(
            input_size=config.lookback,
            width=512  # This could also be moved to config
        )

        # DSOM module
        self.dsom = DifferentiableSOM(
            input_dim=512, # Must match feature_extractor width
            map_size=tuple(config.som.map_size),
            tau=config.som.tau
        )

        # Expert Stacks
        self.trend_stack = NBEATSStack(
            input_size=config.lookback,
            horizon=config.horizon,
            n_blocks=config.experts.trend.n_blocks,
            hidden_units=config.experts.trend.hidden_units,
            theta_dim=config.experts.trend.poly_degree + 1,
            block_type='trend'
        )

        self.seasonal_stack = NBEATSStack(
            input_size=config.lookback,
            horizon=config.horizon,
            n_blocks=config.experts.seasonality.n_blocks,
            hidden_units=config.experts.seasonality.hidden_units,
            theta_dim=config.experts.seasonality.fourier_terms * 2,
            block_type='seasonality'
        )

        self.experts = nn.ModuleList([self.trend_stack, self.seasonal_stack])

        # Add Transformer expert if enabled in config
        if self.config.experts.transformer.enabled:
            self.transformer_stack = TransformerStack(
                input_size=config.lookback,
                horizon=config.horizon,
                d_model=config.experts.transformer.d_model,
                nhead=config.experts.transformer.n_head,
                d_hid=config.experts.transformer.d_hid,
                nlayers=config.experts.transformer.n_layers,
                dropout=config.experts.transformer.dropout
            )
            self.experts.append(self.transformer_stack)

        n_experts = len(self.experts)

        self.router = RegimeGatedRouter(
            n_clusters=config.som.map_size[0] * config.som.map_size[1],
            n_experts=n_experts
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the DSOM-NBEATS model."""
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        # 1. Extract features for DSOM routing
        features = self.feature_extractor(x)

        # 2. Get SOM cluster assignments
        assignments, _ = self.dsom(features.unsqueeze(1))

        # 3. Get predictions from all expert stacks
        expert_preds = [expert(x) for expert in self.experts]

        # 4. Combine expert predictions using the router
        final_pred = self.router(expert_preds, assignments)

        return final_pred, assignments
