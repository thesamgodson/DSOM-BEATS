import torch
from torch.optim import Adam
import sys
import os
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Add project root to path to allow importing from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import necessary components
from src.models.nbeats import DSOM_NBEATS
from src.losses.losses import volatility_aware_mse, som_quantization_loss, cluster_stability_loss
from src.utils.config import load_config
from src.utils.visualization import VisualizationUtils


def filter_by_volatility(dataloader, percentile: float):
    """
    Filters a dataloader for curriculum learning based on target volatility.
    """
    all_x, all_y = [], []
    for x, y in dataloader:
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        return dataloader

    full_x, full_y = torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
    volatilities = torch.std(full_y, dim=1)

    if volatilities.numel() == 0:
        return dataloader

    threshold = np.percentile(volatilities.cpu().numpy(), percentile * 100)
    low_volatility_indices = (volatilities < threshold).nonzero(as_tuple=True)[0]

    if len(low_volatility_indices) == 0:
        low_volatility_indices = torch.argmin(volatilities).unsqueeze(0)

    filtered_x, filtered_y = full_x[low_volatility_indices], full_y[low_volatility_indices]
    filtered_dataset = TensorDataset(filtered_x, filtered_y)
    return DataLoader(
        filtered_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )


class TrainingPipeline:
    """
    Implements the complete training pipeline using configuration from a YAML file.
    """

    def __init__(self, model: DSOM_NBEATS, config):
        self.model = model
        self.config = config
        self.previous_assignments = None
        self.last_assignments = None  # For visualization
        self.best_val_loss = float('inf')
        self.device = next(model.parameters()).device

        logging.info("Initializing Training Pipeline...")
        logging.info(f"Forecast Optimizer LR: {config.training.lr_forecast}, SOM Optimizer LR: {config.training.lr_som}")
        logging.info(f"Stability Loss Lambda: {config.training.lambda_stability}")

        # Optimizers
        self.forecast_optimizer = Adam(
            [p for n, p in model.named_parameters() if 'dsom' not in n and p.requires_grad],
            lr=config.training.lr_forecast
        )
        self.som_optimizer = Adam(
            model.dsom.parameters(),
            lr=config.training.lr_som
        )

        # Loss Functions
        self.forecast_loss = volatility_aware_mse
        self.som_loss = som_quantization_loss
        self.stability_loss = cluster_stability_loss
        logging.info("Training Pipeline initialized successfully.")
        self.epoch = 0

    def save_checkpoint(self, is_best=False):
        """Saves a checkpoint of the model and optimizers."""
        if not self.config.checkpoint.save_every_epoch:
            return

        save_dir = self.config.checkpoint.save_dir
        os.makedirs(save_dir, exist_ok=True)

        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'forecast_optimizer_state_dict': self.forecast_optimizer.state_dict(),
            'som_optimizer_state_dict': self.som_optimizer.state_dict(),
            'previous_assignments': self.previous_assignments
        }

        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(state, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(state, best_path)
            logging.info(f"Saved new best model to {best_path}")

    def load_checkpoint(self, path: str):
        """Loads a checkpoint to resume training."""
        if not os.path.exists(path):
            logging.warning(f"Checkpoint path not found: {path}. Starting from scratch.")
            return

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.forecast_optimizer.load_state_dict(checkpoint['forecast_optimizer_state_dict'])
        self.som_optimizer.load_state_dict(checkpoint['som_optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        self.previous_assignments = checkpoint.get('previous_assignments') # Use .get for backward compatibility

        logging.info(f"Loaded checkpoint from {path}. Resuming from epoch {self.epoch}.")


    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader, epoch: int):
        """
        Runs a single training epoch.
        """
        self.model.train()
        self.epoch = epoch
        total_batches = len(train_loader)
        logging.info(f"--- Starting Epoch {epoch+1} ---")

        # Curriculum learning
        if hasattr(self.config.training, 'curriculum_epochs') and epoch < self.config.training.curriculum_epochs:
            percentile = (epoch + 1) / self.config.training.curriculum_epochs
            logging.info(f"Applying curriculum learning: filtering for bottom {percentile:.0%} volatility samples.")
            active_dataloader = filter_by_volatility(train_loader, percentile=percentile)
            logging.info(f"Filtered dataloader contains {len(active_dataloader.dataset)} samples.")
        else:
            active_dataloader = train_loader

        for batch_idx, (x, y) in enumerate(active_dataloader):
            x, y = x.to(self.device), y.to(self.device)

            if batch_idx % 2 == 0:
                loss = self.train_forecaster_step(x, y)
                if batch_idx % self.config.training.log_interval == 0:
                    logging.info(f"Epoch {epoch+1} [{batch_idx}/{total_batches}]: Forecaster Loss = {loss:.4f}")
            else:
                loss = self.train_som_step(x)
                if batch_idx % self.config.training.log_interval == 0:
                    logging.info(f"Epoch {epoch+1} [{batch_idx}/{total_batches}]: SOM Loss = {loss:.4f}")

        # --- End of Epoch ---
        val_loss = self.validate_epoch(val_loader)
        logging.info(f"--- Epoch {epoch+1} Complete --- Validation Loss: {val_loss:.4f} ---")

        # Save checkpoint
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            logging.info(f"New best validation loss: {self.best_val_loss:.4f}. Saving best model.")

        self.save_checkpoint(is_best=is_best)

        # Perform end-of-epoch visualizations
        self.visualize_epoch(epoch)

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Runs a validation loop and returns the average loss."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, _ = self.model(x)
                loss = self.forecast_loss(pred, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def test(self, test_loader: DataLoader) -> dict:
        """Evaluates the model on the test set."""
        # Load the best model
        checkpoint_path = os.path.join(self.config.checkpoint.save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            logging.error("Best model checkpoint not found. Run training first.")
            return {}

        self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, _ = self.model(x)
                preds.append(pred.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        # Assuming EvaluationMetrics is available
        from src.utils.evaluation import EvaluationMetrics
        r2 = EvaluationMetrics.r2_score(preds, trues)
        mae = EvaluationMetrics.mae(preds, trues)
        smape = EvaluationMetrics.smape(preds, trues)

        metrics = {'r2': r2, 'mae': mae, 'smape': smape}
        logging.info(f"Test Results: {metrics}")
        return metrics

    def visualize_epoch(self, epoch: int):
        """Generates and saves visualizations at the end of an epoch."""
        if not hasattr(self.config, 'visualization') or not self.config.visualization.enabled:
            return

        logging.info(f"Generating visualizations for epoch {epoch+1}...")
        output_dir = self.config.visualization.output_dir

        # Plot SOM Topology (U-Matrix)
        if self.config.visualization.plot_som_topology:
            VisualizationUtils.plot_som_topology(
                prototypes=self.model.dsom.prototypes,
                map_size=self.model.dsom.map_size,
                output_dir=output_dir,
                epoch=epoch + 1
            )

        # Plot Component Planes
        if self.config.visualization.plot_component_planes:
            VisualizationUtils.plot_component_planes(
                prototypes=self.model.dsom.prototypes,
                map_size=self.model.dsom.map_size,
                output_dir=output_dir,
                epoch=epoch + 1
            )

        # Plot Regime Transitions
        if self.config.visualization.plot_regime_transitions and self.last_assignments is not None:
            VisualizationUtils.plot_regime_transitions(
                assignments=self.last_assignments,
                output_dir=output_dir,
                epoch=epoch + 1
            )
        logging.info("Visualizations saved.")


    def train_forecaster_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Performs a single optimization step on the forecasting components."""
        self.forecast_optimizer.zero_grad()
        pred, assignments = self.model(x)
        f_loss = self.forecast_loss(pred, y)

        # Only calculate stability loss if batch sizes are consistent to avoid shape mismatch
        if self.previous_assignments is not None and self.previous_assignments.shape[0] == assignments.shape[0]:
            prev_assign_device = self.previous_assignments.to(assignments.device)
            s_loss = self.stability_loss(
                assignments_t=assignments,
                assignments_t_minus_1=prev_assign_device
            )
        else:
            s_loss = torch.tensor(0.0, device=assignments.device)

        total_loss = f_loss + self.config.training.lambda_stability * s_loss
        total_loss.backward()
        self.forecast_optimizer.step()

        self.previous_assignments = assignments.detach()
        self.last_assignments = assignments.detach()  # Save for visualization
        return total_loss.item()

    def train_som_step(self, x: torch.Tensor) -> float:
        """Performs a single optimization step on the DSOM module."""
        self.som_optimizer.zero_grad()

        with torch.no_grad():
            if x.shape[-1] == 1:
                x_squeezed = x.squeeze(-1)
            else:
                x_squeezed = x
            features = self.model.feature_extractor(x_squeezed)

        features_unsqueezed = features.unsqueeze(1)
        assignments, prototypes = self.model.dsom(features_unsqueezed)

        loss = self.som_loss(features_unsqueezed, prototypes, assignments)
        loss.backward()
        self.som_optimizer.step()
        return loss.item()
