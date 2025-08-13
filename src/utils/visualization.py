import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class VisualizationUtils:
    """
    A utility class for creating visualizations related to the DSOM-BEATS model.
    """

    @staticmethod
    def plot_som_topology(prototypes: torch.Tensor, map_size: tuple[int, int], output_dir: str, epoch: int):
        """
        Generates and saves a U-Matrix visualization of the SOM topology.

        Args:
            prototypes (torch.Tensor): The SOM prototype vectors [n_prototypes, dim].
            map_size (tuple[int, int]): The dimensions (width, height) of the SOM grid.
            output_dir (str): The directory to save the plot.
            epoch (int): The current epoch number, used for the filename.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        map_w, map_h = map_size
        prototypes = prototypes.view(map_w, map_h, -1).cpu().detach().numpy()

        u_matrix = np.zeros((map_w - 1, map_h - 1))

        for i in range(map_w - 1):
            for j in range(map_h - 1):
                d1 = np.linalg.norm(prototypes[i, j] - prototypes[i+1, j])
                d2 = np.linalg.norm(prototypes[i, j] - prototypes[i, j+1])
                u_matrix[i, j] = (d1 + d2) / 2

        plt.figure(figsize=(8, 8))
        plt.imshow(u_matrix, cmap='bone_r', origin='lower')
        plt.title(f'SOM U-Matrix (Epoch {epoch})')
        plt.colorbar(label='Mean Distance to Neighbors')

        plot_path = os.path.join(output_dir, f'som_u_matrix_epoch_{epoch}.png')
        plt.savefig(plot_path)
        plt.close()

    @staticmethod
    def plot_component_planes(prototypes: torch.Tensor, map_size: tuple[int, int], output_dir: str, epoch: int, feature_names: list[str] = None):
        """
        Generates and saves component plane plots for each feature.

        Args:
            prototypes (torch.Tensor): The SOM prototype vectors [n_prototypes, dim].
            map_size (tuple[int, int]): The dimensions (width, height) of the SOM grid.
            output_dir (str): The directory to save the plots.
            epoch (int): The current epoch number.
            feature_names (list[str], optional): Names of the features for titles.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        map_w, map_h = map_size
        n_features = prototypes.shape[1]
        prototypes = prototypes.view(map_w, map_h, -1).cpu().detach().numpy()

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        # Determine grid size for subplots
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = axes.flatten()

        for i in range(n_features):
            ax = axes[i]
            im = ax.imshow(prototypes[:, :, i], cmap='viridis', origin='lower')
            ax.set_title(f'Component Plane: {feature_names[i]}')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax)

        for j in range(n_features, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'SOM Component Planes (Epoch {epoch})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(output_dir, f'som_component_planes_epoch_{epoch}.png')
        plt.savefig(plot_path)
        plt.close()

    @staticmethod
    def plot_regime_transitions(assignments: torch.Tensor, output_dir: str, epoch: int, sample_idx: int = 0):
        """
        Plots the transitions between regimes for a single sample over time.

        Args:
            assignments (torch.Tensor): Soft assignments for a batch [B, T, K].
            output_dir (str): The directory to save the plot.
            epoch (int): The current epoch number.
            sample_idx (int): The index of the sample in the batch to plot.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sample_assignments = assignments[sample_idx].cpu().detach().numpy() # [T, K]
        winning_regime = np.argmax(sample_assignments, axis=1) # [T]

        plt.figure(figsize=(12, 6))
        plt.plot(winning_regime, marker='o', linestyle='-')
        plt.title(f'Regime Transitions for Sample {sample_idx} (Epoch {epoch})')
        plt.xlabel('Time Step')
        plt.ylabel('Winning Regime (Prototype ID)')
        plt.grid(True)

        plot_path = os.path.join(output_dir, f'regime_transitions_epoch_{epoch}_sample_{sample_idx}.png')
        plt.savefig(plot_path)
        plt.close()
