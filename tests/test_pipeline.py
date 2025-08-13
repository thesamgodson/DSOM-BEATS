import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os
from argparse import Namespace
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path to allow importing from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.nbeats import DSOM_NBEATS
from src.data.preprocessing import DataPreprocessor
from src.pipelines.training import TrainingPipeline

class TestEndToEndPipeline(unittest.TestCase):
    """
    Integration test for the full data-to-training pipeline.
    This test verifies that all components (preprocessing, model, training)
    work together as expected.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the components required for the test."""
        # 1. Use a mock config that mimics the structure of the YAML-loaded config
        cls.config = Namespace(
            lookback=96,
            horizon=24,
            data=Namespace(
                name='custom',
                splitting=Namespace(
                    train_ratio=0.7,
                    val_ratio=0.15,
                    test_ratio=0.15
                )
            ),
            som=Namespace(
                map_size=[4, 4],
                tau=1.0
            ),
            experts=Namespace(
                trend=Namespace(n_blocks=2, hidden_units=64, poly_degree=2),
                seasonality=Namespace(n_blocks=2, hidden_units=64, fourier_terms=3),
                transformer=Namespace(enabled=False)
            ),
            training=Namespace(
                lr_forecast=0.001,
                lr_som=0.001,
                lambda_stability=0.1,
                curriculum_epochs=1,
                log_interval=5
            ),
            checkpoint=Namespace(
                save_dir='test_checkpoints/',
                save_every_epoch=False
            ),
            visualization=Namespace(
                enabled=False # Disable for tests
            )
        )

        # 2. Create synthetic data
        n_samples = 1000 # Increased samples for meaningful splits
        time = np.arange(n_samples)
        values = np.sin(time * 0.1) + np.random.normal(0, 0.1, n_samples)
        df = pd.DataFrame({'value': values})

        # 3. Preprocess the data
        preprocessor = DataPreprocessor()
        datasets = preprocessor.process(df, cls.config)

        # 4. Create DataLoaders
        cls.train_loader = DataLoader(TensorDataset(torch.from_numpy(datasets['train'][0]).float(), torch.from_numpy(datasets['train'][1]).float()), batch_size=32, shuffle=True)
        cls.val_loader = DataLoader(TensorDataset(torch.from_numpy(datasets['val'][0]).float(), torch.from_numpy(datasets['val'][1]).float()), batch_size=32, shuffle=False)
        cls.test_loader = DataLoader(TensorDataset(torch.from_numpy(datasets['test'][0]).float(), torch.from_numpy(datasets['test'][1]).float()), batch_size=32, shuffle=False)

        # 5. Initialize Model and Pipeline
        cls.model = DSOM_NBEATS(cls.config)
        cls.pipeline = TrainingPipeline(cls.model, cls.config)

    def test_full_pipeline_runs(self):
        """
        Test that a full training epoch completes without errors.
        """
        try:
            self.pipeline.train_epoch(self.train_loader, self.val_loader, epoch=0)
        except Exception as e:
            self.fail(f"Training pipeline failed with an exception: {e}")

    def test_inference_output_shapes(self):
        """
        Test that the model produces outputs of the correct shape during inference.
        """
        self.model.eval()
        x_batch, _ = next(iter(self.train_loader))

        with torch.no_grad():
            pred, assignments = self.model(x_batch)

        batch_size = x_batch.shape[0]
        n_prototypes = self.config.som.map_size[0] * self.config.som.map_size[1]

        # Check prediction shape
        self.assertEqual(pred.shape, (batch_size, self.config.horizon))

        # Check assignments shape
        self.assertEqual(assignments.shape, (batch_size, 1, n_prototypes))

if __name__ == '__main__':
    unittest.main()
