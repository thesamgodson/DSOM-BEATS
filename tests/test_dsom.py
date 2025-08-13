import unittest
import torch
import sys
import os

# Add the project root to the Python path to allow importing from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.dsom import DifferentiableSOM

class TestDSOM(unittest.TestCase):
    """
    Unit tests for the DifferentiableSOM module as specified in PRD section 7.1.
    """
    def test_differentiability(self):
        """Ensure DSOM maintains gradient flow to inputs and prototypes."""
        input_dim = 10
        map_size = (5, 5)
        batch_size = 32
        seq_len = 100

        model = DifferentiableSOM(input_dim=input_dim, map_size=map_size)
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

        # Prototypes are nn.Parameters, so they should require gradients by default
        self.assertTrue(model.prototypes.requires_grad)

        assignments, prototypes = model(x)

        # Use a simple loss function (sum of all assignments) to check gradient flow
        loss = assignments.sum()
        loss.backward()

        # 1. Check that the input tensor `x` has a gradient
        self.assertIsNotNone(x.grad, "Input tensor x should have a gradient.")
        self.assertNotEqual(x.grad.abs().sum(), 0, "Gradient for x should not be all zeros.")

        # 2. Check that the prototype tensor has a gradient
        self.assertIsNotNone(model.prototypes.grad, "Prototypes should have a gradient.")
        self.assertNotEqual(model.prototypes.grad.abs().sum(), 0, "Gradient for prototypes should not be all zeros.")

    def test_assignment_sum(self):
        """Verify that the soft assignments for each feature vector sum to 1."""
        input_dim = 10
        map_size = (5, 5)
        batch_size = 32
        seq_len = 100

        model = DifferentiableSOM(input_dim=input_dim, map_size=map_size)
        x = torch.randn(batch_size, seq_len, input_dim)

        assignments, _ = model(x)

        # The sum of assignments over the prototypes (dimension -1) should be 1.
        assignment_sums = assignments.sum(dim=-1)

        # Create a tensor of ones with the same shape for comparison
        expected_sums = torch.ones_like(assignment_sums)

        # Use torch.testing.assert_close for robust floating-point comparison
        torch.testing.assert_close(
            assignment_sums,
            expected_sums,
            rtol=1e-5,  # relative tolerance
            atol=1e-8,  # absolute tolerance
            msg="Assignment sums for each feature vector should be approximately 1."
        )

    def test_output_shapes(self):
        """Test that the output shapes of the DSOM module are correct."""
        input_dim = 10
        map_size = (5, 5)
        n_prototypes = map_size[0] * map_size[1]
        batch_size = 32
        seq_len = 100

        model = DifferentiableSOM(input_dim=input_dim, map_size=map_size)
        x = torch.randn(batch_size, seq_len, input_dim)

        assignments, prototypes = model(x)

        self.assertEqual(assignments.shape, (batch_size, seq_len, n_prototypes))
        self.assertEqual(prototypes.shape, (n_prototypes, input_dim))

        # Test with 2D input
        x_2d = torch.randn(batch_size, input_dim)
        assignments_2d, _ = model(x_2d)
        self.assertEqual(assignments_2d.shape, (batch_size, 1, n_prototypes))


    def test_grid_distances_shape(self):
        """Test the shape of the pre-computed grid distances matrix."""
        input_dim = 10
        map_size = (8, 12) # Use a non-square map
        n_prototypes = map_size[0] * map_size[1]

        model = DifferentiableSOM(input_dim=input_dim, map_size=map_size)

        self.assertIsNotNone(model.grid_distances)
        self.assertEqual(model.grid_distances.shape, (n_prototypes, n_prototypes))

if __name__ == '__main__':
    # This allows running the tests directly from the command line
    unittest.main()
