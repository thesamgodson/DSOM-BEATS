#!/usr/bin/env python
# coding: utf-8

# # DSOM-BEATS: Usage Tutorial and Visualization Guide
#
# This notebook provides a complete walkthrough of how to use the DSOM-BEATS model, including:
#
# 1.  **Configuration**: Loading and understanding the model configuration.
# 2.  **Data Preprocessing**: Loading and splitting a benchmark dataset.
# 3.  **Model Initialization**: Setting up the `DSOM_NBEATS` model.
# 4.  **Training & Validation**: Running the training pipeline with a validation loop.
# 5.  **Testing**: Evaluating the best model on the test set.
# 6.  **Visualization**: Viewing the generated plots for SOM topology and regime analysis.

# ## 1. Setup and Imports
#
# First, let's install the required packages and import the necessary modules.

# In[ ]:


# get_ipython().system('pip install -q -r ../requirements.txt')


# In[ ]:


import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader, TensorDataset

# Add project root to system path
# To make the script runnable from anywhere, we add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.models.nbeats import DSOM_NBEATS
from src.pipelines.training import TrainingPipeline
from src.data.preprocessing import DataPreprocessor


# ## 2. Load Configuration
#
# The model and training pipeline are configured using the `config.yml` file. Let's load it to see the parameters. You can change the benchmark dataset by editing the `data.name` field in `config.yml`.

# In[ ]:


config = load_config('config.yml')
print("--- Data Configuration ---")
print(f"Dataset Name: {config.data.name}")
print(f"Splitting Ratios (Train/Val/Test): {config.data.splitting.train_ratio}/{config.data.splitting.val_ratio}/{config.data.splitting.test_ratio}")

print("--- Model Configuration ---")
print(f"Lookback: {config.lookback}, Horizon: {config.horizon}")
print(f"SOM Map Size: {config.som.map_size}")


# ## 3. Load and Preprocess Benchmark Data
#
# We will now use the `DataPreprocessor` to automatically download, preprocess, and split the benchmark dataset specified in `config.yml`.

# In[ ]:


# Initialize the preprocessor
preprocessor = DataPreprocessor()

# Process the dataset specified in the config
datasets = preprocessor.process(config.data.name, config)

# Create DataLoaders for each set
batch_size = 32
train_loader = DataLoader(TensorDataset(torch.from_numpy(datasets['train'][0]).float(), torch.from_numpy(datasets['train'][1]).float()), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(datasets['val'][0]).float(), torch.from_numpy(datasets['val'][1]).float()), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(datasets['test'][0]).float(), torch.from_numpy(datasets['test'][1]).float()), batch_size=batch_size, shuffle=False)

print(f"Loaded and preprocessed '{config.data.name}' dataset.")
print(f"Number of samples (train/val/test): {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")


# ## 4. Initialize Model and Training Pipeline
#
# Now, let's create an instance of the `DSOM_NBEATS` model and the `TrainingPipeline`.

# In[ ]:


# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine number of features from the data
# Input shape is (n_samples, lookback, n_features)
n_features = datasets['train'][0].shape[2]
print(f"Detected {n_features} features.")

# Initialize model
model = DSOM_NBEATS(config, n_features=n_features).to(device)

# Initialize training pipeline
pipeline = TrainingPipeline(model, config)

print("Model and pipeline initialized successfully.")


# ## 5. Run Training and Validation
#
# Let's train the model for a few epochs. After each epoch, the model is evaluated on the validation set, and the best-performing model checkpoint is saved.

# In[ ]:


n_epochs = 5 # Train for a small number of epochs for this demo

for epoch in range(n_epochs):
    pipeline.train_epoch(train_loader, val_loader, epoch)

print("\nTraining complete!")


# ## 6. Test the Best Model
#
# Now we load the best model (saved during training) and evaluate its performance on the unseen test set.

# In[ ]:


test_metrics = pipeline.test(test_loader)

print("--- Test Set Evaluation ---")
print(json.dumps(test_metrics, indent=2))


# ## 7. View Visualizations
#
# The training process should have created a directory (e.g., `visualizations/`) with the output plots for each epoch. Let's display the plots from the final epoch.

# In[ ]:


# from IPython.display import Image, display

# vis_dir = config.visualization.output_dir

# if not os.path.exists(vis_dir):
#     print(f"Visualization directory '{vis_dir}' not found. Make sure training ran correctly.")
# else:
#     # Display U-Matrix for the last epoch
#     u_matrix_path = os.path.join(vis_dir, f'som_u_matrix_epoch_{n_epochs}.png')
#     if os.path.exists(u_matrix_path):
#         print("--- SOM U-Matrix (Last Epoch) ---")
#         # display(Image(filename=u_matrix_path))

#     # Display Component Planes for the last epoch
#     comp_planes_path = os.path.join(vis_dir, f'som_component_planes_epoch_{n_epochs}.png')
#     if os.path.exists(comp_planes_path):
#         print("\n--- SOM Component Planes (Last Epoch) ---")
#         # display(Image(filename=comp_planes_path))
