# DSOM-BEATS: Topology-Aware Hybrid Time-Series Forecaster

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A novel time-series forecasting architecture that combines **Differentiable Self-Organizing Maps (DSOM)** with **N-BEATS** to achieve superior forecasting accuracy through topology-aware feature learning and regime-adaptive prediction routing.

## 🚀 Key Features

- **Differentiable SOM Integration**: End-to-end differentiable clustering for regime detection
- **Regime-Aware Routing**: Dynamic expert selection based on detected market/data regimes
- **Topology Preservation**: Maintains manifold structure of time-series patterns in latent space
- **Volatility-Aware Loss**: Adaptive weighting based on local volatility patterns
- **Curriculum Learning**: Progressive training from low to high volatility samples
- **Alternating Optimization**: Specialized training for forecasting and clustering components

## 📊 Architecture Overview

```
Input Time Series → Feature Extractor → DSOM Clustering
                                          ↓
Trend Expert ←─────── Regime Router ←─── Soft Assignments
Seasonality Expert ←─────────┘
                    ↓
              Final Prediction
```

The system uses a **Differentiable Self-Organizing Map** to identify regime patterns in time-series data, then routes predictions through specialized expert networks (trend and seasonality) based on the detected regimes.

## 🏗️ Project Structure

```
DSOM/
├── src/
│   ├── data/
│   │   └── preprocessing.py      # Data preprocessing pipeline
│   ├── losses/
│   │   └── losses.py            # Custom loss functions
│   ├── models/
│   │   └── nbeats.py           # DSOM-enhanced N-BEATS architecture
│   ├── modules/
│   │   └── dsom.py             # Differentiable SOM implementation
│   ├── pipelines/
│   │   └── training.py         # Training pipeline with curriculum learning
│   └── utils/
│       └── evaluation.py       # Evaluation metrics
├── tests/
│   ├── test_dsom.py            # DSOM unit tests
│   └── test_pipeline.py        # Pipeline integration tests
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/DSOM.git
cd DSOM
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation by running tests:**
```bash
python -m pytest tests/ -v
```

## 🚦 Quick Start

### Basic Usage

1.  **Configure your model**: Edit the `config.yml` file to set your desired parameters for the model, training, and data.

2.  **Run the pipeline**:

```python
import torch
from src.models.nbeats import DSOM_NBEATS
from src.pipelines.training import TrainingPipeline
from src.data.preprocessing import DataPreprocessor
from src.utils.config import load_config

# Load configuration from YAML file
config = load_config('config.yml')

# Initialize model
model = DSOM_NBEATS(config)

# Prepare your data (shape: [batch_size, sequence_length, features])
# X, y = prepare_your_data()  # Implement based on your dataset
# dataloader = create_your_dataloader(X, y)

# Training
pipeline = TrainingPipeline(model, config)

# Optionally, load a checkpoint to resume training
# pipeline.load_checkpoint('checkpoints/checkpoint_epoch_X.pth')

# Start training loop
# start_epoch = pipeline.epoch
# for epoch in range(start_epoch, num_epochs):
#     pipeline.train_epoch(dataloader, epoch)

# Inference
# model.eval()
# with torch.no_grad():
#     predictions, regime_assignments = model(X)
```

### Data Preprocessing

```python
import pandas as pd
from src.data.preprocessing import DataPreprocessor

# Load your time series data
df = pd.read_csv('your_timeseries.csv')

# Configure preprocessing
preprocessing_config = {
    'impute_method': 'forward_fill',
    'remove_outliers': True,
    'outlier_threshold': 1.5,
    'detrend': False,
    'scale_method': 'standard',
    'lookback': 96,
    'horizon': 24,
    'stride': 1
}

# Preprocess data
preprocessor = DataPreprocessor()
X, y = preprocessor.process(df, preprocessing_config)
```

## 🧠 Core Components

### 1. Differentiable Self-Organizing Map (DSOM)

The DSOM module performs topology-aware clustering of time-series features:

```python
from src.modules.dsom import DifferentiableSOM

dsom = DifferentiableSOM(
    input_dim=512,        # Feature dimension
    map_size=(10, 10),    # SOM grid size
    tau=1.0,              # Temperature for soft assignments
    sigma=5.0             # Neighborhood radius
)

# Forward pass
assignments, prototypes = dsom(features)
```

**Key Features:**
- Soft assignments via temperature-controlled softmax
- Learnable prototype vectors
- Precomputed grid distances for neighborhood function
- End-to-end differentiable

### 2. N-BEATS with Regime Routing

Enhanced N-BEATS architecture with specialized expert stacks:

```python
from src.models.nbeats import DSOM_NBEATS

model = DSOM_NBEATS(config)
predictions, assignments = model(input_sequence)
```

**Expert Stacks:**
- **Trend Stack**: Polynomial basis functions for trend modeling.
- **Seasonality Stack**: Fourier basis functions for seasonal patterns.
- **Transformer Stack (Optional)**: A Transformer-based expert for capturing complex temporal patterns. Can be enabled in `config.yml`.
- **Regime Router**: Weighted combination based on SOM assignments.

### 3. Advanced Loss Functions

#### Volatility-Aware MSE
```python
from src.losses.losses import volatility_aware_mse

loss = volatility_aware_mse(predictions, targets, volatility_window=20)
```

#### SOM Quantization Loss
```python
from src.losses.losses import som_quantization_loss

loss = som_quantization_loss(features, prototypes, assignments, beta=0.25)
```

#### Cluster Stability Loss
```python
from src.losses.losses import cluster_stability_loss

loss = cluster_stability_loss(assignments_t, assignments_t_minus_1, gamma=0.1)
```

## 📈 Training Pipeline

The training pipeline implements curriculum learning and alternating optimization:

```python
from src.pipelines.training import TrainingPipeline

pipeline = TrainingPipeline(model, config)

# Training with curriculum learning
for epoch in range(num_epochs):
    pipeline.train_epoch(dataloader, epoch)
```

**Training Features:**
- **Curriculum Learning**: Start with low-volatility samples
- **Alternating Optimization**: Separate updates for forecasting and clustering
- **Stability Regularization**: Penalize rapid regime transitions

## 📊 Evaluation Metrics

```python
from src.utils.evaluation import EvaluationMetrics

# Standard forecasting metrics
r2 = EvaluationMetrics.r2_score(predictions, targets)
mae = EvaluationMetrics.mae(predictions, targets)
smape = EvaluationMetrics.smape(predictions, targets)

# Clustering quality (if ground truth available)
purity = EvaluationMetrics.cluster_purity(assignments, true_labels)
```

## 🧪 Testing

Run the test suite to verify implementation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_dsom.py -v
python -m pytest tests/test_pipeline.py -v

# Run individual tests
python tests/test_dsom.py
```

## 📋 Implementation Status

### ✅ Completed Components

- [x] **Differentiable SOM Module**: Fully implemented with soft assignments
- [x] **N-BEATS Architecture**: Complete with trend/seasonality stacks
- [x] **Regime-Gated Router**: Dynamic expert selection mechanism
- [x] **Advanced Loss Functions**: Volatility-aware, quantization, and stability losses
- [x] **Training Pipeline**: Curriculum learning and alternating optimization
- [x] **Data Preprocessing**: Comprehensive pipeline with scaling and sequence creation
- [x] **Evaluation Metrics**: Standard forecasting and clustering metrics
- [x] **Unit Tests**: DSOM differentiability and assignment validation

### 🚧 Future Enhancements

- [x] **Transformer Integration**: Add transformer-based expert alongside N-BEATS
- [ ] **Visualization Tools**: SOM topology and regime transition plots
- [ ] **Production Deployment**: Model serving and inference optimization
- [x] **Model Checkpointing**: Save/load functionality with reproducibility
- [x] **Configuration Management**: YAML/JSON config file support
- [ ] **Example Notebooks**: Comprehensive tutorials and use cases
- [ ] **Benchmark Datasets**: Integration with standard time-series datasets

## 🔬 Research Background

This implementation is based on the research combining:

1. **Self-Organizing Maps**: Topology-preserving clustering for regime detection
2. **N-BEATS**: State-of-the-art neural basis expansion for time-series forecasting
3. **Mixture of Experts**: Regime-adaptive prediction routing
4. **Curriculum Learning**: Progressive training strategies for improved convergence

## 📚 Key References

- **N-BEATS**: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting" (ICLR 2020)
- **Self-Organizing Maps**: Kohonen, "Self-organizing maps" (Springer, 2001)
- **Differentiable Clustering**: Various papers on end-to-end differentiable clustering (NeurIPS 2019-2023)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original N-BEATS implementation and research team
- Self-Organizing Map research community
- PyTorch and scikit-learn development teams

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a research implementation. For production use, additional testing, optimization, and validation are recommended.
