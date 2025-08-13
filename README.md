# DSOM-BEATS: Topology-Aware Hybrid Time-Series Forecaster

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A novel time-series forecasting architecture that combines **Differentiable Self-Organizing Maps (DSOM)** with **N-BEATS** and **Transformer** experts to achieve superior forecasting accuracy through topology-aware feature learning and regime-adaptive prediction routing.

## 🚀 Key Features

- **Differentiable SOM Integration**: End-to-end differentiable clustering for regime detection
- **Regime-Aware Routing**: Dynamic expert selection based on detected market/data regimes
- **Topology Preservation**: Maintains manifold structure of time-series patterns in latent space
- **Stable Loss Function**: Uses standard Mean Squared Error for robust and stable training.
- **Curriculum Learning**: Progressive training from low to high volatility samples
- **Alternating Optimization**: Specialized training for forecasting and clustering components

## 📊 Architecture Overview

```
Input Time Series → Feature Extractor → DSOM Clustering
                                          ↓
Trend Expert ←─────── Regime Router ←─── Soft Assignments
Seasonality Expert ←─────────┘
Transformer Expert ←─────────┘
                    ↓
              Final Prediction
```

The system uses a **Differentiable Self-Organizing Map** to identify regime patterns in time-series data, then routes predictions through specialized expert networks (trend, seasonality, and transformer) based on the detected regimes.

## 🏗️ Project Structure

```
DSOM-BEATS/
├── src/
│   ├── data/
│   │   └── preprocessing.py      # Data preprocessing pipeline
│   ├── losses/
│   │   └── losses.py            # Custom loss functions (volatility-aware MSE, SOM losses)
│   ├── models/
│   │   └── nbeats.py           # DSOM-enhanced N-BEATS architecture
│   ├── modules/
│   │   ├── dsom.py             # Differentiable SOM implementation
│   │   └── transformer.py      # Transformer expert module
│   ├── pipelines/
│   │   └── training.py         # Training pipeline with curriculum learning
│   └── utils/
│       ├── config.py           # Configuration utilities
│       └── evaluation.py       # Evaluation metrics
├── tests/
│   ├── test_dsom.py            # DSOM unit tests
│   └── test_pipeline.py        # Pipeline integration tests
├── config.yml                  # Model configuration
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thesamgodson/DSOM-BEATS.git
   cd DSOM-BEATS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```python
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

## 🚦 Quick Start

### Basic Usage

1.  **Configure your model**: Edit the `config.yml` file to set your desired parameters for the model, training, and data.

2.  **Run the pipeline**:

```python
import torch
from src.models.nbeats import DSOM_NBEATS
from src.utils.config import load_config

# Load configuration
config = load_config('config.yml')

# Initialize model (assuming 1 feature for this example)
n_features = 1
model = DSOM_NBEATS(config, n_features=n_features)

# Sample data (batch_size=32, lookback=96, features=1)
X = torch.randn(32, 96, n_features)

# Forward pass
forecast, som_assignments = model(X)
print(f"Forecast shape: {forecast.shape}")  # [32, 24]
print(f"SOM assignments shape: {som_assignments.shape}")  # [32, 100]
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

#### Forecasting Loss
The model uses a standard Mean Squared Error (MSE) loss for training the forecasting components. The original custom `volatility_aware_mse` was replaced to ensure stability with multivariate datasets.
```python
from src.losses.losses import volatility_aware_mse as forecast_loss

loss = forecast_loss(predictions, targets)
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
from torch.utils.data import DataLoader, TensorDataset

# Prepare your data (example with synthetic data)
X_train = torch.randn(1000, 96)  # 1000 samples, 96 timesteps
y_train = torch.randn(1000, 24)  # 1000 samples, 24 forecast steps
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize training pipeline
pipeline = TrainingPipeline(model, config)

# Train the model with curriculum learning
for epoch in range(50):  # Example: 50 epochs
    pipeline.train_epoch(train_loader, epoch)
    
    # Save checkpoint
    if epoch % 10 == 0:
        pipeline.save_checkpoint(is_best=False)
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

## 🚀 API Usage

This project includes a FastAPI-based web server for serving the trained model.

### 1. Start the Server

To start the API server, run the `serve.py` script:

```bash
python src/serve.py --port 8000
```

### 2. API Documentation

Once the server is running, interactive API documentation (provided by Swagger UI) is available at:
`http://localhost:8000/docs`

### 3. Making a Prediction

You can send a POST request to the `/predict` endpoint with your time-series data. The input data should be a JSON object containing a `data` field with a list of floats. The length of the list must match the `lookback` parameter defined in `config.yml`.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6
  ]
}'
```

The API will return a JSON object with the forecasted values.

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

### 🚧 Future Work

While the core components are in place, future work could include:
- **Production Deployment**: Optimizing the model for serving in a production environment.
- **Hyperparameter Optimization**: Automated tuning of model and training parameters.
- **Extended Benchmarking**: Evaluation against a wider range of public datasets.

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
