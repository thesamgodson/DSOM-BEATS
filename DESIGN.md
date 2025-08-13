# DSOM-BEATS: AI-Build Blueprint & Product Requirements

This document outlines the design, requirements, and project plan for the DSOM-BEATS model, framing it as a comprehensive AI-build blueprint.

## 1. Objective & Problem Statement

The primary real-world pain point this project addresses is the challenge of forecasting volatile, non-stationary time-series data, particularly in domains like finance and energy where market dynamics or physical conditions can shift abruptly. Traditional forecasting models often struggle to adapt to these regime changes, leading to poor generalization and unreliable predictions. Hybrid models attempt to solve this but can be complex and fail to properly weigh the contributions of their different components. DSOM-BEATS aims to solve this by creating a topology-aware, hybrid forecaster that uses a Differentiable Self-Organizing Map (DSOM) to intelligently route time-series patterns to specialized expert models (N-BEATS, Transformer). The measurable target is to outperform the standard N-BEATS and other state-of-the-art transformer-based models (e.g., FEDformer, PatchTST) by at least 10% on the SMAPE metric across the ETT (Electricity Transformer Temperature) and M4 benchmark datasets, while also demonstrating superior adaptability in identifiable periods of high volatility.

## 2. Non-functional Requirements

### Performance Goals
- **Inference Latency:** For a single forecast (e.g., via the API), the target latency should be under 100ms on a standard CPU (e.g., Intel Xeon) and under 20ms on a GPU (e.g., NVIDIA T4).
- **GPU Memory Footprint:** During training, the model should not exceed 8GB of VRAM to be trainable on consumer-grade GPUs. During inference, the footprint should be less than 2GB.
- **Training Throughput:** The system should be able to process at least 1,000 samples/second during training on a single GPU.

### Scalability
- **Dataset Size:** The data preprocessing and training pipeline must be able to handle datasets with up to 10 million time steps and 100 features without running into memory issues.
- **Forecast Horizon:** The model architecture should be flexible enough to support forecast horizons up to 720 steps (e.g., for monthly forecasting of daily data) without significant degradation in performance or exponential increase in computational cost.

### Key Trade-offs
- **Accuracy vs. Latency:** The use of multiple experts (Trend, Seasonality, Transformer) and a DSOM router increases accuracy and adaptability but also adds computational overhead. The number of blocks in each expert and the size of the DSOM map can be tuned to trade accuracy for lower latency. The Transformer expert, in particular, can be disabled for a significant speed-up at the cost of capturing less complex patterns.
- **Interpretability vs. Complexity:** While the DSOM provides some interpretability by clustering time-series patterns into regimes, the overall model is highly complex. The basis functions of N-BEATS offer some level of interpretability (trend and seasonality decomposition), but the interaction with the DSOM and the optional Transformer expert makes a full causal explanation difficult. This is a deliberate trade-off in favor of performance.

## 3. Data & Benchmark Plan

### Datasets for Training and Benchmarking
The model will be evaluated on a diverse set of widely-used public benchmark datasets to ensure robust and generalizable performance.
- **ETT (Electricity Transformer Temperature):** Includes `ETTh1`, `ETTh2` (hourly) and `ETTm1`, `ETTm2` (15-minute intervals). These datasets are excellent for testing performance on volatile, real-world sensor data.
- **M4 Competition Dataset:** A large collection of over 100,000 time series from various domains (finance, industry, demographics), providing a test of broad generalization.
- **Traffic:** A dataset of road occupancy rates from the California Department of Transportation, useful for modeling data with strong seasonal patterns and long-term trends.
- **Exchange Rate:** A collection of daily exchange rates of eight countries, representing a challenging financial forecasting task.

### Data Splitting Strategy
A consistent data splitting strategy is crucial for reliable evaluation.
- **For ETT, Traffic, Exchange:** A fixed chronological split of 70% for training, 15% for validation, and 15% for testing will be used. This is a standard approach for these datasets.
- **For M4:** Due to the large number of independent series, evaluation will follow the original competition's guidelines, where the forecast horizon is given and the model is trained on the preceding data for each series. Walk-forward validation will be used on longer series where applicable.

### Baselines for Comparison
To demonstrate the model's effectiveness, its performance will be compared against a set of strong and diverse baselines:
- **Classical Models:** `ARIMA` (as a traditional statistical baseline).
- **Canonical N-BEATS:** The original N-BEATS model without the DSOM or Transformer components.
- **State-of-the-Art Transformer Models:** `FEDformer`, `PatchTST`, and `Informer` will be used as modern deep learning baselines.

## 4. Evaluation Success Criteria

### Primary Metrics
- **Forecasting Accuracy:** The primary metrics will be `sMAPE` (Symmetric Mean Absolute Percentage Error) and `MASE` (Mean Absolute Scaled Error), as they are standard for benchmarking across diverse time series. `R²` will be used as a secondary metric.
- **Regime Detection Accuracy:** Where ground-truth regime labels are available or can be synthesized, `Cluster Purity` and `Normalized Mutual Information (NMI)` will be used to evaluate the DSOM's clustering performance.

### Success Thresholds
- The project will be considered a success if the DSOM-BEATS model demonstrates a **statistically significant improvement of at least 10% in sMAPE/MASE** over the canonical N-BEATS baseline on the ETT and M4 datasets.
- The model should rank as **#1 or #2** when compared against all baselines on at least two of the four benchmark dataset categories.

### Ablation Study Plan
To prove the contribution of the DSOM routing mechanism, a rigorous ablation study will be conducted:
1.  **DSOM-BEATS (Full Model):** The complete proposed architecture.
2.  **N-BEATS + Transformer (No DSOM):** Experts are combined using a simple averaging or a learned weighted average, without the DSOM router. This tests the value of the regime-gated routing.
3.  **DSOM + N-BEATS (No Transformer):** The model without the Transformer expert. This measures the contribution of the Transformer in handling complex patterns.
4.  **Canonical N-BEATS:** The simplest baseline to quantify the overall improvement from the hybrid architecture.
The results of this study will be used to explicitly justify the added complexity of the DSOM component.

## 5. Milestones & Deliverables

The project will be developed in four distinct phases:

### Phase 1: Core Model Integration (Complete)
- **Deliverable:** A functional Python script integrating a Differentiable SOM (DSOM) with a standard N-BEATS model.
- **Status:** Complete. The core logic of the `DifferentiableSOM` and `DSOM_NBEATS` classes is implemented.

### Phase 2: Advanced Features (Complete)
- **Deliverable:** Integration of the Transformer expert, volatility-aware loss functions, and a curriculum learning strategy.
- **Status:** Complete. These features are implemented in the training pipeline and model architecture.

### Phase 3: Robust Pipeline & Tooling (Complete)
- **Deliverable:** A full-featured training and evaluation pipeline, including data splitting, validation-based checkpointing, a serving API, and visualization tools.
- **Status:** Complete. The project now includes a `Dockerfile`, a benchmark data manager, a FastAPI server, and a comprehensive example notebook.

### Phase 4: Benchmarking & Analysis (Current Focus)
- **Deliverable:** A comprehensive set of benchmark results comparing DSOM-BEATS against all defined baselines on the specified datasets. This includes a full ablation study.
- **Key Tasks:**
    - Run experiments for all model variants and baselines.
    - Generate paper-ready figures and tables summarizing the results (e.g., performance metrics, U-Matrix visualizations, regime transition plots).
    - Write a final analysis and conclusion.

## 6. Deployment Strategy

### Containerization for Reproducibility
- **Dockerfile:** The project includes a `Dockerfile` that packages the application and all its dependencies into a container. This ensures that the environment is reproducible and the application can be deployed consistently across different systems (local, cloud, on-prem). Both CPU and GPU base images are supported to allow for flexible deployment.

### Model Serialization
- **Format:** The primary serialization format is PyTorch's native `.pth` (or `.pt`) format, which saves the model's `state_dict`. This is simple, robust, and sufficient for the current FastAPI-based deployment.
- **Future Work (Optional):** For higher performance or cross-platform deployment (e.g., mobile, web), the model could be exported to `TorchScript` or `ONNX`.
    - **TorchScript:** Would JIT-compile the model for a portable, optimized format.
    - **ONNX (Open Neural Network Exchange):** Would allow the model to be run on a wide variety of inference engines (e.g., ONNX Runtime, TensorRT), potentially unlocking further performance gains.

### API Scaling Strategy
The current FastAPI application runs as a single process via Uvicorn. For a production environment with high request volume, the following scaling strategies can be employed:
- **Horizontal Scaling:** Run multiple instances of the containerized application behind a load balancer (e.g., Nginx, AWS ELB). This is the most straightforward way to handle more concurrent requests.
- **Gunicorn:** Use `Gunicorn` as a process manager to run multiple Uvicorn worker processes on a single machine, taking advantage of multi-core CPUs.
- **Batch Inference:** For non-real-time use cases, the API could be extended with an endpoint that accepts a batch of prediction requests to improve throughput.

## 7. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **DSOM fails to converge or produces unstable clusters.** | Medium | High | **Mitigation:** Initialize DSOM prototypes using a pre-trained K-Means clustering on a subset of the data. Implement a "warm-start" period where the DSOM is trained with a higher learning rate before the full model. Add regularization to penalize rapid changes in prototype positions. |
| **Model overfits on low-volatility datasets.** | Medium | Medium | **Mitigation:** The implemented curriculum learning (starting with low-volatility samples) can be augmented with volatility-stratified sampling during batch creation to ensure each batch has a diverse mix of volatility profiles. Early stopping based on validation loss is already implemented and serves as the primary defense. |
| **Transformer expert dominates, making N-BEATS components redundant.** | Low | Medium | **Mitigation:** Use a smaller, more constrained Transformer architecture by default. Implement expert-level dropout or regularization to prevent any single expert from dominating the forecast. The ablation study is designed to detect this issue. |
| **Benchmark results are not reproducible across different environments.** | Low | High | **Mitigation:** The `Dockerfile` provides a fully containerized, reproducible environment. All random seeds (numpy, torch) will be fixed for training and evaluation runs to ensure deterministic results. |
| **Difficulty in finding optimal hyperparameters.** | High | Medium | **Mitigation:** Implement a hyperparameter search script using a library like Optuna or Ray Tune. Start with a small search space focused on the most sensitive parameters (e.g., learning rate, DSOM map size, expert block count). |
