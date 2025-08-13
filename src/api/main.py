from fastapi import FastAPI, HTTPException
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.nbeats import DSOM_NBEATS
from src.utils.config import load_config
from src.api.schemas import PredictionInput, PredictionOutput

app = FastAPI(
    title="DSOM-BEATS Forecasting API",
    description="An API for making time-series forecasts using the DSOM-BEATS model.",
    version="1.0.0"
)

# --- Model Loading ---
MODEL = None
CONFIG = None

@app.on_event("startup")
def load_model():
    """Load the model and configuration at startup."""
    global MODEL, CONFIG

    config_path = 'config.yml'
    CONFIG = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The API is designed for univariate forecasting, so n_features is 1.
    MODEL = DSOM_NBEATS(CONFIG, n_features=1).to(device)

    # Load a checkpoint
    checkpoint_dir = CONFIG.checkpoint.save_dir
    # For simplicity, we assume a 'best_model.pth' exists.
    # In a real application, you might want more sophisticated logic to select a checkpoint.
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            MODEL.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint from {checkpoint_path}: {e}")
    else:
        # This is a fallback for when no trained model is available.
        # In a production scenario, you would likely want to prevent the API from starting.
        print(f"WARNING: No checkpoint found at {checkpoint_path}. The model is not trained.")


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Make a prediction for a given time-series sequence.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    # Validate input length
    if len(input_data.data) != CONFIG.lookback:
        raise HTTPException(
            status_code=400,
            detail=f"Input data must have a length of {CONFIG.lookback}, but got {len(input_data.data)}."
        )

    # Convert to tensor and add batch and feature dimensions
    input_tensor = torch.tensor(input_data.data).float().unsqueeze(0).unsqueeze(-1)

    # Make prediction
    with torch.no_grad():
        forecast, _ = MODEL(input_tensor)

    # Return forecast
    return PredictionOutput(forecast=forecast.squeeze(0).tolist())

@app.get("/")
def read_root():
    return {"message": "Welcome to the DSOM-BEATS Forecasting API. Go to /docs for more info."}
