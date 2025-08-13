import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import STL
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    Provides a standardized pipeline for preprocessing time-series data,
    as described in PRD section 3.2.
    """
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.scaler = None
        self.trend = None

    def impute_missing(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Handles missing values in the dataframe."""
        if method == 'forward_fill':
            return data.ffill().bfill() # Forward fill then backfill for any remaining NaNs at the start
        elif method == 'linear':
            return data.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown imputation method: {method}")

    def remove_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Removes outliers from the data. Note: This can be destructive."""
        if method != 'iqr':
            raise ValueError("Only 'iqr' method is currently supported for outlier removal.")

        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Cap the outliers instead of removing rows to maintain time sequence
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        return data

    def detrend(self, data: pd.DataFrame, method: str = 'stl', period: int = 24) -> tuple[pd.DataFrame, pd.Series]:
        """Detrends the time series data."""
        if data.shape[1] > 1:
            # Note: STL works on a single series. For multivariate data,
            # we detrend the first column as a simplification.
            # A more complex approach would be needed for true multivariate detrending.
            target_series = data.iloc[:, 0]
        else:
            target_series = data.iloc[:, 0]

        if method == 'stl':
            stl = STL(target_series, period=period, robust=True)
            res = stl.fit()
            detrended_series = target_series - res.trend
            trend = res.trend

            # Create a new dataframe with the detrended series
            detrended_data = data.copy()
            detrended_data.iloc[:, 0] = detrended_series
            return detrended_data, trend
        else:
            raise ValueError("Only 'stl' detrending is currently supported.")

    def scale(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scales the data using the specified method."""
        if method not in self.scalers:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scaler = self.scalers[method]
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    def create_sequences(self, data: np.ndarray, lookback: int, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
        """Creates sequences of lookback and horizon windows."""
        X, y = [], []
        num_sequences = (len(data) - lookback - horizon) // stride + 1
        for i in range(num_sequences):
            start_idx = i * stride
            end_idx = start_idx + lookback
            label_end_idx = end_idx + horizon

            X.append(data[start_idx:end_idx])
            y.append(data[end_idx:label_end_idx])

        return np.array(X), np.array(y)

    def process(self, data: pd.DataFrame, config) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the full preprocessing pipeline on the data.
        """
        logging.info("Starting data preprocessing...")

        # 1. Handle missing values
        impute_method = config.get('impute_method', 'forward_fill')
        logging.info(f"Step 1: Imputing missing values using method: {impute_method}")
        data = self.impute_missing(data, method=impute_method)

        # 2. Remove outliers (optional)
        if config.get('remove_outliers', False):
            logging.info("Step 2: Removing outliers using IQR method.")
            data = self.remove_outliers(data, method='iqr', threshold=config.get('outlier_threshold', 1.5))

        # 3. Detrending (optional)
        if config.get('detrend', False):
            logging.info("Step 3: Detrending data using STL.")
            data, self.trend = self.detrend(data, method='stl', period=config.get('stl_period', 24))

        # 4. Scaling
        scale_method = config.get('scale_method', 'standard')
        logging.info(f"Step 4: Scaling data using method: {scale_method}")
        data = self.scale(data, method=scale_method)

        # 5. Create sequences
        logging.info(f"Step 5: Creating sequences with lookback={config['lookback']}, horizon={config['horizon']}.")
        X, y = self.create_sequences(
            data.values,
            lookback=config['lookback'],
            horizon=config['horizon'],
            stride=config.get('stride', 1)
        )
        logging.info(f"Preprocessing complete. Created {len(X)} samples.")
        # We typically predict the first feature.
        return X, y[:, :, 0]
