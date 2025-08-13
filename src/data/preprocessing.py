import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import STL
import logging
from .datasets import BenchmarkDatasetManager

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
        self.dataset_manager = BenchmarkDatasetManager()

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

    def _split_data(self, data: pd.DataFrame, train_ratio: float, val_ratio: float):
        """Splits the data into training, validation, and test sets."""
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_data = data[:n_train]
        val_data = data[n_train : n_train + n_val]
        test_data = data[n_train + n_val :]

        return train_data, val_data, test_data

    def process(self, data, config) -> dict:
        """
        Runs the full preprocessing pipeline and returns split datasets.
        The `data` argument can be a pandas DataFrame or a string specifying a benchmark dataset.
        """
        if isinstance(data, str):
            logging.info(f"Loading benchmark dataset: {data}")
            data = self.dataset_manager.load_dataset(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Input `data` must be a pandas DataFrame or a supported dataset name string.")

        # Exclude non-numeric columns like 'date' before processing
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if 'date' in data.columns:
            logging.info("Excluding 'date' column from preprocessing.")
            data = data[numeric_cols]

        logging.info("Starting data preprocessing...")

        # 1. Handle missing values
        impute_method = getattr(config, 'impute_method', 'forward_fill')
        logging.info(f"Step 1: Imputing missing values using method: {impute_method}")
        data = self.impute_missing(data, method=impute_method)

        # 2. Remove outliers (optional)
        if getattr(config, 'remove_outliers', False):
            logging.info("Step 2: Removing outliers using IQR method.")
            data = self.remove_outliers(data, method='iqr', threshold=getattr(config, 'outlier_threshold', 1.5))

        # 3. Detrending (optional)
        if getattr(config, 'detrend', False):
            logging.info("Step 3: Detrending data using STL.")
            data, self.trend = self.detrend(data, method='stl', period=getattr(config, 'stl_period', 24))

        # 4. Scaling
        scale_method = getattr(config, 'scale_method', 'standard')
        logging.info(f"Step 4: Scaling data using method: {scale_method}")
        data = self.scale(data, method=scale_method)

        # 5. Split data into train, validation, and test sets
        train_ratio = config.data.splitting.train_ratio
        val_ratio = config.data.splitting.val_ratio
        logging.info(f"Step 5: Splitting data into train/val/test with ratios {train_ratio}/{val_ratio}/{1-train_ratio-val_ratio}")
        train_df, val_df, test_df = self._split_data(data, train_ratio, val_ratio)

        # 6. Create sequences for each set
        datasets = {}
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            logging.info(f"Creating sequences for {split_name} set...")
            X, y = self.create_sequences(
                split_df.values,
                lookback=config.lookback,
                horizon=config.horizon,
                stride=getattr(config, 'stride', 1)
            )
            # We typically predict the first feature.
            datasets[split_name] = (X, y[:, :, 0])
            logging.info(f"Created {len(X)} samples for {split_name} set.")

        logging.info("Preprocessing complete.")
        return datasets
