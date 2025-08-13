import os
import pandas as pd
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class BenchmarkDatasetManager:
    """
    Manages downloading and loading of benchmark time-series datasets.
    """

    _DATASET_URLS = {
        "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    }

    def __init__(self, data_dir: str = 'data/'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_dataset(self, dataset_name: str):
        """
        Downloads a dataset if it doesn't already exist.

        Args:
            dataset_name (str): The name of the dataset (e.g., 'ETTh1').
        """
        if dataset_name not in self._DATASET_URLS:
            raise ValueError(f"Dataset '{dataset_name}' is not supported. Supported datasets are: {list(self._DATASET_URLS.keys())}")

        url = self._DATASET_URLS[dataset_name]
        filename = f"{dataset_name}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            logging.info(f"Downloading {dataset_name} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                logging.info(f"Successfully downloaded {filename} to {self.data_dir}")
            except Exception as e:
                logging.error(f"Failed to download {dataset_name}: {e}")
                raise
        else:
            logging.info(f"Dataset '{dataset_name}' already exists at {filepath}.")

        return filepath

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Loads a benchmark dataset into a pandas DataFrame.

        Args:
            dataset_name (str): The name of the dataset to load.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        filepath = self.download_dataset(dataset_name)
        df = pd.read_csv(filepath)
        logging.info(f"Loaded '{dataset_name}' with shape {df.shape}")
        return df
