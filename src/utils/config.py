import yaml
from argparse import Namespace

def load_config(config_path: str) -> Namespace:
    """
    Loads a YAML configuration file and returns it as a Namespace object
    for easy attribute access (e.g., config.training.lr_forecast).

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Namespace: A nested Namespace object representing the configuration.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        """Recursively converts a dictionary to a Namespace."""
        if not isinstance(d, dict):
            return d
        namespace = Namespace()
        for key, value in d.items():
            namespace.__setattr__(key, dict_to_namespace(value))
        return namespace

    return dict_to_namespace(config_dict)

if __name__ == '__main__':
    # Example usage:
    # This demonstrates how to load the config and access its values.
    # To run this, execute `python -m src.utils.config` from the root directory.

    # Adjust the path to be relative to the root of the project
    # assuming this script is run from the root directory as a module.
    project_root_config = 'config.yml'

    try:
        config = load_config(project_root_config)

        # --- Accessing values ---
        print("--- Configuration Loaded ---")
        print(f"Lookback: {config.lookback}")
        print(f"Horizon: {config.horizon}")

        # Accessing nested values
        print("\n--- SOM Parameters ---")
        print(f"Map Size: {config.som.map_size}")
        print(f"Tau: {config.som.tau}")

        print("\n--- Expert Parameters ---")
        print(f"Trend Polynomial Degree: {config.experts.trend.poly_degree}")
        print(f"Seasonality Fourier Terms: {config.experts.seasonality.fourier_terms}")

        print("\n--- Training Parameters ---")
        print(f"Forecast LR: {config.training.lr_forecast}")
        print(f"SOM LR: {config.training.lr_som}")
        print(f"Log Interval: {config.training.log_interval}")

        print("\n--- Checkpoint Parameters ---")
        print(f"Checkpoint Directory: {config.checkpoint.save_dir}")

    except FileNotFoundError:
        print(f"Error: The configuration file was not found at '{project_root_config}'.")
        print("Please ensure the file exists and that you are running this script from the project's root directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
