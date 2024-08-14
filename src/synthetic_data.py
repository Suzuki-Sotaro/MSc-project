import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
import config
import logging

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, filename=config.LOG_FILE, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_ar1_process(n_samples: int, mean: float, std_dev: float, theta: float) -> np.ndarray:
    """
    Generate an AR(1) process.

    Args:
    n_samples (int): Number of samples to generate
    mean (float): Mean of the process
    std_dev (float): Standard deviation of the noise
    theta (float): AR(1) coefficient

    Returns:
    np.ndarray: Generated AR(1) process
    """
    print(f"Generating AR(1) process: n_samples={n_samples}, mean={mean}, std_dev={std_dev}, theta={theta}")
    x = np.zeros(n_samples)
    x[0] = mean
    for t in range(1, n_samples):
        x[t] = x[t-1] + (1-theta)*(mean-x[t-1]) + np.random.normal(0, std_dev)
    print(f"AR(1) process generated. Shape: {x.shape}")
    return x

def generate_synthetic_data(n_samples: int, n_dims: int, change_points: List[int],
                            means: List[float], std_devs: List[float], thetas: List[float]) -> Tuple[np.ndarray, List[int]]:
    """
    Generate synthetic multivariate data with change points.

    Args:
    n_samples (int): Number of samples to generate
    n_dims (int): Number of dimensions
    change_points (List[int]): List of change points
    means (List[float]): List of mean values for each segment
    std_devs (List[float]): List of standard deviations for each segment
    thetas (List[float]): List of AR(1) coefficients for each segment

    Returns:
    Tuple[np.ndarray, List[int]]: Generated data and list of change points
    """
    print(f"Generating synthetic data: n_samples={n_samples}, n_dims={n_dims}")
    data = np.zeros((n_samples, n_dims))
    
    for dim in range(n_dims):
        print(f"Generating dimension {dim+1}/{n_dims}")
        start = 0
        for i, cp in enumerate(change_points + [n_samples]):
            end = cp
            segment_data = generate_ar1_process(end - start, means[i], std_devs[i], thetas[i])
            data[start:end, dim] = segment_data
            start = end
    
    print(f"Synthetic data generated. Shape: {data.shape}")
    return data, change_points

def add_outliers(data: np.ndarray, outlier_ratio: float, outlier_scale: float) -> np.ndarray:
    """
    Add random outliers to the data.

    Args:
    data (np.ndarray): Input data
    outlier_ratio (float): Ratio of outliers to add
    outlier_scale (float): Scale factor for outliers

    Returns:
    np.ndarray: Data with added outliers
    """
    print(f"Adding outliers: outlier_ratio={outlier_ratio}, outlier_scale={outlier_scale}")
    n_samples, n_dims = data.shape
    n_outliers = int(n_samples * outlier_ratio)
    
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    outlier_dims = np.random.choice(n_dims, n_outliers)
    
    data_with_outliers = data.copy()
    data_with_outliers[outlier_indices, outlier_dims] += np.random.randn(n_outliers) * outlier_scale
    
    print(f"Outliers added. Number of outliers: {n_outliers}")
    return data_with_outliers

def create_synthetic_dataset() -> Tuple[pd.DataFrame, List[int]]:
    print("Creating synthetic dataset...")
    n_samples = 8760  # One year of hourly data
    n_dims = len(config.SELECTED_BUSES)
    change_points = [2160, 4320, 6480]  # Changes every 3 months
    means = [25, 30, 20, 35]
    std_devs = [2, 3, 2.5, 3.5]
    thetas = [0.7, 0.6, 0.8, 0.5]
    
    data, true_changes = generate_synthetic_data(n_samples, n_dims, change_points, means, std_devs, thetas)
    data_with_outliers = add_outliers(data, outlier_ratio=0.01, outlier_scale=5)
    
    # Create DataFrame
    df = pd.DataFrame(data_with_outliers, columns=[f'Bus{i}' for i in config.SELECTED_BUSES])
    
    # Generate 'Week' column
    weeks = np.repeat(np.arange(1, 53), 168)  # 168 hours per week
    weeks = np.tile(weeks, n_samples // len(weeks) + 1)[:n_samples]  # Ensure correct length
    df['Week'] = weeks
    
    # Generate 'Label' column
    df['Label'] = np.zeros(n_samples)
    for i, cp in enumerate(change_points):
        df.loc[cp:, 'Label'] = i % 2  # Alternate between 0 and 1
    
    print("Synthetic dataset created.")
    print(f"DataFrame shape: {df.shape}")
    print(f"True change points: {true_changes}")
    
    return df, true_changes

def plot_synthetic_data(df: pd.DataFrame, true_changes: List[int]):
    """
    Plot the synthetic data.

    Args:
    df (pd.DataFrame): Synthetic dataset
    true_changes (List[int]): List of true change points
    """
    print("Plotting synthetic data...")
    plt.figure(figsize=(15, 10))
    for col in df.columns:
        if col.startswith('Bus'):
            plt.plot(df[col], label=col, alpha=0.7)
    
    for cp in true_changes:
        plt.axvline(x=cp, color='r', linestyle='--')
    
    plt.title('Synthetic LMP Data')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"{config.RESULTS_DIR}/synthetic_data_plot.png")
    plt.close()
    print(f"Synthetic data plot saved as {config.RESULTS_DIR}/synthetic_data_plot.png")

if __name__ == "__main__":
    print("Running synthetic data generation...")
    np.random.seed(config.RANDOM_SEED)
    
    # Create synthetic dataset
    df, true_changes = create_synthetic_dataset()
    
    # Display summary statistics
    print("\nSummary statistics of synthetic data:")
    print(df.describe())
    
    # Plot synthetic data
    plot_synthetic_data(df, true_changes)
    
    # Save synthetic data to CSV
    df.to_csv(f"{config.RESULTS_DIR}/synthetic_lmp_data.csv", index=False)
    print(f"Synthetic data saved to {config.RESULTS_DIR}/synthetic_lmp_data.csv")
    
    print("Synthetic data generation completed.")