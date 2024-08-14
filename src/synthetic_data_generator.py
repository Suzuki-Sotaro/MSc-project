# 以下はsynthetic_data_generator.pyの内容です。
import numpy as np
from typing import Tuple

def generate_ar1_process(n_samples: int, mean: float, variance: float, theta: float, 
                         change_point: int = None, new_mean: float = None, new_variance: float = None) -> np.ndarray:
    x = np.zeros(n_samples)
    x[0] = mean
    
    for t in range(1, n_samples):
        if change_point is not None and t >= change_point:
            current_mean = new_mean if new_mean is not None else mean
            current_variance = new_variance if new_variance is not None else variance
            
            # Add a sudden change followed by a gradual change
            if t == change_point:
                x[t] = current_mean + np.random.normal(0, np.sqrt(current_variance))
            else:
                transition = min(1, (t - change_point) / 100)  # 100 samples for full transition
                current_mean = mean + transition * (current_mean - mean)
                current_variance = variance + transition * (current_variance - variance)
        else:
            current_mean = mean
            current_variance = variance
        
        e = np.random.normal(0, np.sqrt(current_variance))
        x[t] = theta * x[t-1] + (1 - theta) * current_mean + e
    
    return x

def generate_multi_channel_data(n_channels: int, n_samples: int, mean: float, variance: float, 
                                theta: float, change_point: int = None, new_mean: float = None, 
                                new_variance: float = None) -> np.ndarray:
    """
    Generate multi-channel AR(1) processes with optional change point.

    Args:
    n_channels (int): Number of channels to generate.
    n_samples (int): Number of samples per channel.
    mean (float): Mean of the processes before change.
    variance (float): Variance of the processes before change.
    theta (float): AR(1) coefficient, should be between 0 and 1.
    change_point (int): Index of the change point. If None, no change occurs.
    new_mean (float): Mean of the processes after change. If None, no change in mean.
    new_variance (float): Variance of the processes after change. If None, no change in variance.

    Returns:
    np.ndarray: Generated multi-channel AR(1) processes. Shape: (n_channels, n_samples)
    """
    data = np.zeros((n_channels, n_samples))
    
    for i in range(n_channels):
        data[i] = generate_ar1_process(n_samples, mean, variance, theta, change_point, new_mean, new_variance)
    
    return data

def generate_dataset_with_labels(n_channels: int, n_samples: int, mean: float, variance: float, 
                                 theta: float, change_point: int, new_mean: float, 
                                 new_variance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a labeled dataset with multi-channel AR(1) processes and change point.

    Args:
    n_channels (int): Number of channels to generate.
    n_samples (int): Number of samples per channel.
    mean (float): Mean of the processes before change.
    variance (float): Variance of the processes before change.
    theta (float): AR(1) coefficient, should be between 0 and 1.
    change_point (int): Index of the change point.
    new_mean (float): Mean of the processes after change.
    new_variance (float): Variance of the processes after change.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Generated data and labels.
    """
    data = generate_multi_channel_data(n_channels, n_samples, mean, variance, theta, 
                                       change_point, new_mean, new_variance)
    
    labels = np.zeros(n_samples)
    labels[change_point:] = 1
    
    return data, labels

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n_channels = 8
    n_samples = 1000
    mean = 0
    variance = 1
    theta = 0.5
    change_point = 500
    new_mean = 2
    new_variance = 1.5
    
    data, labels = generate_dataset_with_labels(n_channels, n_samples, mean, variance, 
                                                theta, change_point, new_mean, new_variance)
    
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    
    # Plot the first channel
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(data[0])
    plt.axvline(x=change_point, color='r', linestyle='--', label='Change point')
    plt.title("First channel of generated data")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
    # Plot the labels
    plt.figure(figsize=(12, 3))
    plt.step(range(n_samples), labels)
    plt.title("Labels")
    plt.xlabel("Sample")
    plt.ylabel("Label")
    plt.yticks([0, 1])
    plt.show()