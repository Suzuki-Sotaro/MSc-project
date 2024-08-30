import numpy as np

def calculate_qq_distance(v0, v1):
    """
    Calculate the QQ distance between two data windows.
    
    Args:
    v0, v1 (array-like): Input data windows to compare.
    
    Returns:
    float: QQ distance between v0 and v1.
    """
    s = len(v0)
    v0_sorted = np.sort(v0)
    v1_sorted = np.sort(v1)
    
    quantiles = np.arange(1, s + 1) / s
    qq_distances = np.abs(v1_sorted - v0_sorted)
    
    return np.mean(qq_distances) * np.sqrt(2) / 2

def qq_detection_approach1(data, window_size=24):
    """
    Approach 1: Compare all windows with the first window.
    
    Args:
    data (array-like): Input time series data.
    window_size (int): Size of the sliding window.
    
    Returns:
    array: QQ distances for each window compared to the first window.
    """
    n = len(data)
    detection_statistics = np.zeros(n - window_size + 1)
    
    reference_window = data[:window_size]
    
    for i in range(n - window_size + 1):
        current_window = data[i:i+window_size]
        detection_statistics[i] = calculate_qq_distance(reference_window, current_window)
    
    return detection_statistics

def qq_detection_approach2(data, window_size=24):
    """
    Approach 2: Compare each window with the previous window.
    
    Args:
    data (array-like): Input time series data.
    window_size (int): Size of the sliding window.
    
    Returns:
    array: QQ distances for each window compared to its previous window.
    """
    n = len(data)
    detection_statistics = np.zeros(n - window_size)
    
    for i in range(1, n - window_size + 1):
        prev_window = data[i-1:i+window_size-1]
        current_window = data[i:i+window_size]
        detection_statistics[i-1] = calculate_qq_distance(prev_window, current_window)
    
    return detection_statistics

import matplotlib.pyplot as plt

def analyze_and_visualize(data, window_size=24):
    """
    Analyze the input data using both QQ detection approaches and visualize the results.
    
    Args:
    data (array-like): Input time series data.
    window_size (int): Size of the sliding window.
    """
    stats1 = qq_detection_approach1(data, window_size)
    stats2 = qq_detection_approach2(data, window_size)
    
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Plot results from Approach 1
    plt.subplot(3, 1, 2)
    plt.plot(stats1)
    plt.title('Approach 1: Compare with First Window')
    plt.xlabel('Time')
    plt.ylabel('QQ Distance')
    
    # Plot results from Approach 2
    plt.subplot(3, 1, 3)
    plt.plot(stats2)
    plt.title('Approach 2: Compare with Previous Window')
    plt.xlabel('Time')
    plt.ylabel('QQ Distance')
    
    plt.tight_layout()
    plt.show()

# Generate synthetic data for analysis
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(2, 1, 500)])
analyze_and_visualize(data)