import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List
import matplotlib.pyplot as plt

def gem_change_detection(data: np.ndarray, k: int, alpha: float, h: float) -> Tuple[List[int], List[float]]:
    n_samples, n_features = data.shape
    split_point = n_samples // 2
    reference_set = data[:split_point]
    test_set = data[split_point:]
    
    distances = cdist(reference_set, reference_set)
    knn_distances = np.sort(distances, axis=1)[:, 1:k+1]
    knn_distances_sum = np.sum(knn_distances, axis=1)
    threshold = np.percentile(knn_distances_sum, (1-alpha)*100)
    
    change_points = []
    decision_stats = []
    g_t = 0
    
    for t, sample in enumerate(test_set):
        sample_distances = cdist([sample], reference_set)[0]
        sample_knn_distance = np.sum(np.sort(sample_distances)[:k])
        
        p_value = np.mean(sample_knn_distance <= knn_distances_sum)
        s_t = np.log(alpha / max(p_value, 1e-10))  # p_valueが0になるのを防ぐ
        g_t = max(0, g_t + s_t)
        decision_stats.append(g_t)
        
        if g_t >= h:
            change_points.append(t + split_point)
            g_t = 0  # 変化点を検出したらg_tをリセット
    
    return change_points, decision_stats

def qq_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Q-Q distance between two samples.
    
    Args:
    x (np.ndarray): First sample
    y (np.ndarray): Second sample
    
    Returns:
    float: Q-Q distance
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    return np.sqrt(np.mean((x_sorted - y_sorted)**2))

def qq_change_detection(data: np.ndarray, window_size: int, h: float) -> Tuple[List[int], List[float]]:
    """
    Perform change detection using the Q-Q distance method.
    
    Args:
    data (np.ndarray): Input data of shape (n_samples, n_features)
    window_size (int): Size of the sliding window
    h (float): Threshold for change detection
    
    Returns:
    Tuple[List[int], List[float]]: Detected change points and cumulative Q-Q distances
    """
    print("Starting Q-Q distance change detection...")
    n_samples, n_features = data.shape
    print(f"Input data shape: {data.shape}")
    
    reference_window = data[:window_size]
    print(f"Reference window shape: {reference_window.shape}")
    
    change_points = []
    cumulative_distances = [0]
    
    for t in range(window_size, n_samples):
        test_window = data[t-window_size:t]
        qq_dist = qq_distance(reference_window.flatten(), test_window.flatten())
        cumulative_distance = cumulative_distances[-1] + qq_dist
        cumulative_distances.append(cumulative_distance)
        
        if cumulative_distance > h:
            change_points.append(t)
            print(f"Change detected at t={t}, cumulative distance={cumulative_distance:.4f}")
        
        if t % 100 == 0:
            print(f"Processed {t}/{n_samples} samples, current cumulative distance: {cumulative_distance:.4f}")
    
    print("Q-Q distance change detection completed.")
    return change_points, cumulative_distances

def plot_results(data: np.ndarray, change_points: List[int], decision_stats: List[float], method: str):
    """
    Plot the results of change detection.
    
    Args:
    data (np.ndarray): Input data
    change_points (List[int]): Detected change points
    decision_stats (List[float]): Decision statistics or cumulative distances
    method (str): Name of the detection method
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the first feature of the data
    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0])
    for cp in change_points:
        plt.axvline(x=cp, color='r', linestyle='--')
    plt.title(f"{method} - Data and Detected Changes")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Plot the decision statistic or cumulative distance
    plt.subplot(2, 1, 2)
    plt.plot(decision_stats)
    plt.title(f"{method} - Decision Statistic")
    plt.xlabel("Time")
    plt.ylabel("Statistic Value")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    data = np.random.randn(n_samples, n_features)
    
    # Introduce a change point
    change_point = 500
    data[change_point:] += 1
    
    print("Testing GEM change detection...")
    gem_changes, gem_stats = gem_change_detection(data, k=5, alpha=0.1, h=10)
    plot_results(data, gem_changes, gem_stats, "GEM")
    
    print("\nTesting Q-Q distance change detection...")
    qq_changes, qq_stats = qq_change_detection(data, window_size=50, h=5)
    plot_results(data, qq_changes, qq_stats, "Q-Q Distance")