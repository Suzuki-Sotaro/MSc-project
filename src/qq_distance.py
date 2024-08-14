# 以下はqq_distance.pyの内容です。
import numpy as np
from typing import List, Tuple
def calculate_qq_distance(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Calculate the Q-Q distance between two datasets.
    
    Args:
    reference (np.ndarray): Reference dataset.
    test (np.ndarray): Test dataset.
    
    Returns:
    float: Q-Q distance between the two datasets.
    """
    # Ensure both arrays have the same length
    min_length = min(len(reference), len(test))
    reference = reference[:min_length]
    test = test[:min_length]
    
    # Sort both datasets
    reference_sorted = np.sort(reference)
    test_sorted = np.sort(test)
    
    # Calculate Q-Q distance
    qq_distance = np.sqrt(2) / 2 * np.mean(np.abs(reference_sorted - test_sorted))
    
    return qq_distance

def sliding_window_qq_distance(data: np.ndarray, reference_window_size: int, test_window_size: int) -> List[float]:
    """
    Calculate Q-Q distances using sliding windows.
    
    Args:
    data (np.ndarray): Input time series data.
    reference_window_size (int): Size of the reference window.
    test_window_size (int): Size of the test window.
    
    Returns:
    List[float]: List of Q-Q distances.
    """
    qq_distances = []
    
    for i in range(len(data) - reference_window_size - test_window_size + 1):
        reference = data[i:i+reference_window_size]
        test = data[i+reference_window_size:i+reference_window_size+test_window_size]
        
        qq_distance = calculate_qq_distance(reference, test)
        qq_distances.append(qq_distance)
    
    return qq_distances

def adaptive_reference_window(data: np.ndarray, initial_reference_size: int, test_window_size: int, update_frequency: int) -> List[float]:
    """
    Calculate Q-Q distances using an adaptive reference window.
    
    Args:
    data (np.ndarray): Input time series data.
    initial_reference_size (int): Initial size of the reference window.
    test_window_size (int): Size of the test window.
    update_frequency (int): How often to update the reference window.
    
    Returns:
    List[float]: List of Q-Q distances.
    """
    qq_distances = []
    reference = data[:initial_reference_size]
    
    for i in range(initial_reference_size, len(data) - test_window_size + 1):
        test = data[i:i+test_window_size]
        
        qq_distance = calculate_qq_distance(reference, test)
        qq_distances.append(qq_distance)
        
        # Update reference window
        if i % update_frequency == 0:
            reference = data[i-initial_reference_size:i]
    
    return qq_distances

def detect_change(qq_distances: List[float], threshold: float) -> Tuple[bool, int]:
    """
    Detect change based on cumulative Q-Q distances.
    
    Args:
    qq_distances (List[float]): List of Q-Q distances.
    threshold (float): Threshold for change detection.
    
    Returns:
    Tuple[bool, int]: Whether a change was detected and at which index.
    """
    cumulative_sum = np.cumsum(qq_distances)
    
    if np.any(cumulative_sum > threshold):
        change_index = np.where(cumulative_sum > threshold)[0][0]
        return True, change_index
    
    return False, -1

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(2, 1, 500)])
    
    reference_window_size = 100
    test_window_size = 50
    
    qq_distances = sliding_window_qq_distance(data, reference_window_size, test_window_size)
    
    change_detected, change_index = detect_change(qq_distances, threshold=5)
    
    print(f"Change detected: {change_detected}")
    if change_detected:
        print(f"Change index: {change_index}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title("Input Data")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(qq_distances)
    plt.title("Q-Q Distances")
    plt.show()