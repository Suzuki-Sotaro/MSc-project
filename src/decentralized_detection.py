# 以下はdecentralized_detection.pyのコード
import numpy as np
from typing import Dict, List, Tuple
from cusum_detector import CUSUMDetector
from qq_distance import sliding_window_qq_distance

def method_a_voting(detector_results: List[Tuple[bool, int]], voting_threshold: float) -> Tuple[bool, int]:
    """
    Implement Method A: Voting-based decentralized detection.

    Args:
    detector_results (List[Tuple[bool, int]]): Detection results from each bus.
    voting_threshold (float): Percentage of buses required to vote for a change.

    Returns:
    Tuple[bool, int]: Whether a change was detected and at which index.
    """
    n_buses = len(detector_results)
    change_votes = [result[0] for result in detector_results]
    
    if sum(change_votes) >= voting_threshold * n_buses:
        change_indices = [result[1] for result in detector_results if result[0]]
        return True, max(change_indices)
    
    return False, -1

def method_b_aggregation(detector_statistics: List[float], aggregation_method: str, threshold: float) -> Tuple[bool, float]:
    if aggregation_method == 'mean':
        aggregated_statistic = np.mean(detector_statistics)
    elif aggregation_method == 'median':
        aggregated_statistic = np.median(detector_statistics)
    elif aggregation_method == 'mad':
        median = np.median(detector_statistics)
        mad = np.median(np.abs(detector_statistics - median))
        aggregated_statistic = mad
    else:
        raise ValueError("Invalid aggregation method")

    return aggregated_statistic > threshold * 0.5, aggregated_statistic 

def decentralized_detection(data: np.ndarray, reference_window_size: int, test_window_size: int, 
                            method: str, threshold: float, voting_threshold: float = 0.5,
                            aggregation_method: str = 'mean') -> Tuple[bool, int, List[float]]:
    """
    Perform decentralized detection on multi-channel data.

    Args:
    data (np.ndarray): Multi-channel input data. Shape: (n_channels, n_samples)
    reference_window_size (int): Size of the reference window.
    test_window_size (int): Size of the test window.
    method (str): Detection method ('A' or 'B').
    threshold (float): Detection threshold.
    voting_threshold (float): Percentage of buses required to vote for a change (for method A).
    aggregation_method (str): Method to aggregate statistics (for method B).

    Returns:
    Tuple[bool, int, List[float]]: Whether a change was detected, at which index, and the detector statistics.
    """
    n_channels = data.shape[0]
    detectors = [CUSUMDetector(reference_window_size, test_window_size, threshold) for _ in range(n_channels)]
    
    if method == 'A':
        detector_results = []
        for i, channel_data in enumerate(data):
            change_detected, change_index = detectors[i].detect(channel_data)
            detector_results.append((change_detected, change_index))
        
        change_detected, change_index = method_a_voting(detector_results, voting_threshold)
        detector_statistics = [detector.cusum for detector in detectors]
        
    elif method == 'B':
        detector_statistics = []
        for i, channel_data in enumerate(data):
            qq_distances = sliding_window_qq_distance(channel_data, reference_window_size, test_window_size)
            cusum = 0
            for qq_distance in qq_distances:
                cusum = max(0, cusum + qq_distance)
            detector_statistics.append(cusum)
        
        change_detected, aggregated_statistic = method_b_aggregation(detector_statistics, aggregation_method, threshold)
        change_index = -1 if not change_detected else len(data[0]) - test_window_size
        
    else:
        raise ValueError("Invalid method. Choose 'A' or 'B'.")

    return change_detected, change_index, detector_statistics

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_channels = 8  # Number of buses
    n_samples = 1000
    
    # Generate example data with a change point at sample 500
    data = np.random.normal(0, 1, (n_channels, n_samples))
    data[:, 500:] += 1  # Add a mean shift of 1 to all channels after sample 500
    
    reference_window_size = 100
    test_window_size = 50
    threshold = 5
    
    # Method A
    change_detected_a, change_index_a, statistics_a = decentralized_detection(
        data, reference_window_size, test_window_size, 'A', threshold, voting_threshold=0.5)
    
    print("Method A results:")
    print(f"Change detected: {change_detected_a}")
    print(f"Change index: {change_index_a}")
    print(f"Detector statistics: {statistics_a}")
    print()
    
    # Method B
    change_detected_b, change_index_b, statistics_b = decentralized_detection(
        data, reference_window_size, test_window_size, 'B', threshold, aggregation_method='mean')
    
    print("Method B results:")
    print(f"Change detected: {change_detected_b}")
    print(f"Change index: {change_index_b}")
    print(f"Detector statistics: {statistics_b}")