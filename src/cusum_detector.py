# 以下はcusum_detector.pyの内容
import numpy as np
from typing import List, Tuple
from qq_distance import calculate_qq_distance, sliding_window_qq_distance

class CUSUMDetector:
    def __init__(self, reference_window_size: int, test_window_size: int, threshold: float = None):
        self.reference_window_size = reference_window_size
        self.test_window_size = test_window_size
        self.threshold = threshold
        self.cusum = 0
        self.k = 0.005  # さらに小さくする
        self.cusum_values = []

    def update(self, qq_distance: float) -> float:
        self.cusum = max(0, self.cusum + qq_distance - self.k)
        self.cusum_values.append(self.cusum)
        return self.cusum

    def detect(self, data: np.ndarray) -> Tuple[bool, int, List[float]]:
        qq_distances = sliding_window_qq_distance(data, self.reference_window_size, self.test_window_size)
        
        for i, qq_distance in enumerate(qq_distances):
            self.update(qq_distance)
            if self.cusum > self.threshold:
                return True, i + self.reference_window_size, self.cusum_values
        
        return False, -1, self.cusum_values

    def learn_threshold(self, training_data: np.ndarray, false_alarm_rate: float = 0.01) -> float:
        """
        Learn the detection threshold from training data.

        Args:
        training_data (np.ndarray): Training data assumed to contain no changes.
        false_alarm_rate (float): Desired false alarm rate.

        Returns:
        float: Learned threshold.
        """
        qq_distances = sliding_window_qq_distance(training_data, self.reference_window_size, self.test_window_size)
        
        cusums = []
        cusum = 0
        for qq_distance in qq_distances:
            cusum = max(0, cusum + qq_distance)
            cusums.append(cusum)
        
        self.threshold = np.percentile(cusums, (1 - false_alarm_rate) * 100)
        return self.threshold

def multi_channel_cusum_detector(data: np.ndarray, reference_window_size: int, test_window_size: int, 
                                 threshold: float = None, false_alarm_rate: float = 0.01) -> List[Tuple[bool, int]]:
    """
    Perform CUSUM detection on multiple channels.

    Args:
    data (np.ndarray): Multi-channel input data. Shape: (n_channels, n_samples)
    reference_window_size (int): Size of the reference window.
    test_window_size (int): Size of the test window.
    threshold (float): Detection threshold. If None, it will be learned from data.
    false_alarm_rate (float): Desired false alarm rate for threshold learning.

    Returns:
    List[Tuple[bool, int]]: List of detection results for each channel.
    """
    n_channels = data.shape[0]
    detectors = [CUSUMDetector(reference_window_size, test_window_size, threshold) for _ in range(n_channels)]
    results = []

    for i, channel_data in enumerate(data):
        if detectors[i].threshold is None:
            # Use the first half of the data for threshold learning
            training_data = channel_data[:len(channel_data)//2]
            detectors[i].learn_threshold(training_data, false_alarm_rate)
        
        change_detected, change_index = detectors[i].detect(channel_data)
        results.append((change_detected, change_index))

    return results

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_channels = 3
    n_samples = 1000
    
    # Generate example data with a change point at sample 500
    data = np.random.normal(0, 1, (n_channels, n_samples))
    data[:, 500:] += 1  # Add a mean shift of 1 to all channels after sample 500
    
    reference_window_size = 100
    test_window_size = 50
    
    results = multi_channel_cusum_detector(data, reference_window_size, test_window_size)
    
    for i, (change_detected, change_index) in enumerate(results):
        print(f"Channel {i}:")
        print(f"  Change detected: {change_detected}")
        if change_detected:
            print(f"  Change index: {change_index}")
        print()