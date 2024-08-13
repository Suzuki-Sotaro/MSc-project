# 以下はknn_change_detectionのコードです。
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class KNNChangeDetection:
    def __init__(self, k=5, alpha=0.01, h=10):
        """
        Initialize the KNNChangeDetection class.
        
        Parameters:
        - k: Number of nearest neighbors to use.
        - alpha: Tail probability threshold for detecting outliers.
        - h: CUSUM-like threshold for triggering a change.
        """
        self.k = k
        self.alpha = alpha
        self.h = h
        self.neighbors = None

    def fit(self, reference_data):
        """
        Fit the k-NN model to the reference data.
        
        Parameters:
        - reference_data: Data to use as the reference for k-NN (array-like, shape = [n_samples, n_features]).
        """
        self.neighbors = NearestNeighbors(n_neighbors=self.k)
        self.neighbors.fit(reference_data)
        print(f"k-NN model fitted with k={self.k} using reference data.")

    def detect_changes(self, test_data):
        """
        Detect changes in the test data using k-NN and a CUSUM-like algorithm.
        
        Parameters:
        - test_data: Data to be tested for changes (array-like, shape = [n_samples, n_features]).
        
        Returns:
        - change_points: Indices in the test_data where changes are detected.
        """
        if self.neighbors is None:
            raise ValueError("Model not fitted. Call fit() with reference data before calling detect_changes().")
        
        distances, _ = self.neighbors.kneighbors(test_data)
        tail_probabilities = np.mean(distances, axis=1)
        
        # Calculate the threshold for detecting outliers
        threshold = np.percentile(tail_probabilities, 100 * (1 - self.alpha))
        
        # Apply a CUSUM-like algorithm to detect changes
        cusum = np.cumsum(tail_probabilities - threshold)
        
        # Debugging: Print some information about cusum and threshold
        print(f"Threshold for detecting outliers: {threshold}")
        print(f"Cumulative Sum (first 10 values): {cusum[:10]}")
        print(f"Tail Probabilities (first 10 values): {tail_probabilities[:10]}")
        print(f"Distances (first 10 sets): {distances[:10]}")
        
        change_points = np.where(cusum > self.h)[0]
        
        # If no changes detected, print a warning
        if len(change_points) == 0:
            print("Warning: No changes detected.")
        
        return change_points

# Example usage:
if __name__ == "__main__":
    from data_loader import DataLoader
    
    file_path = './data/LMP.csv'
    data_loader = DataLoader(file_path)
    
    data_loader.load_data()
    data_loader.preprocess_data()
    data = data_loader.get_data()

    # Use the first half of the data as reference and the second half as test
    reference_data = data.iloc[:427, 2:].values  # Assuming the first two columns are 'Week' and 'Label'
    test_data = data.iloc[427:, 2:].values

    # Initialize and fit the k-NN change detection model
    knn_detector = KNNChangeDetection(k=5, alpha=0.01, h=1)  # Lower h to see if it helps detect changes
    knn_detector.fit(reference_data)

    # Detect changes in the test data
    change_points = knn_detector.detect_changes(test_data)
    
    # Output the results
    print(f"Changes detected at the following indices: {change_points}")
