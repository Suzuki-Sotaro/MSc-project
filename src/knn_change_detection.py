# 以下はknn_change_detectionのコードです。
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import chi2

class KNNChangeDetector:
    def __init__(self, k=5, alpha=0.05, h=10, d=1):
        self.k = k
        self.alpha = alpha
        self.h = h
        self.d = d
        self.reference_window = None
        self.cumulative_sum = 0

    def euclidean_distance(self, x, y):
        return euclidean(x.flatten(), y.flatten())

    def compute_knn_distances(self, point, window):
        distances = [self.euclidean_distance(point, x) for x in window]
        return sorted(distances)[1:self.k+1]  # Exclude the point itself
    
    def compute_tail_prob(self, distance, ref_distances):
        chi2_statistic = (self.k * distance**2) / np.mean(ref_distances)**2
        return 1 - chi2.cdf(chi2_statistic, df=2*self.k)

    def update_reference_window(self, new_data):
        new_data = new_data.reshape(1, -1)
        if self.reference_window is None:
            self.reference_window = new_data
        else:
            self.reference_window = np.vstack([self.reference_window[1:], new_data])

    def detect_change(self, new_data):
        new_data = new_data.reshape(1, -1)
        if self.reference_window is None or len(self.reference_window) < self.k + 1:
            self.update_reference_window(new_data)
            return False

        knn_distances = self.compute_knn_distances(new_data, self.reference_window)
        ref_distances = [self.compute_knn_distances(x.reshape(1, -1), self.reference_window) for x in self.reference_window]
        ref_distances = np.mean(ref_distances, axis=0)

        tail_prob = self.compute_tail_prob(np.mean(knn_distances), ref_distances)
        s_t = np.log(self.alpha / tail_prob) if tail_prob > 0 else 0
        
        self.cumulative_sum = max(0, self.cumulative_sum + s_t)

        change_detected = self.cumulative_sum > self.h

        if not change_detected:
            self.update_reference_window(new_data)

        return change_detected

    def reset(self):
        self.reference_window = None
        self.cumulative_sum = 0

def sliding_window(data, window_size, d):
    for i in range(len(data) - window_size + 1):
        yield data[i:i+window_size].reshape(-1, d)

def detect_changes(data, window_size, k=5, alpha=0.05, h=10, d=1):
    detector = KNNChangeDetector(k=k, alpha=alpha, h=h, d=d)
    change_points = []

    for i, window in enumerate(sliding_window(data, window_size, d)):
        if detector.detect_change(window[-1]):  # 最新のデータポイントのみを渡す
            change_points.append(i + window_size - 1)
            detector.reset()

    return change_points

if __name__ == "__main__":
    # Test the KNN Change Detector
    from data_loader import DataLoader

    loader = DataLoader()
    loader.load_data()
    bus_data = loader.get_bus_data(115)['Bus115'].values

    window_size = 10  # Example window size
    k = 5
    alpha = 0.01
    h = 20

    change_points = detect_changes(bus_data, window_size, k, alpha, h)
    print(f"Detected change points: {change_points}")