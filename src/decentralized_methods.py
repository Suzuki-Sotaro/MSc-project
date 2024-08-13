# 以下はdecentralized_methods.pyの内容
import numpy as np
from knn_change_detection import KNNChangeDetector

class DecentralizedChangeDetector:
    def __init__(self, num_buses, k=5, alpha=0.05, h=10):
        self.num_buses = num_buses
        self.detectors = [KNNChangeDetector(k=k, alpha=alpha, h=h) for _ in range(num_buses)]
        self.h = h

    def reset(self):
        for detector in self.detectors:
            detector.reset()

class MethodA(DecentralizedChangeDetector):
    def __init__(self, num_buses, k=5, alpha=0.05, h=10, p=0.5):
        super().__init__(num_buses, k, alpha, h)
        self.p = p

    def detect_change(self, data):
        votes = [int(self.detectors[i].detect_change(data[i].reshape(1, -1))) for i in range(self.num_buses)]
        return sum(votes) >= self.p * self.num_buses

class MethodB(DecentralizedChangeDetector):
    def __init__(self, num_buses, k=5, alpha=0.05, h=10, aggregation_method='average'):
        super().__init__(num_buses, k, alpha, h)
        self.aggregation_method = aggregation_method

    def detect_change(self, data):
        statistics = [self.detectors[i].cumulative_sum for i in range(self.num_buses)]
        
        if self.aggregation_method == 'average':
            aggregated_statistic = np.mean(statistics)
        elif self.aggregation_method == 'median':
            aggregated_statistic = np.median(statistics)
        elif self.aggregation_method == 'outlier_removal':
            # Implement MAD-based outlier removal
            median = np.median(statistics)
            mad = np.median(np.abs(statistics - median))
            threshold = 3 * 1.4826 * mad  # Assuming normal distribution
            valid_statistics = [s for s in statistics if abs(s - median) <= threshold]
            aggregated_statistic = np.mean(valid_statistics) if valid_statistics else np.mean(statistics)
        else:
            raise ValueError("Invalid aggregation method")

        return aggregated_statistic > self.h

def experiment_method_a(data, window_size, k, alpha, h, p_values):
    num_buses = data.shape[0]
    results = {}

    for p in p_values:
        detector = MethodA(num_buses, k=k, alpha=alpha, h=h, p=p)
        change_points = []

        for i in range(len(data) - window_size + 1):
            window_data = [data[j, i:i+window_size] for j in range(num_buses)]
            print(f"Window data shape: {np.array(window_data).shape}")  # デバッグ出力
            if detector.detect_change(window_data):
                change_points.append(i + window_size - 1)
                detector.reset()

        results[p] = change_points

    return results

def experiment_method_b(data, window_size, k=5, alpha=0.05, h=10, methods=['average', 'median', 'outlier_removal']):
    num_buses = data.shape[1]
    results = {}

    for method in methods:
        detector = MethodB(num_buses, k=k, alpha=alpha, h=h, aggregation_method=method)
        change_points = []

        for i in range(len(data) - window_size + 1):
            window_data = [data[i:i+window_size, j] for j in range(num_buses)]
            if detector.detect_change(window_data):
                change_points.append(i + window_size - 1)
                detector.reset()

        results[method] = change_points

    return results

if __name__ == "__main__":
    from data_loader import DataLoader

    # Load and preprocess data
    loader = DataLoader()
    loader.load_data()
    all_bus_data = loader.get_all_selected_bus_data()
    
    # Extract numerical data and transpose to have buses as columns
    data = all_bus_data.iloc[:, 2:].values.T  # Exclude 'Week' and 'Label' columns

    window_size = 10  # Example window size
    k = 5
    alpha = 0.01
    h = 20

    # Run experiments
    method_a_results = experiment_method_a(data, window_size, k, alpha, h)
    method_b_results = experiment_method_b(data, window_size, k, alpha, h)

    # Print results
    print("Method A Results:")
    for p, change_points in method_a_results.items():
        print(f"p = {p}: {change_points}")

    print("\nMethod B Results:")
    for method, change_points in method_b_results.items():
        print(f"{method}: {change_points}")