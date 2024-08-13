# 以下はdecentralized_methods.pyの内容
import numpy as np
from scipy.stats import median_abs_deviation
from statistics import median

class DecentralizedDetection:
    def __init__(self, h_values, p_values=[0.1, 0.2, 0.5, 0.7, 0.9]):
        """
        Initialize the DecentralizedDetection class.
        
        Parameters:
        - h_values: List of local thresholds h[k] for each bus.
        - p_values: List of p% values to experiment with in Method A.
        """
        self.h_values = h_values
        self.p_values = p_values

    def method_a_decision_fusion(self, bus_decisions):
        """
        Implement Method A: Decision Fusion
        
        Parameters:
        - bus_decisions: A 2D array where each row represents a bus and each column a time step, 
                         with 1 indicating a detected change and 0 indicating no change.
                         
        Returns:
        - results: A dictionary where each p value is a key, and the corresponding detection times and accuracy metrics are stored as values.
        """
        num_buses, num_steps = bus_decisions.shape
        results = {}

        for p in self.p_values:
            detection_times = []
            for t in range(num_steps):
                if np.sum(bus_decisions[:, t]) >= p * num_buses:
                    detection_times.append(t)
                    break  # Stop once a change is detected
            
            detection_time = detection_times[0] if detection_times else None
            results[p] = detection_time
        
        return results

    def method_b_statistical_aggregation(self, bus_statistics):
        """
        Implement Method B: Statistical Aggregation
        
        Parameters:
        - bus_statistics: A 2D array where each row represents a bus and each column a time step,
                          with each value representing a bus's statistic s[k] at that time step.
                          
        Returns:
        - results: A dictionary where different aggregation methods are keys, and the corresponding detection times and accuracy metrics are stored as values.
        """
        num_buses, num_steps = bus_statistics.shape
        aggregation_methods = ['average', 'median', 'outlier_detection']
        results = {}

        for method in aggregation_methods:
            aggregated_values = np.zeros(num_steps)

            if method == 'average':
                aggregated_values = np.mean(bus_statistics, axis=0)
            elif method == 'median':
                aggregated_values = np.median(bus_statistics, axis=0)
            elif method == 'outlier_detection':
                for t in range(num_steps):
                    current_statistics = bus_statistics[:, t]
                    mad_value = median_abs_deviation(current_statistics)
                    median_value = median(current_statistics)
                    non_outliers = [s for s in current_statistics if abs(s - median_value) <= mad_value]
                    aggregated_values[t] = np.mean(non_outliers) if non_outliers else median_value

            detection_times = []
            for t in range(num_steps):
                if aggregated_values[t] > np.mean(self.h_values):
                    detection_times.append(t)
                    break  # Stop once a change is detected

            detection_time = detection_times[0] if detection_times else None
            results[method] = detection_time

        return results

# Example usage:
if __name__ == "__main__":
    # Example synthetic data for testing
    h_values = [10, 15, 20, 25, 30, 35, 40, 45]
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]

    # Synthetic bus decision data for Method A (8 buses, 100 time steps)
    bus_decisions = np.random.randint(0, 2, (8, 100))
    
    # Synthetic bus statistics data for Method B (8 buses, 100 time steps)
    bus_statistics = np.random.rand(8, 100) * 50

    # Initialize and run Method A
    detector = DecentralizedDetection(h_values, p_values)
    method_a_results = detector.method_a_decision_fusion(bus_decisions)
    print("Method A Results (Decision Fusion):")
    print(method_a_results)

    # Initialize and run Method B
    method_b_results = detector.method_b_statistical_aggregation(bus_statistics)
    print("Method B Results (Statistical Aggregation):")
    print(method_b_results)
