# 以下はexperiment_runner.pyのコードです。
from data_loader import DataLoader
from knn_change_detection import KNNChangeDetection
from decentralized_methods import DecentralizedDetection
from performance_metrics import PerformanceMetrics
import numpy as np

class ExperimentRunner:
    def __init__(self, file_path, buses, k_values, alpha_values, h_values, p_values):
        """
        Initialize the ExperimentRunner class.
        
        Parameters:
        - file_path: Path to the CSV data file.
        - buses: List of bus column names to use in the experiment.
        - k_values: List of k values for k-NN experiments.
        - alpha_values: List of alpha values for k-NN experiments.
        - h_values: List of h values for both k-NN and decentralized methods.
        - p_values: List of p values for Method A in decentralized methods.
        """
        self.file_path = file_path
        self.buses = buses
        self.k_values = k_values
        self.alpha_values = alpha_values
        self.h_values = h_values
        self.p_values = p_values

    def run_knn_experiment(self):
        # Load and preprocess the data
        data_loader = DataLoader(self.file_path)
        data_loader.load_data()
        data_loader.preprocess_data()
        data = data_loader.get_data()

        # Split data into reference and test sets
        reference_data = data.iloc[:427, 2:].values  # First half for reference
        test_data = data.iloc[427:, 2:].values  # Second half for testing
        true_labels = data['Label'].iloc[427:].values  # True labels for the test set

        results = {}

        for k in self.k_values:
            for alpha in self.alpha_values:
                for h in self.h_values:
                    # Initialize and fit the k-NN model
                    knn_detector = KNNChangeDetection(k=k, alpha=alpha, h=h)
                    knn_detector.fit(reference_data)

                    # Detect changes
                    change_points = knn_detector.detect_changes(test_data)
                    predicted_labels = np.zeros_like(true_labels)
                    if len(change_points) > 0:
                        predicted_labels[change_points] = 1  # Mark detected change points as 1

                    # Calculate performance metrics
                    metrics_calculator = PerformanceMetrics(true_labels, predicted_labels, change_points)
                    classification_metrics = metrics_calculator.calculate_classification_metrics()
                    detection_delay = metrics_calculator.calculate_detection_delay()
                    false_alarm_rate = metrics_calculator.calculate_false_alarm_rate()

                    # Store results including true_labels and predicted_scores for ROC curve
                    results[(k, alpha, h)] = {
                        "classification_metrics": classification_metrics,
                        "detection_delay": detection_delay,
                        "false_alarm_rate": false_alarm_rate,
                        "true_labels": true_labels,
                        "predicted_scores": predicted_labels
                    }

                    print(f"Completed k-NN experiment for k={k}, alpha={alpha}, h={h}")

        return results



    def run_decentralized_experiment(self):
        """
        Run the decentralized change detection experiments (Method A and B).
        
        Returns:
        - method_a_results: Results of Method A (Decision Fusion).
        - method_b_results: Results of Method B (Statistical Aggregation).
        """
        # Load and preprocess the data
        data_loader = DataLoader(self.file_path)
        data_loader.load_data()
        data_loader.preprocess_data()
        data = data_loader.get_data()

        # Extract test data and true labels
        bus_statistics = data[self.buses].values  # Extract statistics for buses
        true_labels = data['Label'].values  # True labels for the full dataset

        # Initialize the decentralized detection class
        detector = DecentralizedDetection(h_values=self.h_values, p_values=self.p_values)

        # Run Method A (Decision Fusion)
        bus_decisions = np.where(bus_statistics > self.h_values[0], 1, 0)  # Simple thresholding for bus decisions
        method_a_results = detector.method_a_decision_fusion(bus_decisions)

        # Run Method B (Statistical Aggregation)
        method_b_results = detector.method_b_statistical_aggregation(bus_statistics)

        return method_a_results, method_b_results

    def save_results(self, knn_results, method_a_results, method_b_results, output_file="experiment_results.txt"):
        """
        Save the results of the experiments to a file.
        
        Parameters:
        - knn_results: Results of the k-NN experiment.
        - method_a_results: Results of Method A (Decision Fusion).
        - method_b_results: Results of Method B (Statistical Aggregation).
        - output_file: File to save the results.
        """
        with open(output_file, "w") as file:
            file.write("k-NN Experiment Results:\n")
            for key, value in knn_results.items():
                file.write(f"Parameters (k={key[0]}, alpha={key[1]}, h={key[2]}):\n")
                file.write(f"Classification Metrics: {value['classification_metrics']}\n")
                file.write(f"Detection Delay: {value['detection_delay']}\n")
                file.write(f"False Alarm Rate: {value['false_alarm_rate']}\n\n")

            file.write("Method A (Decision Fusion) Results:\n")
            for p, detection_time in method_a_results.items():
                file.write(f"p={p}: Detection Time = {detection_time}\n")

            file.write("\nMethod B (Statistical Aggregation) Results:\n")
            for method, detection_time in method_b_results.items():
                file.write(f"Aggregation Method = {method}: Detection Time = {detection_time}\n")

        print(f"Results saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    k_values = [5, 10, 15]
    alpha_values = [0.001, 0.01, 0.05]
    h_values = [5, 10, 20]
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]

    # Initialize the experiment runner
    experiment_runner = ExperimentRunner(file_path, buses, k_values, alpha_values, h_values, p_values)

    # Run the k-NN experiment
    knn_results = experiment_runner.run_knn_experiment()

    # Run the decentralized methods experiment
    method_a_results, method_b_results = experiment_runner.run_decentralized_experiment()

    # Save all results to a file
    experiment_runner.save_results(knn_results, method_a_results, method_b_results)
