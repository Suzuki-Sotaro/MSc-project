# 以下はexperiment_runner.pyのコードです。
import numpy as np
from data_loader import DataLoader
from knn_change_detection import KNNChangeDetector, detect_changes
from decentralized_methods import MethodA, MethodB, experiment_method_a, experiment_method_b
from performance_metrics import evaluate_performance

class ExperimentRunner:
    def __init__(self, data_path='./data/LMP.csv'):
        self.loader = DataLoader(data_path)
        self.loader.load_data()
        self.all_bus_data = self.loader.get_all_selected_bus_data()
        self.data = self.all_bus_data.iloc[:, 2:].values.T  # Exclude 'Week' and 'Label' columns
        self.true_changes = self.get_true_changes()

    def get_true_changes(self):
        labels = self.all_bus_data['Label'].values
        return np.where(np.diff(labels) != 0)[0] + 1

    def run_centralized_experiment(self, window_sizes, k_values, alpha_values, h_values):
        results = []
        for window_size in window_sizes:
            for k in k_values:
                for alpha in alpha_values:
                    for h in h_values:
                        for bus_idx in range(self.data.shape[0]):
                            bus_data = self.data[bus_idx].reshape(-1, 1)  # 2D arrayに変換
                            detected_changes = detect_changes(bus_data, window_size, k, alpha, h, d=1)
                            performance = evaluate_performance(self.true_changes, detected_changes, len(bus_data))
                            results.append({
                                'method': 'Centralized',
                                'bus': self.loader.selected_buses[bus_idx],
                                'window_size': window_size,
                                'k': k,
                                'alpha': alpha,
                                'h': h,
                                **performance
                            })
        return results

    def run_method_a_experiment(self, window_sizes, k_values, alpha_values, h_values, p_values):
        results = []
        for window_size in window_sizes:
            for k in k_values:
                for alpha in alpha_values:
                    for h in h_values:
                        method_a_results = experiment_method_a(self.data, window_size, k, alpha, h, p_values)
                        for p, detected_changes in method_a_results.items():
                            performance = evaluate_performance(self.true_changes, detected_changes, self.data.shape[1])
                            results.append({
                                'method': 'Method A',
                                'window_size': window_size,
                                'k': k,
                                'alpha': alpha,
                                'h': h,
                                'p': p,
                                **performance
                            })
        return results

    def run_method_b_experiment(self, window_sizes, k_values, alpha_values, h_values, aggregation_methods):
        results = []
        for window_size in window_sizes:
            for k in k_values:
                for alpha in alpha_values:
                    for h in h_values:
                        method_b_results = experiment_method_b(self.data, window_size, k, alpha, h, aggregation_methods)
                        for method, detected_changes in method_b_results.items():
                            performance = evaluate_performance(self.true_changes, detected_changes, self.data.shape[1])
                            results.append({
                                'method': 'Method B',
                                'window_size': window_size,
                                'k': k,
                                'alpha': alpha,
                                'h': h,
                                'aggregation_method': method,
                                **performance
                            })
        return results

    def run_all_experiments(self):
        window_sizes = [5, 10, 20]
        k_values = [3, 5, 7]
        alpha_values = [0.01, 0.05, 0.1, 0.2]
        h_values = [5, 10, 15, 20]
        p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        aggregation_methods = ['average', 'median', 'outlier_removal']

        centralized_results = self.run_centralized_experiment(window_sizes, k_values, alpha_values, h_values)
        method_a_results = self.run_method_a_experiment(window_sizes, k_values, alpha_values, h_values, p_values)
        method_b_results = self.run_method_b_experiment(window_sizes, k_values, alpha_values, h_values, aggregation_methods)

        return centralized_results + method_a_results + method_b_results

if __name__ == "__main__":
    runner = ExperimentRunner()
    all_results = runner.run_all_experiments()

    # Print summary of results
    print(f"Total experiments run: {len(all_results)}")
    
    # Example: Print the best result for each method based on F1 score
    for method in ['Centralized', 'Method A', 'Method B']:
        method_results = [r for r in all_results if r['method'] == method]
        best_result = max(method_results, key=lambda x: x['F1 Score'])
        print(f"\nBest result for {method}:")
        for key, value in best_result.items():
            print(f"{key}: {value}")

    # You can save the results to a file for further analysis
    import json
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)