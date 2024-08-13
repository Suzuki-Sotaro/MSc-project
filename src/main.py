# 以下はmain.pyのコードです。
from experiment_runner import ExperimentRunner
from visualization import Visualization

def main():
    # Define the path to your data file and other experiment parameters
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    k_values = [5, 10, 15]
    alpha_values = [0.001, 0.01, 0.05]
    h_values = [5, 10, 20]
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]

    # Initialize the experiment runner
    experiment_runner = ExperimentRunner(file_path, buses, k_values, alpha_values, h_values, p_values)

    # Run the k-NN experiment
    print("Running k-NN experiment...")
    knn_results = experiment_runner.run_knn_experiment()

    # Run the decentralized methods experiment
    print("Running decentralized methods experiment...")
    method_a_results, method_b_results = experiment_runner.run_decentralized_experiment()

    # Save all results to a file
    print("Saving results...")
    experiment_runner.save_results(knn_results, method_a_results, method_b_results)

    # Initialize the visualization class with the k-NN results
    print("Visualizing results...")
    visualizer = Visualization(knn_results)

    # Plot detection performance metrics
    visualizer.plot_detection_performance()

    # Plot ROC curves
    visualizer.plot_roc_curve()

    # Plot parameter sensitivity analysis
    visualizer.plot_parameter_sensitivity()

    print("Experiment completed and results visualized.")

if __name__ == "__main__":
    main()
