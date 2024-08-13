# 以下はmain.pyのコードです。
import os
import json
from experiment_runner import ExperimentRunner
from visualization import Visualizer
from performance_metrics import evaluate_performance

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Set up directories
    results_dir = "results"
    plots_dir = "plots"
    ensure_directory(results_dir)
    ensure_directory(plots_dir)

    # Run experiments
    print("Running experiments...")
    runner = ExperimentRunner()
    all_results = runner.run_all_experiments()

    # Save raw results
    results_file = os.path.join(results_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {results_file}")

    # Analyze results
    print("\nAnalyzing results...")
    methods = ['Centralized', 'Method A', 'Method B']
    metrics = ['F1 Score', 'Average Detection Delay', 'False Alarm Rate', 'Detection Rate']

    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        print(f"\nBest results for {method}:")
        for metric in metrics:
            best_result = max(method_results, key=lambda x: x[metric])
            print(f"  Best {metric}: {best_result[metric]:.4f}")
            print(f"    Parameters: {', '.join([f'{k}={v}' for k, v in best_result.items() if k not in ['method'] + metrics])}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = Visualizer(results_file)
    visualizer.generate_all_plots()
    print(f"Plots saved in {plots_dir}")

    # Perform additional analysis
    print("\nPerforming additional analysis...")
    
    # Compare centralized vs decentralized methods
    centralized_results = [r for r in all_results if r['method'] == 'Centralized']
    decentralized_results = [r for r in all_results if r['method'] in ['Method A', 'Method B']]
    
    centralized_f1 = sum(r['F1 Score'] for r in centralized_results) / len(centralized_results)
    decentralized_f1 = sum(r['F1 Score'] for r in decentralized_results) / len(decentralized_results)
    
    print(f"Average F1 Score:")
    print(f"  Centralized: {centralized_f1:.4f}")
    print(f"  Decentralized: {decentralized_f1:.4f}")

    # Analyze the effect of window size
    window_sizes = sorted(set(r['window_size'] for r in all_results))
    print("\nEffect of window size on F1 Score:")
    for size in window_sizes:
        size_results = [r for r in all_results if r['window_size'] == size]
        avg_f1 = sum(r['F1 Score'] for r in size_results) / len(size_results)
        print(f"  Window size {size}: {avg_f1:.4f}")

    # Analyze the best performing bus for centralized method
    best_bus = max(centralized_results, key=lambda x: x['F1 Score'])
    print(f"\nBest performing bus: Bus {best_bus['bus']} with F1 Score: {best_bus['F1 Score']:.4f}")

    print("\nExperiment complete. Check the results and plots directories for detailed output.")

if __name__ == "__main__":
    main()