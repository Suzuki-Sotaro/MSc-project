import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import config
import logging

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, filename=config.LOG_FILE, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def plot_data_with_changes(data: np.ndarray, true_changes: List[int], detected_changes: List[int], method_name: str):
    """
    Plot the data with true and detected change points.

    Args:
    data (np.ndarray): Input data
    true_changes (List[int]): List of true change points
    detected_changes (List[int]): List of detected change points
    method_name (str): Name of the detection method
    """
    print(f"Plotting data with changes for {method_name}...")
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot(data, label='Data')
    
    for tc in true_changes:
        plt.axvline(x=tc, color='r', linestyle='--', label='True change' if tc == true_changes[0] else '')
    
    for dc in detected_changes:
        plt.axvline(x=dc, color='g', linestyle=':', label='Detected change' if dc == detected_changes[0] else '')
    
    plt.title(f'Data with Change Points - {method_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"{config.RESULTS_DIR}/{method_name}_changes.png")
    plt.close()
    print(f"Plot saved as {config.RESULTS_DIR}/{method_name}_changes.png")
    logging.info(f"Data plot with changes created for {method_name}")

def plot_performance_comparison(results: pd.DataFrame):
    """
    Plot a comparison of performance metrics for different methods.

    Args:
    results (pd.DataFrame): DataFrame containing results from all experiments
    """
    print("Plotting performance comparison...")
    metrics = ['avg_delay', 'false_alarm_rate', 'precision', 'recall', 'f1_score']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        sns.barplot(x='method', y=metric, data=results)
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/performance_comparison.png")
    plt.close()
    print(f"Performance comparison plot saved as {config.RESULTS_DIR}/performance_comparison.png")
    logging.info("Performance comparison plot created")

def plot_method_a_results(results: pd.DataFrame):
    """
    Plot the results of Method A for different p values.

    Args:
    results (pd.DataFrame): DataFrame containing Method A results
    """
    print("Plotting Method A results...")
    metrics = ['avg_delay', 'false_alarm_rate', 'precision', 'recall', 'f1_score']
    p_values = [float(method.split('=')[1][:-1]) for method in results['method']]
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(p_values, results[metric], marker='o')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlabel('p value')
        plt.ylabel(metric)
    
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/method_a_results.png")
    plt.close()
    print(f"Method A results plot saved as {config.RESULTS_DIR}/method_a_results.png")

def plot_method_b_results(results: pd.DataFrame):
    """
    Plot the results of Method B for different h_methods and aggregation types.

    Args:
    results (pd.DataFrame): DataFrame containing Method B results
    """
    print("Plotting Method B results...")
    metrics = ['avg_delay', 'false_alarm_rate', 'precision', 'recall', 'f1_score']
    
    # h_methodsとagg_typesを正確に抽出
    h_methods = results['method'].apply(lambda x: x.split(',')[0].split('(')[1].strip()).unique()
    agg_types = results['method'].apply(lambda x: x.split(',')[1].strip()[:-1]).unique()
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)), sharex=True)
    for i, metric in enumerate(metrics):
        for agg_type in agg_types:
            # 各h_methodに対応する値を正確に抽出
            values = []
            for h_method in h_methods:
                method_name = f'Method B ({h_method}, {agg_type})'
                value = results[results['method'] == method_name][metric].values
                values.append(value[0] if len(value) > 0 else np.nan)
            
            axes[i].plot(h_methods, values, marker='o', label=agg_type)
        
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric)
        axes[i].legend()
    
    axes[-1].set_xlabel('h method')
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/method_b_results.png")
    plt.close()
    print(f"Method B results plot saved as {config.RESULTS_DIR}/method_b_results.png")
    
def create_heatmap(data: np.ndarray, title: str):
    """
    Create a heatmap of the input data.

    Args:
    data (np.ndarray): Input data
    title (str): Title of the heatmap
    """
    print(f"Creating heatmap: {title}")
    plt.figure(figsize=config.PLOT_FIGSIZE)
    sns.heatmap(data, cmap='YlOrRd', cbar_kws={'label': 'Value'})
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Time')
    plt.savefig(f"{config.RESULTS_DIR}/{title.replace(' ', '_').lower()}.png")
    plt.close()
    print(f"Heatmap saved as {config.RESULTS_DIR}/{title.replace(' ', '_').lower()}.png")
    logging.info(f"Heatmap created: {title}")

def visualize_results(data: np.ndarray, results: pd.DataFrame, true_changes: List[int]):
    print("Starting visualization of results...")
    
    for _, row in results.iterrows():
        plot_data_with_changes(data[:, 0], true_changes, row['detected_changes'], row['method'])
    
    plot_performance_comparison(results)
    
    # Method Aの結果をプロット
    method_a_results = results[results['method'].str.startswith('Method A')]
    if not method_a_results.empty:
        plot_method_a_results(method_a_results)
    
    # Method Bの結果をプロット
    method_b_results = results[results['method'].str.startswith('Method B')]
    if not method_b_results.empty:
        plot_method_b_results(method_b_results)
    
    print("Visualization of results completed.")

if __name__ == "__main__":
    # Example usage
    print("Running example visualization...")
    
    # Generate some example data and results
    np.random.seed(config.RANDOM_SEED)
    data = np.random.randn(1000, 10)
    true_changes = [250, 500, 750]
    
    results = pd.DataFrame([
        {'method': 'GEM', 'detected_changes': [248, 503, 755], 'avg_delay': 2.33, 'false_alarm_rate': 0.001, 'precision': 0.95, 'recall': 0.9, 'f1_score': 0.92},
        {'method': 'Q-Q', 'detected_changes': [251, 498, 752], 'avg_delay': 2.67, 'false_alarm_rate': 0.002, 'precision': 0.93, 'recall': 0.88, 'f1_score': 0.90},
        {'method': 'Method A', 'results': {0.1: {'avg_delay': 3.0, 'false_alarm_rate': 0.003, 'precision': 0.91, 'recall': 0.87, 'f1_score': 0.89},
                                           0.5: {'avg_delay': 2.5, 'false_alarm_rate': 0.002, 'precision': 0.94, 'recall': 0.89, 'f1_score': 0.91},
                                           0.9: {'avg_delay': 2.0, 'false_alarm_rate': 0.001, 'precision': 0.96, 'recall': 0.91, 'f1_score': 0.93}}},
        {'method': 'Method B', 'results': {'mean': {'average': {'avg_delay': 2.4, 'false_alarm_rate': 0.002, 'precision': 0.94, 'recall': 0.89, 'f1_score': 0.91},
                                                    'median': {'avg_delay': 2.6, 'false_alarm_rate': 0.001, 'precision': 0.95, 'recall': 0.88, 'f1_score': 0.91},
                                                    'mad': {'avg_delay': 2.5, 'false_alarm_rate': 0.002, 'precision': 0.93, 'recall': 0.9, 'f1_score': 0.91}},
                                           'max': {'average': {'avg_delay': 2.3, 'false_alarm_rate': 0.003, 'precision': 0.92, 'recall': 0.91, 'f1_score': 0.91},
                                                   'median': {'avg_delay': 2.5, 'false_alarm_rate': 0.002, 'precision': 0.93, 'recall': 0.89, 'f1_score': 0.91},
                                                   'mad': {'avg_delay': 2.4, 'false_alarm_rate': 0.002, 'precision': 0.94, 'recall': 0.9, 'f1_score': 0.92}}}}
    ])
    
    visualize_results(data, results, true_changes)
    print("Example visualization completed.")