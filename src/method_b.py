import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def local_statistic(data, h):
    """
    Calculate the local statistics (e.g., cumulative sum) for a single bus's data.
    
    Args:
        data (np.ndarray): The time series data for a single bus.
        h (float): The threshold for triggering the local change detection.
    
    Returns:
        np.ndarray: The local statistic values.
    """
    print(f"Calculating local statistics with threshold h={h}...")
    s = np.zeros(len(data))
    
    for i in range(1, len(data)):
        s[i] = max(0, s[i-1] + (data[i] - h))
    
    print(f"Local statistics calculated for {len(data)} data points.")
    return s

def aggregate_statistics(statistics, method='average'):
    """
    Aggregate the local statistics from all buses using a specified method.
    
    Args:
        statistics (np.ndarray): The local statistics from all buses (each row is a bus).
        method (str): The aggregation method ('average', 'median', 'minimum', 'maximum').
    
    Returns:
        np.ndarray: The aggregated statistics.
    """
    print(f"Aggregating statistics using method '{method}'...")
    
    if method == 'average':
        agg_stat = np.mean(statistics, axis=0)
    elif method == 'median':
        agg_stat = np.median(statistics, axis=0)
    elif method == 'minimum':
        agg_stat = np.min(statistics, axis=0)
    elif method == 'maximum':
        agg_stat = np.max(statistics, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    print(f"Aggregation complete using method '{method}'.")
    return agg_stat

def detect_changes(agg_stat, global_threshold):
    """
    Detect changes based on the aggregated statistics and a global threshold.
    
    Args:
        agg_stat (np.ndarray): The aggregated statistics.
        global_threshold (float): The global threshold for detecting a change.
    
    Returns:
        np.ndarray: A binary array indicating detected changes (1) and no changes (0).
    """
    print(f"Detecting changes with global threshold={global_threshold}...")
    detected_changes = agg_stat > global_threshold
    print(f"Changes detected at {np.sum(detected_changes)} time points.")
    return detected_changes.astype(int)

def evaluate_performance(true_labels, pred_labels):
    """
    Evaluate the performance of the change detection using various metrics.
    
    Args:
        true_labels (np.ndarray): The true labels indicating change points.
        pred_labels (np.ndarray): The predicted labels indicating detected change points.
    
    Returns:
        dict: A dictionary containing confusion matrix, accuracy, precision, recall, and F1 score.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    print("Performance Evaluation:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def plot_results(data, agg_stat, global_decision, true_labels, title):
    """
    Plot the time series data, aggregated statistics, and true vs detected change points.
    
    Args:
        data (pd.DataFrame): The original time series data for all buses.
        agg_stat (np.ndarray): The aggregated statistics.
        global_decision (np.ndarray): The global change detection decisions.
        true_labels (np.ndarray): The true labels indicating change points.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for i, bus in enumerate(data.columns):
        plt.plot(data[bus], label=f'Bus {bus}')
    plt.title('Time Series Data')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(agg_stat, label='Aggregated Statistic', color='orange')
    plt.plot(global_decision * np.max(agg_stat), label='Detected Changes', linestyle='--', color='red')
    plt.plot(true_labels * np.max(agg_stat), label='True Labels', linestyle='-.')
    plt.title('Aggregated Statistic and Detected Changes')
    plt.legend()
    
    plt.suptitle(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

def run_method_b(X, y, local_thresholds, global_threshold, aggregation_methods):
    """
    Run Method B (Aggregation-based Decision Making) on a dataset with multiple buses and evaluate performance.
    
    Args:
        X (pd.DataFrame): The feature data (LMP values for buses).
        y (pd.Series): The true labels indicating change points.
        local_thresholds (list): List of local thresholds for each bus.
        global_threshold (float): The global threshold for detecting a change.
        aggregation_methods (list): List of aggregation methods to experiment with.
    """
    assert len(local_thresholds) == len(X.columns), "Number of local thresholds must match number of buses"
    
    results = []
    
    for method in aggregation_methods:
        print(f"Analyzing with aggregation method '{method}'...")
        
        # Step 1: Calculate local statistics for each bus
        local_statistics = np.zeros((len(X.columns), len(X)))
        for i, bus in enumerate(X.columns):
            print(f"Analyzing Bus {bus}...")
            data = X[bus].values
            local_statistics[i] = local_statistic(data, local_thresholds[i])
        
        # Step 2: Aggregate the local statistics
        agg_stat = aggregate_statistics(local_statistics, method)
        
        # Step 3: Detect changes based on aggregated statistics
        global_decision = detect_changes(agg_stat, global_threshold)
        
        # Evaluate the performance
        performance = evaluate_performance(y.values, global_decision)
        results.append({'Aggregation Method': method, 'Performance': performance})
        
        # Plot the results
        plot_results(X, agg_stat, global_decision, y.values, f"Method B - Aggregation Method: {method}")
    
    print("Method B analysis completed.")
    return results

def main():
    # Example usage (replace with actual data loading)
    import pandas as pd
    
    # Dummy data for demonstration (replace with actual data)
    X = pd.DataFrame({
        'Bus1': np.random.normal(0, 1, 1000),
        'Bus2': np.random.normal(0, 1, 1000),
        'Bus3': np.random.normal(0, 1, 1000)
    })
    y = pd.Series([0]*500 + [1]*500)  # Simulated labels
    
    # Local thresholds for each bus
    local_thresholds = [0.5, 0.5, 0.5]
    
    # Global threshold for change detection
    global_threshold = 1.0
    
    # Aggregation methods to experiment with
    aggregation_methods = ['average', 'median', 'minimum', 'maximum']
    
    # Run Method B analysis
    results = run_method_b(X, y, local_thresholds, global_threshold, aggregation_methods)
    
    print("Final results:")
    for result in results:
        print(f"Aggregation Method: {result['Aggregation Method']}:")
        for key, value in result['Performance'].items():
            print(f"  {key}: {value}")
    
if __name__ == "__main__":
    main()
