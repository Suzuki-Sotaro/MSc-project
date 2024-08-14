import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_qq_distance(reference_window, test_window):
    """
    Calculate the Q-Q distance between two windows of data.
    
    Args:
        reference_window (np.ndarray): The reference window data (before change).
        test_window (np.ndarray): The test window data (potentially after change).
    
    Returns:
        float: The Q-Q distance between the two data windows.
    """
    print("Calculating Q-Q distance between reference and test windows...")
    
    # Sort the data in each window
    reference_sorted = np.sort(reference_window)
    test_sorted = np.sort(test_window)
    
    # Calculate the Q-Q distance as the average absolute difference between corresponding quantiles
    qq_distance = np.mean(np.abs(reference_sorted - test_sorted))
    
    print(f"Q-Q distance: {qq_distance:.4f}")
    return qq_distance

def sliding_qq_distance(data, window_size, threshold, sliding_step=1):
    """
    Apply the Q-Q distance method with sliding windows to detect changes in the time series data.
    
    Args:
        data (np.ndarray): The time series data for a single bus.
        window_size (int): The size of the windows to compare.
        threshold (float): The threshold for detecting a change.
        sliding_step (int): The step size for sliding the windows.
    
    Returns:
        tuple: Tuple containing:
            - change_point (int): The index where the change is detected, or -1 if no change is detected.
            - qq_distances (np.ndarray): The Q-Q distances over time.
    """
    print(f"Applying sliding Q-Q distance with window size={window_size}, threshold={threshold}, sliding step={sliding_step}...")
    
    n = len(data)
    qq_distances = np.zeros(n - window_size * 2)
    
    for i in range(0, len(qq_distances), sliding_step):
        reference_window = data[i:i + window_size]
        test_window = data[i + window_size:i + 2 * window_size]
        qq_distances[i] = calculate_qq_distance(reference_window, test_window)
        
        if np.sum(qq_distances[:i+1]) > threshold:
            change_point = i + window_size
            print(f"Change detected at index {change_point} (Cumulative Q-Q distance exceeded threshold)")
            return change_point, qq_distances
        
    print("No change detected.")
    return -1, qq_distances  # No change detected

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

def plot_results(data, qq_distances, true_labels, change_point, title):
    """
    Plot the time series data, Q-Q distances, and true vs detected change points.
    
    Args:
        data (np.ndarray): The time series data.
        qq_distances (np.ndarray): The Q-Q distances.
        true_labels (np.ndarray): The true labels indicating change points.
        change_point (int): The detected change point index.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data')
    plt.plot(np.arange(len(qq_distances)) + len(data) - len(qq_distances), qq_distances, label='Q-Q Distance', linestyle='--')
    plt.plot(true_labels * np.max(qq_distances), label='True Labels', linestyle='-.')
    
    if change_point != -1:
        plt.axvline(change_point, color='red', linestyle=':', label='Detected Change')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_qq_analysis(X, y, window_size=30, threshold=10, sliding_step=1):
    """
    Run Q-Q distance analysis on a dataset with multiple buses and evaluate performance.
    
    Args:
        X (pd.DataFrame): The feature data (LMP values for buses).
        y (pd.Series): The true labels indicating change points.
        window_size (int): The size of the windows to compare.
        threshold (float): The threshold for detecting a change.
        sliding_step (int): The step size for sliding the windows.
    """
    results = []
    
    for bus in X.columns:
        print(f"Analyzing Bus {bus}...")
        data = X[bus].values
        true_labels = y.values
        
        change_point, qq_distances = sliding_qq_distance(data, window_size, threshold, sliding_step)
        
        # Generate predicted labels based on detected change point
        pred_labels = np.zeros_like(true_labels)
        if change_point != -1:
            pred_labels[change_point:] = 1
        
        # Evaluate the performance
        performance = evaluate_performance(true_labels, pred_labels)
        results.append({'Bus': bus, 'Performance': performance})
        
        # Plot the results
        plot_results(data, qq_distances, true_labels, change_point, f"Bus {bus} - Q-Q Distance Analysis")
    
    print("Q-Q distance analysis completed.")
    return results

def main():
    # Example usage (replace with actual data loading)
    import pandas as pd
    
    # Dummy data for demonstration (replace with actual data)
    X = pd.DataFrame({
        'Bus1': np.random.normal(0, 1, 1000),
        'Bus2': np.random.normal(0, 1, 1000)
    })
    y = pd.Series([0]*500 + [1]*500)  # Simulated labels
    
    # Run Q-Q distance analysis
    results = run_qq_analysis(X, y, window_size=30, threshold=15, sliding_step=1)
    
    print("Final results:")
    for result in results:
        print(f"Bus {result['Bus']}:")
        for key, value in result['Performance'].items():
            print(f"  {key}: {value}")
    
if __name__ == "__main__":
    main()
