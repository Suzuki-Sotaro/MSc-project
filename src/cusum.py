import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_cusum(data, threshold, drift=0):
    """
    Calculate the CUSUM statistics and detect change points.
    
    Args:
        data (np.ndarray): The time series data for a single bus.
        threshold (float): The threshold for detecting a change.
        drift (float): The drift parameter, used to adjust sensitivity.
    
    Returns:
        tuple: Tuple containing:
            - change_point (int): The index where the change is detected, or -1 if no change is detected.
            - cusum (np.ndarray): The CUSUM values over time.
    """
    print(f"Calculating CUSUM with threshold={threshold} and drift={drift}...")
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    
    for i in range(1, len(data)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - drift)
        cusum_neg[i] = max(0, cusum_neg[i-1] - data[i] - drift)
        
        if cusum_pos[i] > threshold:
            print(f"Change detected at index {i} (CUSUM positive exceeded threshold)")
            return i, cusum_pos
        
        if cusum_neg[i] > threshold:
            print(f"Change detected at index {i} (CUSUM negative exceeded threshold)")
            return i, cusum_neg
    
    print("No change detected.")
    return -1, cusum_pos  # No change detected

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

def plot_results(data, cusum, true_labels, change_point, title):
    """
    Plot the time series data, CUSUM values, and true vs detected change points.
    
    Args:
        data (np.ndarray): The time series data.
        cusum (np.ndarray): The CUSUM values.
        true_labels (np.ndarray): The true labels indicating change points.
        change_point (int): The detected change point index.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data')
    plt.plot(cusum, label='CUSUM', linestyle='--')
    plt.plot(true_labels * np.max(cusum), label='True Labels', linestyle='-.')
    
    if change_point != -1:
        plt.axvline(change_point, color='red', linestyle=':', label='Detected Change')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_cusum_analysis(X, y, threshold=5, drift=0):
    """
    Run CUSUM analysis on a dataset with multiple buses and evaluate performance.
    
    Args:
        X (pd.DataFrame): The feature data (LMP values for buses).
        y (pd.Series): The true labels indicating change points.
        threshold (float): The threshold for detecting a change.
        drift (float): The drift parameter, used to adjust sensitivity.
    """
    results = []
    
    for bus in X.columns:
        print(f"Analyzing Bus {bus}...")
        data = X[bus].values
        true_labels = y.values
        
        change_point, cusum = calculate_cusum(data, threshold, drift)
        
        # Generate predicted labels based on detected change point
        pred_labels = np.zeros_like(true_labels)
        if change_point != -1:
            pred_labels[change_point:] = 1
        
        # Evaluate the performance
        performance = evaluate_performance(true_labels, pred_labels)
        results.append({'Bus': bus, 'Performance': performance})
        
        # Plot the results
        plot_results(data, cusum, true_labels, change_point, f"Bus {bus} - CUSUM Analysis")
    
    print("CUSUM analysis completed.")
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
    
    # Run CUSUM analysis
    results = run_cusum_analysis(X, y, threshold=10, drift=0.5)
    
    print("Final results:")
    for result in results:
        print(f"Bus {result['Bus']}:")
        for key, value in result['Performance'].items():
            print(f"  {key}: {value}")
    
if __name__ == "__main__":
    main()
