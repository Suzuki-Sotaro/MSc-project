import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def local_detection(data, threshold):
    """
    Perform local change detection on a single bus's data using a simple threshold.
    
    Args:
        data (np.ndarray): The time series data for a single bus.
        threshold (float): The detection threshold.
    
    Returns:
        np.ndarray: A binary array indicating detected changes (1) and no changes (0).
    """
    print(f"Performing local detection with threshold={threshold}...")
    changes = np.zeros(len(data))
    cusum = np.zeros(len(data))
    
    for i in range(1, len(data)):
        cusum[i] = cusum[i-1] + (data[i] - data[i-1])
        if cusum[i] > threshold:
            changes[i] = 1
            cusum[i] = 0  # Reset after detection
    
    print(f"Local detection completed. Changes detected: {np.sum(changes)} time points.")
    return changes

def fusion_decision(votes, p):
    """
    Make a global decision based on the fusion of local detections.
    
    Args:
        votes (np.ndarray): The binary votes (1 for detected change, 0 for no change) from all buses.
        p (float): The required percentage of buses to report a change for the global decision.
    
    Returns:
        np.ndarray: A binary array indicating the global change detection decision (1) and no changes (0).
    """
    print(f"Performing fusion decision with p={p}...")
    num_buses = votes.shape[0]
    required_votes = int(np.ceil(p * num_buses))
    
    global_decision = np.sum(votes, axis=0) >= required_votes
    print(f"Global decision made. Changes detected: {np.sum(global_decision)} time points.")
    return global_decision.astype(int)

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

def plot_results(data, global_decision, true_labels, title):
    """
    Plot the time series data, true vs detected change points.
    
    Args:
        data (pd.DataFrame): The original time series data for all buses.
        global_decision (np.ndarray): The global change detection decisions.
        true_labels (np.ndarray): The true labels indicating change points.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 6))
    
    for i, bus in enumerate(data.columns):
        plt.plot(data[bus], label=f'Bus {bus}')
    
    plt.plot(true_labels * np.max(data.values), label='True Labels', linestyle='-.')
    plt.plot(global_decision * np.max(data.values), label='Detected Changes', linestyle='--', color='red')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_method_a(X, y, thresholds, p_values):
    """
    Run Method A (Decision Making via Fusion) on a dataset with multiple buses and evaluate performance.
    
    Args:
        X (pd.DataFrame): The feature data (LMP values for buses).
        y (pd.Series): The true labels indicating change points.
        thresholds (list): List of thresholds for local detection for each bus.
        p_values (list): List of p-values to experiment with for global decision making.
    """
    assert len(thresholds) == len(X.columns), "Number of thresholds must match number of buses"
    
    results = []
    
    for p in p_values:
        print(f"Analyzing with p={p}...")
        
        # Step 1: Perform local detection for each bus
        local_votes = np.zeros((len(X.columns), len(X)))
        for i, bus in enumerate(X.columns):
            print(f"Analyzing Bus {bus}...")
            data = X[bus].values
            local_votes[i] = local_detection(data, thresholds[i])
        
        # Step 2: Perform fusion decision to get global change detection
        global_decision = fusion_decision(local_votes, p)
        
        # Evaluate the performance
        performance = evaluate_performance(y.values, global_decision)
        results.append({'p': p, 'Performance': performance})
        
        # Plot the results
        plot_results(X, global_decision, y.values, f"Method A - Fusion Decision with p={p}")
    
    print("Method A analysis completed.")
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
    
    # Thresholds for local detection for each bus
    thresholds = [0.5, 0.5, 0.5]
    
    # p values to experiment with
    p_values = [0.1, 0.5, 0.9]
    
    # Run Method A analysis
    results = run_method_a(X, y, thresholds, p_values)
    
    print("Final results:")
    for result in results:
        print(f"p={result['p']}:")
        for key, value in result['Performance'].items():
            print(f"  {key}: {value}")
    
if __name__ == "__main__":
    main()
