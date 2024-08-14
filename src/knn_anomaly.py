import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def calculate_knn_distances(data, k):
    """
    Calculate the distance to the k-nearest neighbors for each point in the dataset.
    
    Args:
        data (np.ndarray): The time series data for a single bus.
        k (int): The number of nearest neighbors to consider.
    
    Returns:
        np.ndarray: The distances to the k-nearest neighbors for each point.
    """
    print(f"Calculating k-NN distances with k={k}...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(data.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(data.reshape(-1, 1))
    
    # Average distance to k nearest neighbors
    knn_distances = distances.mean(axis=1)
    print(f"k-NN distances calculated for {len(data)} data points.")
    return knn_distances

def detect_outliers(knn_distances, alpha):
    """
    Detect outliers based on k-NN distances and a specified tail probability (alpha).
    
    Args:
        knn_distances (np.ndarray): The distances to the k-nearest neighbors.
        alpha (float): The tail probability threshold for detecting outliers.
    
    Returns:
        np.ndarray: A binary array indicating outliers (1) and non-outliers (0).
    """
    print(f"Detecting outliers using alpha={alpha}...")
    threshold = np.quantile(knn_distances, 1 - alpha)
    outliers = knn_distances > threshold
    print(f"Outliers detected: {np.sum(outliers)} out of {len(knn_distances)} data points.")
    return outliers.astype(int)

def cusum_algorithm(data, threshold):
    """
    Apply the CUSUM algorithm to detect changes in the outlier data.
    
    Args:
        data (np.ndarray): The binary outlier data.
        threshold (float): The threshold for detecting a change in the CUSUM statistic.
    
    Returns:
        np.ndarray: A binary array indicating detected changes (1) and no changes (0).
    """
    print(f"Applying CUSUM with threshold={threshold}...")
    cusum = np.zeros(len(data))
    detected = np.zeros(len(data))
    
    for i in range(1, len(data)):
        cusum[i] = cusum[i-1] + data[i]
        if cusum[i] > threshold:
            detected[i] = 1
            cusum[i] = 0  # Reset after detection
    
    print(f"Change detected at {np.sum(detected)} time points.")
    return detected

def evaluate_performance(true_labels, pred_labels):
    """
    Evaluate the performance of the anomaly detection using various metrics.
    
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

def plot_results(data, knn_distances, outliers, changes, true_labels, title):
    """
    Plot the time series data, k-NN distances, detected outliers, and true vs detected change points.
    
    Args:
        data (np.ndarray): The time series data.
        knn_distances (np.ndarray): The k-NN distances.
        outliers (np.ndarray): The detected outliers.
        changes (np.ndarray): The detected changes using the CUSUM algorithm.
        true_labels (np.ndarray): The true labels indicating change points.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(data, label='Data')
    plt.title('Time Series Data')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(knn_distances, label='k-NN Distances', color='orange')
    plt.scatter(np.where(outliers)[0], knn_distances[outliers], color='red', label='Outliers')
    plt.title('k-NN Distances and Detected Outliers')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(true_labels * np.max(knn_distances), label='True Labels', linestyle='-.')
    plt.plot(changes * np.max(knn_distances), label='Detected Changes', linestyle='--', color='green')
    plt.title('True vs Detected Change Points')
    plt.legend()
    
    plt.suptitle(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

def run_knn_cusum_analysis(X, y, k=5, alpha=0.01, cusum_threshold=10):
    """
    Run k-NN and CUSUM analysis on a dataset with multiple buses and evaluate performance.
    
    Args:
        X (pd.DataFrame): The feature data (LMP values for buses).
        y (pd.Series): The true labels indicating change points.
        k (int): The number of nearest neighbors to consider for k-NN.
        alpha (float): The tail probability threshold for detecting outliers.
        cusum_threshold (float): The threshold for the CUSUM algorithm to detect changes.
    """
    results = []
    
    for bus in X.columns:
        print(f"Analyzing Bus {bus}...")
        data = X[bus].values
        true_labels = y.values
        
        # Step 1: Calculate k-NN distances
        knn_distances = calculate_knn_distances(data, k)
        
        # Step 2: Detect outliers based on k-NN distances
        outliers = detect_outliers(knn_distances, alpha)
        
        # Step 3: Apply CUSUM to detect changes based on outliers
        detected_changes = cusum_algorithm(outliers, cusum_threshold)
        
        # Evaluate the performance
        performance = evaluate_performance(true_labels, detected_changes)
        results.append({'Bus': bus, 'Performance': performance})
        
        # Plot the results
        plot_results(data, knn_distances, outliers, detected_changes, true_labels, f"Bus {bus} - k-NN and CUSUM Analysis")
    
    print("k-NN and CUSUM analysis completed.")
    return results

def main():
    # Example usage (replace with actual data loading)
    # Import necessary modules to load your actual data
    import pandas as pd
    
    # Dummy data for demonstration (replace with actual data)
    X = pd.DataFrame({
        'Bus1': np.random.normal(0, 1, 1000),
        'Bus2': np.random.normal(0, 1, 1000)
    })
    y = pd.Series([0]*500 + [1]*500)  # Simulated labels
    
    # Run k-NN and CUSUM analysis
    results = run_knn_cusum_analysis(X, y, k=10, alpha=0.05, cusum_threshold=15)
    
    print("Final results:")
    for result in results:
        print(f"Bus {result['Bus']}:")
        for key, value in result['Performance'].items():
            print(f"  {key}: {value}")
    
if __name__ == "__main__":
    main()
