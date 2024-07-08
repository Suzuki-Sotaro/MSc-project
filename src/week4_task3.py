import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# This task loads bus data, applies Non-parametric CuSum to detect change points,
# and evaluates the results for different window sizes and thresholds.
# The results are saved in a specified directory.

# Create directory for Task 3 results
task3_dir = os.path.join('results', 'Task3')
os.makedirs(task3_dir, exist_ok=True)

# Load data from the specified CSV file
file_path = './data/A_LMPFreq3_Labeled.csv'
data = pd.read_csv(file_path)

# Specify bus numbers to analyze
bus_numbers = [115, 116, 117, 118, 119, 121, 135, 139]

# Extract the last 855 values
data_last_855 = data.iloc[-855:]

# Extract data for each specified bus
bus_data = {bus: data_last_855[f'Bus{bus}'].values for bus in bus_numbers}

# Extract labels
labels = data_last_855['Label'].values

# Function to execute non-parametric CuSum
def calculate_cusum(data, window_size, threshold):
    n = len(data)
    reference_window = data[:window_size]
    cusum = np.zeros(n)
    for t in range(window_size, n):
        test_window = data[t-window_size:t]
        distance = np.abs(np.mean(reference_window) - np.mean(test_window))
        cusum[t] = cusum[t-1] + distance
        if cusum[t] > threshold:
            return t, cusum  # Detect change point
    return -1, cusum  # No change point detected

# Set window sizes and thresholds
window_sizes = [30, 50, 70]
thresholds = [5, 10, 15]

# List to store results
results = []

# Execute CuSum for each bus data, plot the results, and store the results
for window_size in window_sizes:
    for threshold in thresholds:
        fig, axes = plt.subplots(len(bus_numbers), 1, figsize=(14, 16), sharex=True)
        fig.suptitle(f'Window Size: {window_size}, Threshold: {threshold}', fontsize=16)
        
        for idx, bus in enumerate(bus_numbers):
            ax = axes[idx]
            data_feature = bus_data[bus]
            change_point, cusum = calculate_cusum(data_feature, window_size, threshold)
            
            # Plot bus data
            ax.plot(data_feature, label=f'Bus {bus} Data')
            # Plot CuSum results
            ax.plot(cusum, label=f'CuSum (Bus {bus})', linestyle='--')
            # Plot true labels
            ax.plot(labels * max(cusum), label='True Labels', linestyle='-.')
            
            # Mark detected change point
            if change_point != -1:
                ax.axvline(change_point, color='r', linestyle=':', label='Detected Change')
            
            ax.set_title(f'Bus {bus}')
            ax.set_ylabel('Value')
            ax.legend()
            
            # Store results
            pred_labels = np.zeros(len(labels))
            if change_point != -1:
                pred_labels[change_point:] = 1
                
            cm = confusion_matrix(labels, pred_labels)
            accuracy = accuracy_score(labels, pred_labels)
            precision = precision_score(labels, pred_labels)
            recall = recall_score(labels, pred_labels)
            f1 = f1_score(labels, pred_labels)
            
            results.append({
                'Bus': bus,
                'Window Size': window_size,
                'Threshold': threshold,
                'Change Point': change_point,
                'Confusion Matrix': cm,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(task3_dir, f'cusum_window_{window_size}_threshold_{threshold}.png'))
        plt.close()

# Create and save results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df[['Bus', 'Window Size', 'Threshold', 'Change Point', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
results_df.to_csv(os.path.join(task3_dir, 'results.csv'), index=False)
print(results_df)
