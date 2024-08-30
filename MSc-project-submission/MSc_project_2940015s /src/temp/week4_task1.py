import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# This task generates synthetic time-series data using an AR(1) process,
# applies both Non-parametric and Parametric CuSum methods to detect change points,
# and evaluates their performance in terms of false alarm rates and expected delay.
# The results are saved in a specified directory.

# Create directory for Task 1 results
task1_dir = os.path.join('results', 'Task1')
os.makedirs(task1_dir, exist_ok=True)

# Set parameters for data generation
mean0 = 0
variance0 = 1
mean1 = 1
variance1 = 1
theta_values = [0.1, 0.5, 0.8]
t_change = 100  # Change point
n_samples = 200  # Total number of data points

# Function to generate AR(1) process data
def generate_ar1_data(mean0, variance0, mean1, variance1, theta, t_change, n_samples):
    data = np.zeros(n_samples)
    data[0] = mean0
    for t in range(1, t_change):
        e_t = np.random.normal(0, np.sqrt(variance0))
        data[t] = data[t-1] + (1-theta)*(mean0 - data[t-1]) + e_t
    for t in range(t_change, n_samples):
        e_t = np.random.normal(0, np.sqrt(variance1))
        data[t] = data[t-1] + (1-theta)*(mean1 - data[t-1]) + e_t
    return data

# Generate and plot AR(1) process data for different theta values
plt.figure(figsize=(12, 6))
for theta in theta_values:
    data = generate_ar1_data(mean0, variance0, mean1, variance1, theta, t_change, n_samples)
    plt.plot(data, label=f'theta = {theta}')
plt.axvline(t_change, color='r', linestyle='--', label='Change Point')
plt.title('AR(1) Process Data with Different Theta Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(task1_dir, 'ar1_process_data.png'))
plt.close()

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

# Function to execute parametric CuSum
def parametric_cusum(data, mean0, mean1, variance0, variance1, threshold):
    n = len(data)
    cusum = np.zeros(n)
    for t in range(1, n):
        k_t = (mean1 - mean0) / np.sqrt(variance1 + variance0)
        cusum[t] = max(0, cusum[t-1] + k_t * (data[t] - (mean0 + mean1) / 2))
        if cusum[t] > threshold:
            return t, cusum  # Detect change point
    return -1, cusum  # No change point detected

# Set window sizes and threshold
window_sizes = [10, 20, 30]
threshold = 5  # Threshold

# Store experiment results
results = []

for theta in theta_values:
    data = generate_ar1_data(mean0, variance0, mean1, variance1, theta, t_change, n_samples)
    
    for window_size in window_sizes:
        # Non-parametric CuSum
        change_point_np, cusum_np = calculate_cusum(data, window_size, threshold)
        
        # Parametric CuSum
        change_point_p, cusum_p = parametric_cusum(data, mean0, mean1, variance0, variance1, threshold)
        
        # Set true and predicted labels
        true_labels = np.zeros(n_samples)
        true_labels[t_change:] = 1
        
        pred_labels_np = np.zeros(n_samples)
        if change_point_np != -1:
            pred_labels_np[change_point_np:] = 1
        
        pred_labels_p = np.zeros(n_samples)
        if change_point_p != -1:
            pred_labels_p[change_point_p:] = 1
        
        # Calculate confusion matrix
        cm_np = confusion_matrix(true_labels, pred_labels_np)
        cm_p = confusion_matrix(true_labels, pred_labels_p)
        
        # Calculate false alarm rate and expected delay
        false_alarm_rate_np = cm_np[0, 1] / (cm_np[0, 0] + cm_np[0, 1])
        expected_delay_np = change_point_np - t_change if change_point_np != -1 else n_samples
        
        false_alarm_rate_p = cm_p[0, 1] / (cm_p[0, 0] + cm_p[0, 1])
        expected_delay_p = change_point_p - t_change if change_point_p != -1 else n_samples
        
        # Store results
        results.append({
            'Theta': theta,
            'Window Size': window_size,
            'Change Point (Non-Parametric)': change_point_np,
            'False Alarm Rate (Non-Parametric)': false_alarm_rate_np,
            'Expected Delay (Non-Parametric)': expected_delay_np,
            'Change Point (Parametric)': change_point_p,
            'False Alarm Rate (Parametric)': false_alarm_rate_p,
            'Expected Delay (Parametric)': expected_delay_p
        })

        # Save plots
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Data')
        plt.axvline(t_change, color='r', linestyle='--', label='True Change Point')
        plt.plot(cusum_np, label=f'CuSum (Non-Parametric, window size = {window_size})')
        if change_point_np != -1:
            plt.axvline(change_point_np, linestyle=':', label=f'Detected Change (Non-Parametric, window size = {window_size})')
        plt.plot(cusum_p, label=f'CuSum (Parametric, window size = {window_size})')
        if change_point_p != -1:
            plt.axvline(change_point_p, linestyle=':', label=f'Detected Change (Parametric, window size = {window_size})')
        plt.title(f'CuSum for Theta = {theta}, Window Size = {window_size}')
        plt.xlabel('Time')
        plt.ylabel('CuSum Value')
        plt.legend()
        plt.savefig(os.path.join(task1_dir, f'cusum_theta_{theta}_window_{window_size}.png'))
        plt.close()

# Create and save results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(task1_dir, 'results.csv'), index=False)
print(results_df)
