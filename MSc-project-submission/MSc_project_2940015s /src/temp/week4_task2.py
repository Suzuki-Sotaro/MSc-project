import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This task loads gas sensor data, applies Non-parametric CuSum to detect change points,
# and evaluates the results for different window sizes. The results are saved in a specified directory.

# Create directory for Task 2 results
task2_dir = os.path.join('results', 'Task2')
os.makedirs(task2_dir, exist_ok=True)

# Function to load gas sensor data
def load_gas_sensor_data(data_dir):
    data = []
    for i in range(1, 11):
        file_path = os.path.join(data_dir, f'batch{i}.dat')
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                features = [float(p.split(':')[1]) for p in parts[1:]]
                data.append(features)
    return np.array(data)

# Data directory
data_dir = './data/gas_sensor_dataset/'

# Load gas sensor data
gas_sensor_data = load_gas_sensor_data(data_dir)

# Check the shape of the data
print(gas_sensor_data.shape)

# Focus on one feature (e.g., the first feature)
feature_index = 0
data_feature = gas_sensor_data[:, feature_index]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data_feature)
plt.title(f'Gas Sensor Data - Feature {feature_index + 1}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig(os.path.join(task2_dir, f'gas_sensor_feature_{feature_index + 1}.png'))
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

# Set window sizes and threshold
window_sizes = [50, 100, 150]
threshold = 10  # Threshold

# Execute CuSum and plot results
results = []
plt.figure(figsize=(12, 6))
plt.plot(data_feature, label='Data')
for window_size in window_sizes:
    change_point, cusum = calculate_cusum(data_feature, window_size, threshold)
    plt.plot(cusum, label=f'CuSum (window size = {window_size})')
    if change_point != -1:
        plt.axvline(change_point, linestyle=':', label=f'Detected Change (window size = {window_size})')
    results.append({
        'Window Size': window_size,
        'Change Point': change_point
    })
plt.title(f'Non-Parametric CuSum for Feature {feature_index + 1}')
plt.xlabel('Time')
plt.ylabel('CuSum Value')
plt.legend()
plt.savefig(os.path.join(task2_dir, f'cusum_feature_{feature_index + 1}.png'))
plt.close()

# Create and save results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(task2_dir, 'results.csv'), index=False)
print(results_df)
