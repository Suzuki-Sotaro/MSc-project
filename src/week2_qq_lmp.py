import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the data
file_path = './data/A_LMPFreq3_Labeled.csv'
data_ghadeer = pd.read_csv(file_path)

# Function to calculate Q-Q distance
def calculate_qq_distance(window1, window2):
    quantiles = np.linspace(0.01, 0.99, 100)
    q_window1 = np.quantile(window1, quantiles)
    q_window2 = np.quantile(window2, quantiles)
    qq_distance = np.sqrt(2) / 2 * np.mean(np.abs(q_window1 - q_window2))
    return qq_distance

# Function to binarize data based on a threshold
def binarize_data(data, threshold):
    return (data >= threshold).astype(int)

# Set window size and slide step
window_size = 24
bus_columns = [col for col in data_ghadeer.columns if col.startswith('Bus')]
n_windows = len(data_ghadeer) - window_size + 1

# Set binary threshold based on the initial window
initial_thresholds = {bus: data_ghadeer[bus][:window_size].mean() for bus in bus_columns}

# Lists to store results for approach 1 and approach 2
distances_approach1_all_buses = []
distances_approach2_all_buses = []
binary_decision_approach1 = []
binary_decision_approach2 = []

# Calculate distances for each bus
for bus in bus_columns:
    data_series = data_ghadeer[bus]
    binary_series = binarize_data(data_series, initial_thresholds[bus])
    
    # Approach 1: Compare initial window with all subsequent windows
    distances_approach1 = []
    initial_window = binary_series[:window_size]

    for start in range(1, n_windows):
        subsequent_window = binary_series[start:start + window_size]
        distance = calculate_qq_distance(initial_window, subsequent_window)
        distances_approach1.append(distance)
    
    distances_approach1_all_buses.append(distances_approach1)
    
    # Approach 2: Compare window x with window x-1
    distances_approach2 = []

    for start in range(1, n_windows):
        window_x_minus_1 = binary_series[start - 1:start - 1 + window_size]
        window_x = binary_series[start:start + window_size]
        distance = calculate_qq_distance(window_x_minus_1, window_x)
        distances_approach2.append(distance)
    
    distances_approach2_all_buses.append(distances_approach2)

# Calculate mean distances for each approach
mean_distances_approach1 = np.mean(distances_approach1_all_buses, axis=0)
mean_distances_approach2 = np.mean(distances_approach2_all_buses, axis=0)

# Get true labels
labels = data_ghadeer['Label'][window_size:].reset_index(drop=True)

# Perform majority voting for each approach
for start in range(n_windows - 1):
    binary_decision1 = [distances_approach1_all_buses[bus][start] for bus in range(len(bus_columns))]
    binary_decision_approach1.append(int(np.mean(binary_decision1) > np.mean(binary_decision_approach1)))

    binary_decision2 = [distances_approach2_all_buses[bus][start] for bus in range(len(bus_columns))]
    binary_decision_approach2.append(int(np.mean(binary_decision2) > np.mean(binary_decision_approach2)))

# Plot results for Approach 1
plt.figure(figsize=(14, 7))
plt.plot(mean_distances_approach1, label="Q-Q Distance (Approach 1)")
plt.plot(labels.values, label="True Labels", linestyle='--')
plt.plot(binary_decision_approach1, label="Binary Decision Fusion (Approach 1)", linestyle=':')
plt.title("Q-Q Distances and Binary Decision Fusion for Approach 1")
plt.xlabel("Window Start Time")
plt.ylabel("Q-Q Distance / Label / Binary Decision")
plt.legend()
plt.show()

# Plot results for Approach 2
plt.figure(figsize=(14, 7))
plt.plot(mean_distances_approach2, label="Q-Q Distance (Approach 2)")
plt.plot(labels.values, label="True Labels", linestyle='--')
plt.plot(binary_decision_approach2, label="Binary Decision Fusion (Approach 2)", linestyle='-.')
plt.title("Q-Q Distances and Binary Decision Fusion for Approach 2")
plt.xlabel("Window Start Time")
plt.ylabel("Q-Q Distance / Label / Binary Decision")
plt.legend()
plt.show()

# Function to evaluate performance
def evaluate_performance(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Evaluate performance for both approaches
performance_approach1 = evaluate_performance(binary_decision_approach1, labels)
performance_approach2 = evaluate_performance(binary_decision_approach2, labels)

# Create a DataFrame to display the performance results
performance_df = pd.DataFrame([performance_approach1, performance_approach2], index=['Approach 1', 'Approach 2'])
performance_df
