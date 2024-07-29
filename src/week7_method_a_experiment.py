import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Create directory for Method A results
method_a_dir = os.path.join('results', 'MethodA')
os.makedirs(method_a_dir, exist_ok=True)

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

# Set initial parameters
window_size = 50  # You may need to optimize this
initial_threshold = 10  # Initial threshold to be optimized for each bus

# Optimize local threshold h[k] for each bus
local_thresholds = {}
for bus in bus_numbers:
    best_threshold = initial_threshold
    best_f1_score = 0
    for threshold in range(1, 20):
        change_point, cusum = calculate_cusum(bus_data[bus], window_size, threshold)
        pred_labels = np.zeros(len(labels))
        if change_point != -1:
            pred_labels[change_point:] = 1
        f1 = f1_score(labels, pred_labels)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
    local_thresholds[bus] = best_threshold
    print(f'Optimized threshold for Bus {bus}: {best_threshold}')

# Voting scheme implementation
p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
results = []

for p in p_values:
    votes = np.zeros(len(labels))
    detection_times = []
    for bus in bus_numbers:
        threshold = local_thresholds[bus]
        change_point, cusum = calculate_cusum(bus_data[bus], window_size, threshold)
        if change_point != -1:
            votes[change_point:] += 1
    
    for t in range(len(votes)):
        if votes[t] >= p * len(bus_numbers):
            detection_times.append(t)
            break
    
    if detection_times:
        detection_time = detection_times[0]
    else:
        detection_time = -1
    
    pred_labels = np.zeros(len(labels))
    if detection_time != -1:
        pred_labels[detection_time:] = 1
    
    cm = confusion_matrix(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    
    results.append({
        'p': p,
        'Detection Time': detection_time,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(method_a_dir, 'results.csv'), index=False)
print(results_df)
