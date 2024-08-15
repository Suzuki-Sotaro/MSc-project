# The following is the code for qq_detection.py.
import numpy as np
import pandas as pd

# Function to binarize data based on a threshold
def binarize_data(data, threshold):
    return (data >= threshold).astype(int)

import numpy as np
import pandas as pd

# Function to calculate Q-Q distance
def calculate_qq_distance(window1, window2):
    quantiles = np.linspace(0.01, 0.99, 100)
    q_window1 = np.quantile(window1, quantiles)
    q_window2 = np.quantile(window2, quantiles)
    qq_distance = np.sqrt(2) / 2 * np.mean(np.abs(q_window1 - q_window2))
    return qq_distance

# Function to calculate Q-Q distance and detect changes.
def qq_detection(data, buses, window_size, threshold):
    results = []
    n_windows = len(data) - window_size + 1

    # Set binary threshold based on the initial window
    initial_thresholds = {bus: data[bus][:window_size].mean() for bus in buses}

    # Calculate distances for each bus
    for start in range(1, n_windows):
        total_qq_distance = 0
        
        for bus in buses:
            data_series = data[bus]
            binary_series = binarize_data(data_series, initial_thresholds[bus])
            window1 = binary_series[start - 1:start - 1 + window_size]
            window2 = binary_series[start:start + window_size]
            qq_distance = calculate_qq_distance(window1, window2)
            total_qq_distance += qq_distance
        
        # Change is detected if the total Q-Q distance exceeds the threshold
        change_detected = int(total_qq_distance >= threshold)
        results.append(change_detected)
    
    return results
