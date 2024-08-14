# 以下はcusum_detection.pyの内容です
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def generate_data(mean_before, sigma_before, mean_after, sigma_after, change_point, n_samples):
    data_before = np.random.normal(mean_before, sigma_before, change_point)
    data_after = np.random.normal(mean_after, sigma_after, n_samples - change_point)
    return np.concatenate((data_before, data_after))

def cusum(data, mean_before, sigma_before, mean_after, sigma_after, threshold):
    n = len(data)
    cusum_scores = np.zeros(n)
    detection_point = -1
    
    for i in range(1, n):
        likelihood_ratio = np.log((1 / (np.sqrt(2 * np.pi) * sigma_after)) * np.exp(-0.5 * ((data[i] - mean_after) / sigma_after) ** 2)) - \
                           np.log((1 / (np.sqrt(2 * np.pi) * sigma_before)) * np.exp(-0.5 * ((data[i] - mean_before) / sigma_before) ** 2))
        cusum_scores[i] = max(0, cusum_scores[i-1] + likelihood_ratio)
        
        if cusum_scores[i] > threshold and detection_point == -1:
            detection_point = i
    
    return cusum_scores, detection_point

def calculate_statistics(df, buses):
    statistics = {}
    for bus in buses:
        bus_data = df[bus].values
        mean_before = np.mean(bus_data[df['Label'] == 0])
        sigma_before = np.std(bus_data[df['Label'] == 0])
        mean_after = np.mean(bus_data[df['Label'] == 1])
        sigma_after = np.std(bus_data[df['Label'] == 1])
        statistics[bus] = {
            'mean_before': mean_before,
            'sigma_before': sigma_before,
            'mean_after': mean_after,
            'sigma_after': sigma_after
        }
    return statistics

def run_experiments(mean_before, sigma_before, mean_after, sigma_after, change_point, n_samples, threshold_values, num_experiments):
    detection_delays = {h: [] for h in threshold_values}
    false_alarms = {h: 0 for h in threshold_values}
    
    for _ in range(num_experiments):
        data = generate_data(mean_before, sigma_before, mean_after, sigma_after, change_point, n_samples)
        
        for h in threshold_values:
            _, detection_point = cusum(data, mean_before, sigma_before, mean_after, sigma_after, h)
            
            if detection_point != -1:
                if detection_point < change_point:
                    false_alarms[h] += 1
                else:
                    detection_delays[h].append(detection_point - change_point)
    
    avg_detection_delays = {h: np.mean(delays) if delays else 0 for h, delays in detection_delays.items()}
    false_alarm_rates = {h: false_alarms[h] / num_experiments for h in threshold_values}
    
    return avg_detection_delays, false_alarm_rates

def plot_results(data, change_point, cusum_scores, detection_points, threshold_values):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(data, label='Data')
    ax1.axvline(change_point, color='red', linestyle='--', label='True Change Point')

    for i, (threshold, cusum_score) in enumerate(zip(threshold_values, cusum_scores)):
        if detection_points[i] != -1:
            ax1.axvline(detection_points[i], color=f'C{i}', linestyle='--', label=f'Detected Change (h={threshold})')
        ax2.plot(cusum_score, color=f'C{i}', label=f'CuSum Scores (h={threshold})')
        ax2.axhline(threshold, color=f'C{i}', linestyle='--', label=f'Threshold (h={threshold})')

    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Value')
    ax1.set_title('CuSum Change Detection')
    ax1.legend()

    ax2.set_xlabel('Sample')
    ax2.set_ylabel('CuSum Score')
    ax2.set_title('CuSum Scores')
    ax2.legend()

    plt.tight_layout()
    # plt.show()

def sensitivity_analysis(mean_values, sigma_values, change_point, n_samples, threshold_values, num_experiments):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i, ((mean_before, mean_after), (sigma_before, sigma_after)) in enumerate(zip(mean_values, sigma_values)):
        avg_detection_delays, false_alarm_rates = run_experiments(mean_before, sigma_before, mean_after, sigma_after,
                                                                  change_point, n_samples, threshold_values, num_experiments)
        
        ax1.plot(threshold_values, [avg_detection_delays[h] for h in threshold_values], marker='o', 
                 label=f'Mean: {mean_before}->{mean_after}, Sigma: {sigma_before}->{sigma_after}')
        ax2.plot(threshold_values, [false_alarm_rates[h] for h in threshold_values], marker='o', 
                 label=f'Mean: {mean_before}->{mean_after}, Sigma: {sigma_before}->{sigma_after}')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Average Detection Delay')
    ax1.set_title('Sensitivity Analysis - Average Detection Delay')
    ax1.legend()

    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('False Alarm Rate')
    ax2.set_title('Sensitivity Analysis - False Alarm Rate')
    ax2.legend()

    plt.tight_layout()
    # plt.show()
