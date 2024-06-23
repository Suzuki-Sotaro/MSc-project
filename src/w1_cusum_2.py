import numpy as np
import matplotlib.pyplot as plt

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
    
    return detection_point

def run_experiments(mean_before, sigma_before, mean_after, sigma_after, change_point, n_samples, threshold_values, num_experiments):
    detection_delays = {h: [] for h in threshold_values}
    false_alarms = {h: 0 for h in threshold_values}
    
    for _ in range(num_experiments):
        data = generate_data(mean_before, sigma_before, mean_after, sigma_after, change_point, n_samples)
        
        for h in threshold_values:
            detection_point = cusum(data, mean_before, sigma_before, mean_after, sigma_after, h)
            
            if detection_point != -1:
                if detection_point < change_point:
                    false_alarms[h] += 1
                else:
                    detection_delays[h].append(detection_point - change_point)
    
    avg_detection_delays = {h: np.mean(delays) if delays else 0 for h, delays in detection_delays.items()}
    false_alarm_rates = {h: false_alarms[h] / num_experiments for h in threshold_values}
    
    return avg_detection_delays, false_alarm_rates

# パラメータ設定
mean_before = 4
sigma_before = 1
mean_after = 6
sigma_after = 2
change_point = 50
n_samples = 100
threshold_values = [5, 10, 15]
num_experiments = 100

# 実験の実行
avg_detection_delays, false_alarm_rates = run_experiments(mean_before, sigma_before, mean_after, sigma_after,
                                                          change_point, n_samples, threshold_values, num_experiments)

# 結果のプロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(threshold_values, [avg_detection_delays[h] for h in threshold_values])
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Average Detection Delay')
ax1.set_title('Average Detection Delay vs. Threshold')

ax2.bar(threshold_values, [false_alarm_rates[h] for h in threshold_values])
ax2.set_xlabel('Threshold')
ax2.set_ylabel('False Alarm Rate')
ax2.set_title('False Alarm Rate vs. Threshold')

plt.tight_layout()

# 感度分析
mean_values = [(4, 6), (4, 8), (4, 10)]
sigma_values = [(1, 2), (1, 3), (1, 4)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for i, ((mean_before, mean_after), (sigma_before, sigma_after)) in enumerate(zip(mean_values, sigma_values)):
    avg_detection_delays, false_alarm_rates = run_experiments(mean_before, sigma_before, mean_after, sigma_after,
                                                              change_point, n_samples, threshold_values, num_experiments)
    
    ax1.plot(threshold_values, [avg_detection_delays[h] for h in threshold_values], marker='o', label=f'Mean: {mean_before}->{mean_after}, Sigma: {sigma_before}->{sigma_after}')
    ax2.plot(threshold_values, [false_alarm_rates[h] for h in threshold_values], marker='o', label=f'Mean: {mean_before}->{mean_after}, Sigma: {sigma_before}->{sigma_after}')

ax1.set_xlabel('Threshold')
ax1.set_ylabel('Average Detection Delay')
ax1.set_title('Sensitivity Analysis - Average Detection Delay')
ax1.legend()

ax2.set_xlabel('Threshold')
ax2.set_ylabel('False Alarm Rate')
ax2.set_title('Sensitivity Analysis - False Alarm Rate')
ax2.legend()

plt.tight_layout()
plt.show()