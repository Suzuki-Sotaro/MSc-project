# 以下はweek1_change_detection.pyの内容です
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
    
    return cusum_scores, detection_point

# パラメータ設定
mean_before = 4
sigma_before = 1
mean_after = 6
sigma_after = 2
change_point = 50
n_samples = 100
threshold_values = [5, 10, 15]
colors = ['magenta', 'cyan', 'orange']

# データ生成
data = generate_data(mean_before, sigma_before, mean_after, sigma_after, change_point, n_samples)

# プロット
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(data, label='Data')
ax1.axvline(change_point, color='red', linestyle='--', label='Change Point')

for threshold, color in zip(threshold_values, colors):
    cusum_scores, detection_point = cusum(data, mean_before, sigma_before, mean_after, sigma_after, threshold)
    if detection_point != -1:
        ax1.axvline(detection_point, color=color, linestyle='--', label=f'Detected Change (h={threshold})')
    
    ax2.plot(cusum_scores, color=color, label=f'CuSum Scores (h={threshold})')
    ax2.axhline(threshold, color=color, linestyle='--', label=f'Threshold (h={threshold})')

ax1.set_xlabel('Sample')
ax1.set_ylabel('Value')
ax1.set_title('CuSum Change Detection')
ax1.legend()

ax2.set_xlabel('Sample')
ax2.set_ylabel('CuSum Score')
ax2.set_title('CuSum Scores')
ax2.legend()

plt.tight_layout()
plt.show()