import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ディレクトリの作成
task2_dir = os.path.join('results', 'Task2')
os.makedirs(task2_dir, exist_ok=True)

# データの読み込み
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

# データディレクトリ
data_dir = './data/gas_sensor_dataset/'

# データの読み込み
gas_sensor_data = load_gas_sensor_data(data_dir)

# データの形状を確認
print(gas_sensor_data.shape)

# 一つの特徴量に焦点を当てる (例: 最初の特徴量)
feature_index = 0
data_feature = gas_sensor_data[:, feature_index]

# データのプロット
plt.figure(figsize=(12, 6))
plt.plot(data_feature)
plt.title(f'Gas Sensor Data - Feature {feature_index + 1}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig(os.path.join(task2_dir, f'gas_sensor_feature_{feature_index + 1}.png'))
plt.close()

# 非パラメトリックCuSumの実行
def calculate_cusum(data, window_size, threshold):
    n = len(data)
    reference_window = data[:window_size]
    cusum = np.zeros(n)
    for t in range(window_size, n):
        test_window = data[t-window_size:t]
        distance = np.abs(np.mean(reference_window) - np.mean(test_window))
        cusum[t] = cusum[t-1] + distance
        if cusum[t] > threshold:
            return t, cusum  # 変更点を検出
    return -1, cusum  # 変更点を検出できない場合

# ウィンドウサイズとしきい値の設定
window_sizes = [50, 100, 150]
threshold = 10  # しきい値

# CuSumの実行とプロット
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

# 結果のデータフレーム作成
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(task2_dir, 'results.csv'), index=False)
print(results_df)
