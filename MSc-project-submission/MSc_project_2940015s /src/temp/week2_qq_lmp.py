# qq_distance.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_qq_distance(window1, window2):
    quantiles = np.linspace(0.01, 0.99, 100)
    q_window1 = np.quantile(window1, quantiles)
    q_window2 = np.quantile(window2, quantiles)
    qq_distance = np.sqrt(2) / 2 * np.mean(np.abs(q_window1 - q_window2))
    return qq_distance

def binarize_data(data, threshold):
    return (data >= threshold).astype(int)

def analyze_qq(data_ghadeer, window_size, buses):
    n_windows = len(data_ghadeer) - window_size + 1
    
    # 初期しきい値を設定
    initial_thresholds = {bus: data_ghadeer[bus][:window_size].mean() for bus in buses}

    # 結果を保存するリスト
    distances_approach1_all_buses = []
    distances_approach2_all_buses = []
    binary_decision_approach1 = []
    binary_decision_approach2 = []

    # 各バスに対して距離を計算
    for bus in buses:
        data_series = data_ghadeer[bus]
        binary_series = binarize_data(data_series, initial_thresholds[bus])
        
        # アプローチ 1: 最初のウィンドウとすべての後続ウィンドウを比較
        distances_approach1 = []
        initial_window = binary_series[:window_size]

        for start in range(1, n_windows):
            subsequent_window = binary_series[start:start + window_size]
            distance = calculate_qq_distance(initial_window, subsequent_window)
            distances_approach1.append(distance)
        
        distances_approach1_all_buses.append(distances_approach1)
        
        # アプローチ 2: ウィンドウ x とウィンドウ x-1 を比較
        distances_approach2 = []

        for start in range(1, n_windows):
            window_x_minus_1 = binary_series[start - 1:start - 1 + window_size]
            window_x = binary_series[start:start + window_size]
            distance = calculate_qq_distance(window_x_minus_1, window_x)
            distances_approach2.append(distance)
        
        distances_approach2_all_buses.append(distances_approach2)

    # 平均距離を計算
    mean_distances_approach1 = np.mean(distances_approach1_all_buses, axis=0)
    mean_distances_approach2 = np.mean(distances_approach2_all_buses, axis=0)

    # 真のラベルを取得
    labels = data_ghadeer['Label'][window_size:].reset_index(drop=True)

    # 多数決を行う
    for start in range(n_windows - 1):
        binary_decision1 = [distances_approach1_all_buses[bus][start] for bus in range(len(buses))]
        binary_decision_approach1.append(int(np.mean(binary_decision1) > np.mean(binary_decision_approach1)))

        binary_decision2 = [distances_approach2_all_buses[bus][start] for bus in range(len(buses))]
        binary_decision_approach2.append(int(np.mean(binary_decision2) > np.mean(binary_decision_approach2)))

    return mean_distances_approach1, mean_distances_approach2, binary_decision_approach1, binary_decision_approach2, labels

def evaluate_performance(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }
