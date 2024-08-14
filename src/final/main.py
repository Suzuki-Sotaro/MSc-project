# 以下はmain.pyの内容です。
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from cusum_detection import cusum, plot_results, calculate_statistics
from glr_detection import glr_detect  # GLRのインポート
from qq_detection import qq_detection  # Q-Q距離のインポート
from method_a import analyze_method_a  # Method Aのインポート
from method_b import analyze_method_b  # Method Bのインポート  
from knn_detection import analyze_knn  # KNNのインポート

# データの読み込みと前処理
def load_and_preprocess_data(file_path, buses, n_samples):
    df = pd.read_csv(file_path)
    selected_buses = ['Week', 'Label'] + buses
    df = df[selected_buses]
    df_last_855 = df.tail(n_samples)
    return df_last_855

# CUSUMによる変更検出
def analyze_cusum(df, buses, statistics, threshold_values):
    results = []
    
    for bus in buses:
        data = df[bus].values
        label = df['Label'].values
        
        mean_before = statistics[bus]['mean_before']
        sigma_before = statistics[bus]['sigma_before']
        mean_after = statistics[bus]['mean_after']
        sigma_after = statistics[bus]['sigma_after']
        
        cusum_scores = []
        detection_points = []
        
        for threshold in threshold_values:
            scores, detection_point = cusum(data, mean_before, sigma_before, mean_after, sigma_after, threshold)
            cusum_scores.append(scores)
            detection_points.append(detection_point)
            
            # 変更検出結果を2値分類として評価
            predicted = np.zeros_like(label)
            if detection_point != -1:
                predicted[detection_point:] = 1
            
            accuracy = accuracy_score(label, predicted)
            recall = recall_score(label, predicted)
            precision = precision_score(label, predicted)
            f1 = f1_score(label, predicted)
            auc = roc_auc_score(label, cusum_scores[-1])

            results.append({
                'Bus': bus,
                'Threshold': threshold,
                'Detection_Point': detection_point,
                'Accuracy': accuracy,
                'Recall': recall,
                'Precision': precision,
                'F1_Score': f1,
                'AUC': auc
            })
        
        # 結果のプロット
        plot_results(data, np.where(label == 1)[0][0], cusum_scores, detection_points, threshold_values)

    return pd.DataFrame(results)

# `theta0`を変更前のデータの平均として設定
def calculate_theta0(df, buses):
    theta0_dict = {}
    for bus in buses:
        # 変更前データ (Label == 0) の平均を計算
        theta0 = df[df['Label'] == 0][bus].mean()
        theta0_dict[bus] = theta0
    return theta0_dict

# GLRによる変更検出
def analyze_glr(df, buses, statistics, glr_threshold_values, theta0_dict):
    results = []
    
    for bus in buses:
        data = df[bus].values
        label = df['Label'].values
        
        sigma = statistics[bus]['sigma_before']
        theta0 = theta0_dict[bus]  # 各バスごとの`theta0`を使用
        
        glr_scores = []
        detection_points = []
        
        for threshold in glr_threshold_values:
            detection_point, scores = glr_detect(data, theta0, sigma, threshold)
            glr_scores.append(scores)
            detection_points.append(detection_point)
            
            # 変更検出結果を2値分類として評価
            predicted = np.zeros_like(label)
            if detection_point != -1:
                predicted[detection_point:] = 1
            
            accuracy = accuracy_score(label, predicted)
            recall = recall_score(label, predicted)
            precision = precision_score(label, predicted)
            f1 = f1_score(label, predicted)
            auc = roc_auc_score(label, scores)

            results.append({
                'Bus': bus,
                'Threshold': threshold,
                'Detection_Point': detection_point,
                'Accuracy': accuracy,
                'Recall': recall,
                'Precision': precision,
                'F1_Score': f1,
                'AUC': auc
            })
        
        # 結果のプロット（オプション）
        plot_results(data, np.where(label == 1)[0][0], glr_scores, detection_points, glr_threshold_values)

    return pd.DataFrame(results)

# 精度の計算
def evaluate_performance(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)

    return {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1_Score': f1,
        'AUC': auc
    }

def save_results(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

# Method Aによる変更検出と評価
def analyze_method_a_results(df, buses, window_size):
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]  # pの値の設定
    method_a_results = analyze_method_a(df, buses, window_size, p_values)
    return method_a_results

def save_results(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

# Method Bによる変更検出と評価
def analyze_method_b_results(df, buses, window_size):
    method_b_results = analyze_method_b(df, buses, window_size)
    return method_b_results

def save_results(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

# KNNによる変更検出と評価
def analyze_knn_results(df, buses, d, k_values, alpha_values, h_values):
    knn_results = analyze_knn(df, buses, d, k_values, alpha_values, h_values)
    return knn_results

def main():
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    n_samples = 855
    window_size = 24
    
    # データの読み込みと前処理
    df = load_and_preprocess_data(file_path, buses, n_samples)
    
    # データの統計的性質を計算してパラメータを設定
    statistics = calculate_statistics(df, buses)
    
    # CUSUMによる変更検出と評価
    cusum_threshold_values = [5, 10, 15]
    cusum_results = analyze_cusum(df, buses, statistics, cusum_threshold_values)
    save_results(cusum_results, './results', 'cusum_analysis_results.csv')
    
    # GLRによる変更検出と評価
    glr_threshold_values = [1, 2, 3]
    theta0_dict = {bus: statistics[bus]['mean_before'] for bus in buses}
    glr_results = analyze_glr(df, buses, statistics, glr_threshold_values, theta0_dict)
    save_results(glr_results, './results', 'glr_analysis_results.csv')
    
    # Q-Q距離による変更検出と評価
    qq_threshold = 0.1  # Q-Q距離のしきい値
    qq_results = qq_detection(df, buses, window_size, qq_threshold)
    
    # Q-Q結果を評価
    labels = df['Label'][window_size:].reset_index(drop=True)
    qq_performance = evaluate_performance(qq_results, labels)
    qq_results_df = pd.DataFrame([qq_performance], index=['Q-Q Detection'])
    save_results(qq_results_df, './results', 'qq_analysis_results.csv')

    # Method Aによる変更検出と評価
    method_a_results = analyze_method_a_results(df, buses, window_size)
    save_results(method_a_results, './results', 'method_a_analysis_results.csv')
    
    # Method Bによる変更検出と評価
    method_b_results = analyze_method_b_results(df, buses, window_size)
    save_results(method_b_results, './results', 'method_b_analysis_results.csv')

    # KNNによる変更検出と評価
    d = 3  # d次元に変換
    k_values = [10, 15, 20]
    alpha_values = [0.01, 0.05, 0.1]
    h_values = [10, 20, 30]
    knn_results = analyze_knn_results(df, buses, d, k_values, alpha_values, h_values)
    save_results(knn_results, './results', 'knn_analysis_results.csv')
    
    print("Analysis completed and results saved in './results/'")

if __name__ == '__main__':
    main()