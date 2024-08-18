# Below is the content of main.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from cusum_detection import analyze_cusum_with_methods, calculate_statistics
from glr_detection import analyze_glr
from qq_detection import qq_detection  # Import Q-Q distance
from method_a import analyze_method_a  # Import Method A
from method_b import analyze_method_b  # Import Method B  
from gem_detection import analyze_gem_with_methods

# Load and preprocess the data
def load_and_preprocess_data(file_path, buses, n_samples):
    df = pd.read_csv(file_path)
    selected_buses = ['Week', 'Label'] + buses
    df = df[selected_buses]
    df_last_855 = df.tail(n_samples)
    return df_last_855

def save_results(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

# Change detection and evaluation using Method A
def analyze_method_a_results(df, buses, window_size):
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]  # Set the values for p
    method_a_results = analyze_method_a(df, buses, window_size, p_values)
    return method_a_results

# Change detection and evaluation using Method B
def analyze_method_b_results(df, buses, window_size):
    method_b_results = analyze_method_b(df, buses, window_size)
    return method_b_results

def generate_summary_table(method_results_dict):
    summary_df = pd.DataFrame()

    for method_name, results in method_results_dict.items():
        method_df = results.copy()
        method_df['Method'] = method_name
        summary_df = pd.concat([summary_df, method_df], ignore_index=True)
    
    summary_df = summary_df[['Method', 'Bus', 'Threshold', 'Detection_Point', 'Accuracy', 'Recall', 'Precision', 'F1_Score', 'AUC']]
    return summary_df

def save_summary_table(summary_df, output_path):
    summary_df.to_csv(output_path, index=False)
    print(f"Summary table saved to {output_path}")
    
def analyze_gem_results(df, buses, d, k_values, alpha_values, h_values):
    # Method Aのパラメータ
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    
    # Method Bのパラメータ
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    
    # GEM検出とMethod A、Bを組み合わせた分析を実行
    gem_results, gem_results_a, gem_results_b, time = analyze_gem_with_methods(
        df, buses, d, k_values, alpha_values, h_values,
        p_values, aggregation_methods, sink_threshold_methods
    )
    save_results(gem_results, './results/table/', 'gem_analysis_results.csv')
    save_results(gem_results_a, './results/table/', 'gem_analysis_results_a.csv')
    save_results(gem_results_b, './results/table/', 'gem_analysis_results_b.csv')

def main():
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    n_samples = 855
    window_size = 24
    
    # Load and preprocess the data
    df = load_and_preprocess_data(file_path, buses, n_samples)
    
    # Calculate the statistical properties of the data and set parameters
    statistics = calculate_statistics(df, buses)
    
    # CUSUM analysis with Method A and B
    cusum_threshold_values = [5, 10, 15]
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    
    cusum_results = []
    
    for threshold in cusum_threshold_values:
        method_a_results, method_b_results = analyze_cusum_with_methods(
            df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods
        )
        
        # Add threshold information to the results
        method_a_results['Threshold'] = threshold
        method_b_results['Threshold'] = threshold
        
        cusum_results.extend(method_a_results.to_dict('records'))
        cusum_results.extend(method_b_results.to_dict('records'))
    
    cusum_results_df = pd.DataFrame(cusum_results)
    
    # Save CUSUM results
    save_results(cusum_results_df, './results/table/', 'cusum_analysis_results_with_methods_ab.csv')
    
    # Change detection and evaluation using GLR
    glr_threshold_values = [0.1, 1, 10]
    theta0_dict = {bus: statistics[bus]['mean_before'] for bus in buses}
    glr_results = analyze_glr(df, buses, statistics, glr_threshold_values)
    save_results(glr_results, './results/table/', 'glr_analysis_results.csv')
    
    # Change detection and evaluation using Method A
    method_a_results = analyze_method_a_results(df, buses, window_size)
    save_results(method_a_results, './results/table/', 'method_a_analysis_results.csv')
    
    # Change detection and evaluation using Method B
    method_b_results = analyze_method_b_results(df, buses, window_size)
    save_results(method_b_results, './results/table/', 'method_b_analysis_results.csv')

    # Change detection and evaluation using GEM
    d = 3  # Transform to d-dimensions
    k_values = [10, 20]
    alpha_values = [0.01, 0.1, 1, 10]
    h_values = [1, 10]
    analyze_gem_results(df, buses, d, k_values, alpha_values, h_values)

    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    method_a_results, method_b_results = qq_detection(df, buses, window_size, p_values, aggregation_methods, sink_threshold_methods)

    # 結果の保存
    save_results(method_a_results, './results/table/', 'qq_method_a_results.csv')
    save_results(method_b_results, './results/table/', 'qq_method_b_results.csv')
        

if __name__ == '__main__':
    main()
