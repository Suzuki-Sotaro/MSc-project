# Below is the content of main.py
import os
import numpy as np
import pandas as pd
from cusum_detection import analyze_cusum_with_methods, calculate_statistics
from glr_detection import analyze_glr
from qq_detection import qq_detection
from gem_detection import analyze_gem_with_methods
from pca_detection import analyze_pca_with_methods  

def load_and_preprocess_data(file_path, buses, n_samples):
    df = pd.read_csv(file_path)
    selected_buses = ['Week', 'Label'] + buses
    return df[selected_buses].tail(n_samples)

def save_results(results, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

def run_cusum_analysis(df, buses, statistics, params):
    cusum_results_a, cusum_results_b, individual_bus_results = analyze_cusum_with_methods(
        df, buses, statistics, params['cusum_threshold_values'], 
        params['p_values'], params['aggregation_methods'], 
        params['sink_threshold_methods']
    )
    return pd.DataFrame(individual_bus_results), pd.DataFrame(cusum_results_a), pd.DataFrame(cusum_results_b)


def run_glr_analysis(df, buses, statistics, params):
    return analyze_glr(df, buses, statistics, params['glr_threshold_values'])

def run_gem_analysis(df, buses, params):
    return analyze_gem_with_methods(
        df, buses, params['d'], params['k_values'], params['alpha_values'], 
        params['h_values'], params['p_values'], params['aggregation_methods'], 
        params['sink_threshold_methods']
    )

def run_qq_analysis(df, buses, params):
    return qq_detection(df, buses, params['window_sizes'], params['p_values'], 
                        params['aggregation_methods'], params['sink_threshold_methods'])
    
def run_pca_analysis(df, buses, params):  
    return analyze_pca_with_methods(
        df, buses, params['d'], params['gamma_values'], params['h_values'],
        params['alpha'], params['p_values'], params['aggregation_methods'],
        params['sink_threshold_methods']
    )

def save_multiple_results(results_dict, output_dir):
    for name, results in results_dict.items():
        save_results(results, output_dir, f'{name}.csv')

def main():
    # Configuration
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    n_samples = 855
    output_dir = './results/table/'

    # Common parameters
    params = {
        'window_sizes': [12, 24, 48],  # QQ検出用の複数のウィンドウサイズ
        'cusum_threshold_values': [0.1, 1, 10],
        'p_values': [0.1, 0.2, 0.5, 0.7, 0.9],
        'aggregation_methods': ['average', 'median', 'outlier_detection'],
        'sink_threshold_methods': ['average', 'minimum', 'maximum', 'median'],
        'glr_threshold_values': [0.01, 0.1, 1],
        'd': 3,
        'k_values': [10],
        'alpha_values': [0.1, 0.3, 0.5, 0.7], 
        'h_values': [3, 5, 7, 10],
        'gamma_values': [0.9, 0.95, 0.99], 
        'alpha': 0.05  # PCA
    }

    # Load and preprocess the data
    df = load_and_preprocess_data(file_path, buses, n_samples)
    statistics = calculate_statistics(df, buses)

    # Run analyses
    cusum_results, cusum_results_a, cusum_results_b = run_cusum_analysis(df, buses, statistics, params)
    glr_results, glr_results_a, glr_results_b = run_glr_analysis(df, buses, statistics, params)
    gem_results, gem_results_a, gem_results_b = run_gem_analysis(df, buses, params)
    qq_results, qq_results_a, qq_results_b = run_qq_analysis(df, buses, params)
    pca_results, pca_results_a, pca_results_b = run_pca_analysis(df, buses, params)  

    # Save results
    results_dict = {
        'cusum_analysis_results': cusum_results,
        'cusum_analysis_results_method_a': cusum_results_a,
        'cusum_analysis_results_method_b': cusum_results_b,
        'glr_analysis_results': glr_results,
        'glr_analysis_results_method_a': glr_results_a,
        'glr_analysis_results_method_b': glr_results_b,
        'gem_analysis_results': gem_results,
        'gem_analysis_results_method_a': gem_results_a.drop(columns='Detections'),
        'gem_analysis_results_method_b': gem_results_b.drop(columns='Detections'),
        'qq_analysis_results': qq_results,
        'qq_analysis_results_method_a': qq_results_a.drop(columns='Detections'),
        'qq_analysis_results_method_b': qq_results_b.drop(columns='Detections'),
        'pca_analysis_results': pca_results,  
        'pca_analysis_results_method_a': pca_results_a.drop(columns='Detections'),  
        'pca_analysis_results_method_b': pca_results_b.drop(columns='Detections')  
    }
    save_multiple_results(results_dict, output_dir)

if __name__ == '__main__':
    main()