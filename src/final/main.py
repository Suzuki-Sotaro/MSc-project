# Below is the content of main.py
import os
import numpy as np
import pandas as pd
from cusum_detection import analyze_cusum_with_methods, calculate_statistics
from glr_detection import analyze_glr
from qq_detection import qq_detection
from method_a import analyze_method_a
from method_b import analyze_method_b
from gem_detection import analyze_gem_with_methods

# Load and preprocess the data
def load_and_preprocess_data(file_path, buses, n_samples):
    df = pd.read_csv(file_path)
    selected_buses = ['Week', 'Label'] + buses
    df = df[selected_buses].tail(n_samples)
    return df

def save_results(results, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

def analyze_method_results(method, df, buses, window_size, p_values=None):
    if method == 'A':
        return analyze_method_a(df, buses, window_size, p_values)
    elif method == 'B':
        return analyze_method_b(df, buses, window_size)

def main():
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    n_samples = 855
    window_size = 24

    # Load and preprocess the data
    df = load_and_preprocess_data(file_path, buses, n_samples)

    # Calculate the statistical properties of the data
    statistics = calculate_statistics(df, buses)

    # Parameters
    cusum_threshold_values = [5, 10, 15]
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    glr_threshold_values = [0.1, 1, 10]
    d = 3
    k_values = [10, 20]
    alpha_values = [0.01, 0.1, 1, 10]
    h_values = [1, 10]

    # CUSUM analysis with Method A and B
    cusum_results = []
    individual_bus_results = []
    for threshold in cusum_threshold_values:
        method_a_results, method_b_results, bus_results = analyze_cusum_with_methods(
            df, buses, statistics, threshold, p_values, aggregation_methods, sink_threshold_methods
        )
        for results in [method_a_results, method_b_results]:
            results['Threshold'] = threshold
            cusum_results.extend(results.to_dict('records'))
        individual_bus_results.extend(bus_results.to_dict('records'))
    
    cusum_results_df = pd.DataFrame(cusum_results).drop(columns='Detections')
    save_results(cusum_results_df, './results/table/', 'cusum_analysis_results_with_methods_ab.csv')
    
    individual_bus_results_df = pd.DataFrame(individual_bus_results)
    save_results(individual_bus_results_df, './results/table/', 'cusum_analysis_results_individual_buses.csv')


    # GLR analysis using Method A and B
    glr_results, glr_individual_results = analyze_glr(df, buses, statistics, glr_threshold_values)
    save_results(glr_results, './results/table/', 'glr_analysis_results.csv')
    save_results(glr_individual_results, './results/table/', 'glr_analysis_results_individual_buses.csv')

    # Method A analysis
    method_a_results = analyze_method_results('A', df, buses, window_size, p_values)
    save_results(method_a_results, './results/table/', 'method_a_analysis_results.csv')

    # Method B analysis
    method_b_results = analyze_method_results('B', df, buses, window_size)
    save_results(method_b_results, './results/table/', 'method_b_analysis_results.csv')

    # GEM analysis using Method A and B
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    aggregation_methods = ['average', 'median', 'outlier_detection']
    sink_threshold_methods = ['average', 'minimum', 'maximum', 'median']
    gem_results, gem_results_a, gem_results_b, gem_individual_results = analyze_gem_with_methods(
        df, buses, d, k_values, alpha_values, h_values, p_values, aggregation_methods, sink_threshold_methods
    )
    save_results(gem_results, './results/table/', 'gem_analysis_results.csv')
    save_results(gem_results_a, './results/table/', 'gem_analysis_results_a.csv')
    save_results(gem_results_b, './results/table/', 'gem_analysis_results_b.csv')
    save_results(gem_individual_results, './results/table/', 'gem_analysis_results_individual_buses.csv')


    # Q-Q detection analysis using Method A and B
    method_a_results, method_b_results, qq_individual_results = qq_detection(df, buses, window_size, p_values, aggregation_methods, sink_threshold_methods)
    save_results(method_a_results, './results/table/', 'qq_method_a_results.csv')
    save_results(method_b_results, './results/table/', 'qq_method_b_results.csv')
    save_results(qq_individual_results, './results/table/', 'qq_analysis_results_individual_buses.csv')


if __name__ == '__main__':
    main()
