# Below is the content of main.py
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from cusum_detection import analyze_cusum_with_methods, calculate_statistics
from glr_detection import analyze_glr
from qq_detection import qq_detection
from gem_detection import analyze_gem_with_methods
from pca_detection import analyze_pca_with_methods  

def load_and_preprocess_data(file_path, n_samples):
    df = pd.read_csv(file_path)
    bus_columns = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    selected_buses = ['Week', 'Label'] + bus_columns
    return df[selected_buses].tail(n_samples).reset_index(), bus_columns

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
    combined_results_ab = pd.concat([pd.DataFrame(cusum_results_a), pd.DataFrame(cusum_results_b)], ignore_index=True)
    return pd.DataFrame(individual_bus_results), combined_results_ab

def run_glr_analysis(df, buses, params):
    window_size = params['window_size_glr'] 
    glr_results = analyze_glr(df, buses, window_size, params['glr_threshold_values'])
    combined_results_ab = pd.concat([glr_results[1], glr_results[2]], ignore_index=True)
    return glr_results[0], combined_results_ab

def run_gem_analysis(df, buses, params):
    gem_results = analyze_gem_with_methods(
        df, buses, params['d'], params['k_values'], params['alpha_values'], 
        params['h_values'], params['p_values'], params['aggregation_methods'], 
        params['sink_threshold_methods']
    )
    combined_results_ab = pd.concat([gem_results[1], gem_results[2]], ignore_index=True)
    return gem_results[0], combined_results_ab

def run_qq_analysis(df, buses, params):
    qq_results = qq_detection(df, buses, params['window_sizes'], params['qq_threshold_values'], params['p_values'], 
                        params['aggregation_methods'], params['sink_threshold_methods'])
    combined_results_ab = pd.concat([qq_results[1], qq_results[2]], ignore_index=True)
    return qq_results[0], combined_results_ab
    
def run_pca_analysis(df, buses, params):  
    pca_results = analyze_pca_with_methods(
        df, buses, params['d'], params['gamma_values'], params['h_values'],
        params['alpha'], params['p_values'], params['aggregation_methods'],
        params['sink_threshold_methods']
    )
    combined_results_ab = pd.concat([pca_results[1], pca_results[2]], ignore_index=True)
    return pca_results[0], combined_results_ab

def run_analysis(analysis_func, df, buses, *args):
    return analysis_func(df, buses, *args)

def parallel_analysis(df, buses, statistics, params):
    with Pool(processes=cpu_count()) as pool:
        cusum_results = pool.apply_async(run_analysis, (run_cusum_analysis, df, buses, statistics, params))
        glr_results = pool.apply_async(run_analysis, (run_glr_analysis, df, buses, params))
        gem_results = pool.apply_async(run_analysis, (run_gem_analysis, df, buses, params))
        qq_results = pool.apply_async(run_analysis, (run_qq_analysis, df, buses, params))
        pca_results = pool.apply_async(run_analysis, (run_pca_analysis, df, buses, params))

        cusum_results = cusum_results.get()
        glr_results = glr_results.get()
        gem_results = gem_results.get()
        qq_results = qq_results.get()
        pca_results = pca_results.get()

    return cusum_results, glr_results, gem_results, qq_results, pca_results

def main():
    file_path = './data/LMP.csv'
    n_samples = 855
    output_dir = './results/table/'
    output_dir_figures = './results/figure/'
    params = {
        'window_sizes': [24],
        'window_size_glr': 24,
        'cusum_threshold_values': [0.1, 0.5, 1.0, 2.0],
        'glr_threshold_values': [1000, 1500, 2000, 2500],
        'qq_threshold_values': [0.01, 0.05, 0.1, 0.2],
        'd': 3,
        'k_values': [10],
        'alpha_values': [0.1, 0.3, 0.5, 0.7, 0.9],
        'h_values': [1, 3, 5, 7, 10],
        'gamma_values': [0.3, 0.5, 0.7, 0.9],
        'alpha': 0.05,
        'p_values': [0.1, 0.2, 0.5, 0.7, 0.9],
        'aggregation_methods': ['average', 'median', 'outlier_detection'],
        'sink_threshold_methods': ['average', 'minimum', 'maximum', 'median']
    }

    df, buses = load_and_preprocess_data(file_path, n_samples)
    statistics = calculate_statistics(df, buses)

    cusum_results, glr_results, gem_results, qq_results, pca_results = parallel_analysis(df, buses, statistics, params)

    results_dict = {
        'cusum_analysis_results': cusum_results[0].drop(columns=['Data', 'Label', 'Detection']),
        'cusum_analysis_results_method_ab': cusum_results[1].drop(columns=['Data', 'Label', 'Detection']),
        'glr_analysis_results': glr_results[0].drop(columns=['Data', 'Label', 'Detection']),
        'glr_analysis_results_method_ab': glr_results[1].drop(columns=['Data', 'Label', 'Detection']),
        'gem_analysis_results': gem_results[0].drop(columns=['Data', 'Label', 'Detection']),
        'gem_analysis_results_method_ab': gem_results[1].drop(columns=['Data', 'Label', 'Detection']),
        'qq_analysis_results': qq_results[0].drop(columns=['Data', 'Label', 'Detection']),
        'qq_analysis_results_method_ab': qq_results[1].drop(columns=['Data', 'Label', 'Detection']),
        'pca_analysis_results': pca_results[0].drop(columns=['Data', 'Label', 'Detection']),
        'pca_analysis_results_method_ab': pca_results[1].drop(columns=['Data', 'Label', 'Detection'])
    }

    for name, results in results_dict.items():
        save_results(results, output_dir, f'{name}.csv')

if __name__ == '__main__':
    
    # import visualizations
    # visualizations.main()
    main()
    
    
    