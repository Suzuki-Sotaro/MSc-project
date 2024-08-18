# Below is the content of main.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from cusum_detection import cusum, plot_results, calculate_statistics
from glr_detection import glr_detect, plot_glr_results  # Import GLR
from qq_detection import qq_detection  # Import Q-Q distance
from method_a import analyze_method_a  # Import Method A
from method_b import analyze_method_b  # Import Method B  
from gem_detection import analyze_gem  # Import GEM

# Load and preprocess the data
def load_and_preprocess_data(file_path, buses, n_samples):
    df = pd.read_csv(file_path)
    selected_buses = ['Week', 'Label'] + buses
    df = df[selected_buses]
    df_last_855 = df.tail(n_samples)
    return df_last_855

# Change detection using CUSUM
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
            
            # Evaluate the change detection results as binary classification
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
        
        # Plot the results
        # Example usage within CUSUM analysis
        plot_results(data, np.where(label == 1)[0][0], cusum_scores, detection_points, threshold_values, 
             method_name='CUSUM', 
             save_path=f'./results/figure/cusum_{bus}_threshold_{threshold}.png')

    return pd.DataFrame(results)

# Set `theta0` as the mean of the data before the change
def calculate_theta0(df, buses):
    theta0_dict = {}
    for bus in buses:
        # Calculate the mean of the data before the change (Label == 0)
        theta0 = df[df['Label'] == 0][bus].mean()
        theta0_dict[bus] = theta0
    return theta0_dict

# Change detection using GLR
def analyze_glr(df, buses, statistics, glr_threshold_values, theta0_dict):
    results = []
    
    for bus in buses:
        data = df[bus].values
        label = df['Label'].values
        
        sigma = statistics[bus]['sigma_before']
        theta0 = theta0_dict[bus]  # Use `theta0` for each bus
        
        glr_scores = []
        detection_points = []
        
        for threshold in glr_threshold_values:
            detection_point, scores = glr_detect(data, theta0, sigma, threshold)
            glr_scores.append(scores)
            detection_points.append(detection_point)
            
            # Evaluate the change detection results as binary classification
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
        
        # Plot the results (optional)
        # Example usage within GLR analysis
        plot_glr_results(data, np.where(label == 1)[0][0], glr_scores, detection_points, glr_threshold_values, 
                 method_name='GLR', 
                 save_path=f'./results/figure/glr_{bus}_threshold_{threshold}.png')

    return pd.DataFrame(results)

# Calculate performance metrics
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

# Change detection and evaluation using Method A
def analyze_method_a_results(df, buses, window_size):
    p_values = [0.1, 0.2, 0.5, 0.7, 0.9]  # Set the values for p
    method_a_results = analyze_method_a(df, buses, window_size, p_values)
    return method_a_results

def save_results(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

# Change detection and evaluation using Method B
def analyze_method_b_results(df, buses, window_size):
    method_b_results = analyze_method_b(df, buses, window_size)
    return method_b_results

def save_results(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

# Change detection and evaluation using GEM
def analyze_gem_results(df, buses, d, k_values, alpha_values, h_values):
    gem_results = analyze_gem(df, buses, d, k_values, alpha_values, h_values)
    return gem_results

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


def main():
    file_path = './data/LMP.csv'
    buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
    n_samples = 855
    window_size = 24
    
    # Load and preprocess the data
    df = load_and_preprocess_data(file_path, buses, n_samples)
    
    # Calculate the statistical properties of the data and set parameters
    statistics = calculate_statistics(df, buses)
    
    # Change detection and evaluation using CUSUM
    cusum_threshold_values = [5, 10, 15]
    cusum_results = analyze_cusum(df, buses, statistics, cusum_threshold_values)
    save_results(cusum_results, './results/table/', 'cusum_analysis_results.csv')
    
    # Change detection and evaluation using GLR
    glr_threshold_values = [1, 2, 3]
    theta0_dict = {bus: statistics[bus]['mean_before'] for bus in buses}
    glr_results = analyze_glr(df, buses, statistics, glr_threshold_values, theta0_dict)
    save_results(glr_results, './results/table/', 'glr_analysis_results.csv')
    
    # Change detection and evaluation using Q-Q distance
    qq_threshold = 0.1  # Q-Q distance threshold
    qq_results = qq_detection(df, buses, window_size, qq_threshold)
    
    # Evaluate Q-Q results
    labels = df['Label'][window_size:].reset_index(drop=True)
    qq_performance = evaluate_performance(qq_results, labels)
    qq_results_df = pd.DataFrame([qq_performance], index=['Q-Q Detection'])
    save_results(qq_results_df, './results/table/', 'qq_analysis_results.csv')

    # Change detection and evaluation using Method A
    method_a_results = analyze_method_a_results(df, buses, window_size)
    save_results(method_a_results, './results/table/', 'method_a_analysis_results.csv')
    
    # Change detection and evaluation using Method B
    method_b_results = analyze_method_b_results(df, buses, window_size)
    save_results(method_b_results, './results/table/', 'method_b_analysis_results.csv')

    # Change detection and evaluation using GEM
    d = 3  # Transform to d-dimensions
    k_values = [10, 15, 20]
    alpha_values = [0.01, 0.05, 0.1, 1, 5, 10]
    h_values = [1, 5, 10, 20, 30]
    gem_results = analyze_gem_results(df, buses, d, k_values, alpha_values, h_values)
    save_results(gem_results, './results/table/', 'gem_analysis_results.csv')
    
    # Collecting results
    method_results_dict = {
        'CUSUM': cusum_results,
        'GLR': glr_results,
        'Q-Q': qq_results_df,
        'Method A': method_a_results,
        'Method B': method_b_results,
        'GEM': gem_results
    }
    
    # Generate and save summary table
    summary_table = generate_summary_table(method_results_dict)
    save_summary_table(summary_table, './results/table/comparative_analysis_summary.csv')

    # Visualization and plotting are handled in individual methods with enhanced plotting features.
    print("Analysis completed and results saved in './results/'")

if __name__ == '__main__':
    main()
