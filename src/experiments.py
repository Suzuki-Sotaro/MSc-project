import pandas as pd
from data_loader import load_data, preprocess_data, split_data
from cusum import run_cusum_analysis
from qq_distance import run_qq_analysis
from knn_anomaly import run_knn_cusum_analysis
from method_a import run_method_a
from method_b import run_method_b

def run_all_experiments(data_file_path, output_dir):
    """
    Run all experiments using various change detection methods and save the results.
    
    Args:
        data_file_path (str): The path to the CSV file containing the LMP data.
        output_dir (str): The directory where results will be saved.
    """
    print("Starting all experiments...")
    
    # Step 1: Load and preprocess the data
    data = load_data(data_file_path)
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets (use the full dataset for now)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("\n=== Running CUSUM Analysis ===")
    cusum_results = run_cusum_analysis(X_train, y_train, threshold=10, drift=0.5)
    
    print("\n=== Running Q-Q Distance Analysis ===")
    qq_results = run_qq_analysis(X_train, y_train, window_size=30, threshold=15, sliding_step=1)
    
    print("\n=== Running k-NN and CUSUM Analysis ===")
    knn_results = run_knn_cusum_analysis(X_train, y_train, k=10, alpha=0.05, cusum_threshold=15)
    
    print("\n=== Running Method A (Decision Making via Fusion) ===")
    thresholds = [0.5] * X_train.shape[1]  # Assuming a single threshold for simplicity
    p_values = [0.1, 0.5, 0.9]
    method_a_results = run_method_a(X_train, y_train, thresholds, p_values)
    
    print("\n=== Running Method B (Aggregation-based Decision Making) ===")
    local_thresholds = [0.5] * X_train.shape[1]  # Assuming a single threshold for simplicity
    global_threshold = 1.0
    aggregation_methods = ['average', 'median', 'minimum', 'maximum']
    method_b_results = run_method_b(X_train, y_train, local_thresholds, global_threshold, aggregation_methods)
    
    # Step 5: Save the results
    print("\nSaving results to the output directory...")
    
    # Convert results to DataFrames for saving
    cusum_df = pd.DataFrame(cusum_results)
    qq_df = pd.DataFrame(qq_results)
    knn_df = pd.DataFrame(knn_results)
    method_a_df = pd.DataFrame(method_a_results)
    method_b_df = pd.DataFrame(method_b_results)
    
    # Save to CSV files
    cusum_df.to_csv(f"{output_dir}/cusum_results.csv", index=False)
    qq_df.to_csv(f"{output_dir}/qq_results.csv", index=False)
    knn_df.to_csv(f"{output_dir}/knn_results.csv", index=False)
    method_a_df.to_csv(f"{output_dir}/method_a_results.csv", index=False)
    method_b_df.to_csv(f"{output_dir}/method_b_results.csv", index=False)
    
    print("All results saved successfully.")

def main():
    # Path to the data file
    data_file_path = './data/LMP.csv'
    
    # Directory to save the results
    output_dir = './results'
    
    # Run all experiments
    run_all_experiments(data_file_path, output_dir)
    
    print("All experiments completed.")

if __name__ == "__main__":
    main()
