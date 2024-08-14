import os
from config import DATA_FILE_PATH, OUTPUT_DIR, VERBOSE
from data_loader import load_data, preprocess_data, split_data
from experiments import run_all_experiments
from evaluation import load_results, compare_methods, save_comparison
from utils import ensure_directory_exists

def main():
    """
    Main execution function for running experiments and evaluating results.
    """
    # Ensure the output directory exists
    ensure_directory_exists(OUTPUT_DIR)
    
    # Step 1: Load and preprocess the data
    if VERBOSE:
        print("Loading and preprocessing data...")
    data = load_data(DATA_FILE_PATH)
    X, y = preprocess_data(data)
    
    # Step 2: Run all experiments
    if VERBOSE:
        print("Running experiments...")
    run_all_experiments(DATA_FILE_PATH, OUTPUT_DIR)
    
    # Step 3: Load and evaluate the results
    if VERBOSE:
        print("Loading and evaluating results...")
    results = load_results(OUTPUT_DIR)
    comparison_df = compare_methods(results)
    
    # Step 4: Display the comparison
    if VERBOSE:
        print("\nFinal Comparison of Change Detection Methods:")
        print(comparison_df)
    
    # Step 5: Save the comparison results
    if VERBOSE:
        print("Saving comparison results...")
    save_comparison(comparison_df, OUTPUT_DIR)
    
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
