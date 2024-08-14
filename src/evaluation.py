import pandas as pd
import os

def load_results(output_dir):
    """
    Load the results from the output directory where the experiment results are stored.
    
    Args:
        output_dir (str): The directory where results are saved.
    
    Returns:
        dict: A dictionary containing DataFrames of results from different methods.
    """
    print(f"Loading results from {output_dir}...")
    
    results = {}
    
    try:
        results['CUSUM'] = pd.read_csv(os.path.join(output_dir, 'cusum_results.csv'))
        results['Q-Q Distance'] = pd.read_csv(os.path.join(output_dir, 'qq_results.csv'))
        results['k-NN'] = pd.read_csv(os.path.join(output_dir, 'knn_results.csv'))
        results['Method A'] = pd.read_csv(os.path.join(output_dir, 'method_a_results.csv'))
        results['Method B'] = pd.read_csv(os.path.join(output_dir, 'method_b_results.csv'))
        print("Results loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
        raise
    
    return results

def compare_methods(results):
    """
    Compare the performance of different change detection methods using evaluation metrics.
    
    Args:
        results (dict): A dictionary containing DataFrames of results from different methods.
    
    Returns:
        pd.DataFrame: A DataFrame summarizing the comparison of different methods.
    """
    print("Comparing methods based on performance metrics...")
    
    comparison = []
    
    for method, df in results.items():
        print(f"\nEvaluating {method}...")
        avg_accuracy = df['Performance'].apply(lambda x: eval(x)['Accuracy']).mean()
        avg_precision = df['Performance'].apply(lambda x: eval(x)['Precision']).mean()
        avg_recall = df['Performance'].apply(lambda x: eval(x)['Recall']).mean()
        avg_f1 = df['Performance'].apply(lambda x: eval(x)['F1 Score']).mean()
        
        print(f"{method} - Average Accuracy: {avg_accuracy:.4f}")
        print(f"{method} - Average Precision: {avg_precision:.4f}")
        print(f"{method} - Average Recall: {avg_recall:.4f}")
        print(f"{method} - Average F1 Score: {avg_f1:.4f}")
        
        comparison.append({
            'Method': method,
            'Average Accuracy': avg_accuracy,
            'Average Precision': avg_precision,
            'Average Recall': avg_recall,
            'Average F1 Score': avg_f1
        })
    
    comparison_df = pd.DataFrame(comparison)
    return comparison_df

def save_comparison(comparison_df, output_dir):
    """
    Save the comparison results to a CSV file in the output directory.
    
    Args:
        comparison_df (pd.DataFrame): The DataFrame containing the comparison results.
        output_dir (str): The directory where the comparison results will be saved.
    """
    output_file = os.path.join(output_dir, 'method_comparison.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"Comparison results saved to {output_file}")

def main():
    # Directory where the results are stored
    output_dir = './results'
    
    # Load the results
    results = load_results(output_dir)
    
    # Compare the methods based on performance
    comparison_df = compare_methods(results)
    
    # Display the comparison
    print("\nComparison of Change Detection Methods:")
    print(comparison_df)
    
    # Save the comparison results
    save_comparison(comparison_df, output_dir)
    
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
