import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import numpy as np

# Input directory
input_dir = "./results/table/"
output_dir = "./results/plots/"
os.makedirs(output_dir, exist_ok=True)

# Helper function to plot data for Method A and B
def plot_anomalies(x, y_true, y_pred, method_name, ax):
    # Plot True Anomalies and Detected Anomalies
    ax.plot(x, y_true, label='True Anomalies', color='blue', linestyle='--')
    ax.plot(x, y_pred, label='Detected Anomalies', color='red', linestyle='-')
    ax.set_title(f'{method_name}')
    ax.legend()

# Function to visualize all methods in one plot
def visualize_methods(csv_file, output_file):
    df = pd.read_csv(input_dir + csv_file)
    
    methods = df['Method'].unique()
    fig, axs = plt.subplots(len(methods), 1, figsize=(10, 4*len(methods)), sharex=True)
    fig.suptitle(f'{csv_file.replace("_", " ").replace(".csv", "").title()}')

    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        for _, row in method_data.iterrows():
            true_labels = ast.literal_eval(row['Label'])
            detected_labels = ast.literal_eval(row['Detection'])
            
            # Ensure the x axis matches the length of the labels
            x = range(len(true_labels))
            plot_anomalies(x, true_labels, detected_labels, method, axs[i])
    
    plt.xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add spacing between title and plots
    plt.savefig(output_dir + output_file)
    plt.show()

# Visualizing methods for all algorithms
visualize_methods("cusum_analysis_results_method_ab.csv", "cusum_analysis_results_method_ab.png")
visualize_methods("gem_analysis_results_method_ab.csv", "gem_analysis_results_method_ab.png")
visualize_methods("glr_analysis_results_method_ab.csv", "glr_analysis_results_method_ab.png")
visualize_methods("pca_analysis_results_method_ab.csv", "pca_analysis_results_method_ab.png")
visualize_methods("qq_analysis_results_method_ab.csv", "qq_analysis_results_method_ab.png")
