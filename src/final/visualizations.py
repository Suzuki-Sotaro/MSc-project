import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import numpy as np

# Input directory
input_dir = "./results/table_processed/"
output_dir = "./results/plots/"
os.makedirs(output_dir, exist_ok=True)

# Helper function to plot data
def plot_data(x, data, y_true, y_pred, title, bus_name, ax, label):
    # Normalize data to [0, 1] range
    data_min, data_max = min(data), max(data)
    normalized_data = [(val - data_min) / (data_max - data_min) for val in data]
    
    # Adjust labels to avoid empty parentheses
    data_label = 'Data' + (f' ({label})' if label else '')
    true_label = 'True Anomalies' + (f' ({label})' if label else '')
    detected_label = 'Detected Anomalies' + (f' ({label})' if label else '')
    
    ax.plot(x, normalized_data, label=data_label, color='green', linestyle='-.')
    ax.plot(x, y_true, label=true_label, color='blue', linestyle='--')
    ax.plot(x, y_pred, label=detected_label, color='red', linestyle='-')
    ax.set_title(bus_name)
    ax.legend()

# Function to visualize all buses in one plot
def visualize_all_buses(csv_file, output_file):
    df = pd.read_csv(input_dir + csv_file)
    
    buses = df['Bus'].unique()
    fig, axs = plt.subplots(len(buses), 1, figsize=(10, 3*len(buses)), sharex=True)
    fig.suptitle(f'{csv_file.replace("_", " ").replace(".csv", "").title()}')
    
    for i, bus in enumerate(buses):
        bus_data = df[df['Bus'] == bus]
        for _, row in bus_data.iterrows():
            data = ast.literal_eval(row['Data'])
            true_labels = ast.literal_eval(row['Label'])
            detected_labels = ast.literal_eval(row['Detection'])
            plot_data(range(len(data)), data, true_labels, detected_labels, f"{bus}", bus, axs[i], "")
    
    plt.xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add spacing between title and plots
    plt.savefig(output_dir + output_file)
    plt.show()

# Function to visualize one bus with varying parameters
def visualize_single_bus_varying_params(csv_file, output_file, varying_param):
    df = pd.read_csv(input_dir + csv_file)
    
    unique_params = df[varying_param].unique()
    fig, axs = plt.subplots(len(unique_params), 1, figsize=(10, 3*len(unique_params)), sharex=True)
    fig.suptitle(f'{csv_file.replace("_", " ").replace(".csv", "").title()}')

    for i, param_value in enumerate(unique_params):
        param_data = df[df[varying_param] == param_value]
        for _, row in param_data.iterrows():
            data = ast.literal_eval(row['Data'])
            true_labels = ast.literal_eval(row['Label'])
            detected_labels = ast.literal_eval(row['Detection'])
            label = f'{varying_param}: {param_value}'
            plot_data(range(len(data)), data, true_labels, detected_labels, f"{row['Bus']}", row['Bus'], axs[i], label)
    
    plt.xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add spacing between title and plots
    plt.savefig(output_dir + output_file)
    plt.show()

# Visualizing all buses
visualize_all_buses("cusum_all_buses_fixed_threshold.csv", "cusum_all_buses_fixed_threshold.png")
visualize_all_buses("glr_all_buses_fixed_threshold.csv", "glr_all_buses_fixed_threshold.png")
visualize_all_buses("qq_all_buses_fixed_threshold.csv", "qq_all_buses_fixed_threshold.png")
visualize_all_buses("gem_all_buses_fixed_alpha_threshold.csv", "gem_all_buses_fixed_alpha_threshold.png")
visualize_all_buses("pca_all_buses_fixed_gamma_threshold.csv", "pca_all_buses_fixed_gamma_threshold.png")

# Visualizing single bus with varying parameters
visualize_single_bus_varying_params("cusum_bus_116_varying_threshold.csv", "cusum_bus_116_varying_threshold.png", "Cusum Threshold")
visualize_single_bus_varying_params("glr_bus_135_varying_threshold.csv", "glr_bus_135_varying_threshold.png", "GLR Threshold")
visualize_single_bus_varying_params("qq_bus_119_varying_threshold.csv", "qq_bus_119_varying_threshold.png", "QQ Threshold")
visualize_single_bus_varying_params("gem_bus_115_varying_threshold.csv", "gem_bus_115_varying_threshold.png", "Threshold")
visualize_single_bus_varying_params("gem_bus_115_varying_alpha.csv", "gem_bus_115_varying_alpha.png", "Alpha")
visualize_single_bus_varying_params("pca_bus_115_varying_threshold.csv", "pca_bus_115_varying_threshold.png", "Threshold")
visualize_single_bus_varying_params("pca_bus_115_varying_gamma.csv", "pca_bus_115_varying_gamma.png", "Gamma")
