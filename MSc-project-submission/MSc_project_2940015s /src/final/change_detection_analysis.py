import pandas as pd
import os

# Input and output directories
input_dir = "./results/table/"
output_dir = "./results/table_processed/"
os.makedirs(output_dir, exist_ok=True)

# Function to filter and save data
def process_and_save(csv_file, output_file, filter_conditions):
    df = pd.read_csv(input_dir + csv_file)
    
    # Apply filters
    filtered_df = df
    for condition in filter_conditions:
        filtered_df = filtered_df[filtered_df[condition[0]] == condition[1]]
    
    # Save the filtered dataframe to the specified output file
    filtered_df.to_csv(output_dir + output_file, index=False)
    print(f"Processed data saved to {output_file}")

# CUSUM analysis
cusum_file = "cusum_analysis_results.csv"
process_and_save(cusum_file, "cusum_all_buses_fixed_threshold.csv", [("Cusum Threshold", 0.5)])
process_and_save(cusum_file, "cusum_bus_116_varying_threshold.csv", [("Bus", "Bus116")])

# GLR analysis
glr_file = "glr_analysis_results.csv"
process_and_save(glr_file, "glr_all_buses_fixed_threshold_window_size.csv", [("GLR Threshold", 2000), ("Window Size", 24)])
process_and_save(glr_file, "glr_bus_135_varying_threshold.csv", [("Bus", "Bus135"), ("Window Size", 24)])
process_and_save(glr_file, "glr_bus_135_varying_window_size.csv", [("Bus", "Bus135"), ("GLR Threshold", 2000)])

# QQ analysis
qq_file = "qq_analysis_results.csv"
process_and_save(qq_file, "qq_all_buses_fixed_threshold_window_size.csv", [("QQ Threshold", 0.1), ("Window Size", 24)])
process_and_save(qq_file, "qq_bus_119_varying_threshold.csv", [("Bus", "Bus119"), ("Window Size", 24)])
process_and_save(qq_file, "qq_bus_119_varying_window_size.csv", [("Bus", "Bus119"), ("QQ Threshold", 0.1)])

# GEM analysis
gem_file = "gem_analysis_results.csv"
# Initial analysis
process_and_save(gem_file, "gem_all_buses_fixed_alpha_threshold_k.csv", [("Alpha", 0.2), ("Threshold", 4), ("K", 16)])
# Bus-specific analysis 1: varying Threshold
process_and_save(gem_file, "gem_bus_115_varying_threshold.csv", [("Bus", "Bus115"), ("Alpha", 0.2), ("K", 16)])
# Bus-specific analysis 2: varying Alpha
process_and_save(gem_file, "gem_bus_115_varying_alpha.csv", [("Bus", "Bus115"), ("Threshold", 4), ("K", 16)])
# Bus-specific analysis 3: varying K
process_and_save(gem_file, "gem_bus_115_varying_k.csv", [("Bus", "Bus115"), ("Threshold", 4), ("Alpha", 0.2)])

# PCA analysis
pca_file = "pca_analysis_results.csv"
# Initial analysis
process_and_save(pca_file, "pca_all_buses_fixed_gamma_threshold_alpha.csv", [("Gamma", 0.9), ("Threshold", 4), ("Alpha", 0.2)])
# Bus-specific analysis 1: varying Threshold
process_and_save(pca_file, "pca_bus_115_varying_threshold.csv", [("Bus", "Bus115"), ("Gamma", 0.9), ("Alpha", 0.2)])
# Bus-specific analysis 2: varying Gamma
process_and_save(pca_file, "pca_bus_115_varying_gamma.csv", [("Bus", "Bus115"), ("Threshold", 4), ("Alpha", 0.2)])
# Bus-specific analysis 3: varying Alpha
process_and_save(pca_file, "pca_bus_115_varying_alpha.csv", [("Bus", "Bus115"), ("Threshold", 4), ("Gamma", 0.9)])
