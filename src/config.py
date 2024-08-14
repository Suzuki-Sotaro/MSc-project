# config.py

# General Settings
DATA_FILE_PATH = './data/LMP.csv'        # Path to the CSV file containing the LMP data
OUTPUT_DIR = './results'                 # Directory where results will be saved

# Data Preprocessing
NORMALIZATION_METHOD = 'minmax'          # Method for normalizing data ('minmax', 'zscore', etc.)
MISSING_VALUE_STRATEGY = 'mean'          # Strategy for handling missing values ('mean', 'median', 'drop')

# Experiment Settings
SPLIT_RATIO = 0.7                        # Ratio for splitting data into training and testing sets

# CUSUM Analysis Parameters
CUSUM_THRESHOLD = 10                     # Threshold for detecting changes using CUSUM
CUSUM_DRIFT = 0.5                        # Drift parameter for the CUSUM algorithm

# Q-Q Distance Analysis Parameters
QQ_WINDOW_SIZE = 30                      # Window size for the Q-Q distance method
QQ_THRESHOLD = 15                        # Threshold for detecting changes using Q-Q distance
QQ_SLIDING_STEP = 1                      # Sliding step size for the Q-Q distance method

# k-NN and CUSUM Analysis Parameters
KNN_K = 10                               # Number of nearest neighbors for k-NN
KNN_ALPHA = 0.05                         # Tail probability threshold for detecting outliers
KNN_CUSUM_THRESHOLD = 15                 # Threshold for the CUSUM algorithm when applied to k-NN outliers

# Method A (Decision Making via Fusion) Parameters
METHOD_A_THRESHOLDS = [0.5] * 140        # Local detection thresholds for each bus (adjust length based on number of buses)
METHOD_A_P_VALUES = [0.1, 0.5, 0.9]      # Values of p to experiment with for global decision making

# Method B (Aggregation-based Decision Making) Parameters
METHOD_B_LOCAL_THRESHOLDS = [0.5] * 140  # Local thresholds for each bus (adjust length based on number of buses)
METHOD_B_GLOBAL_THRESHOLD = 1.0          # Global threshold for detecting changes based on aggregated statistics
METHOD_B_AGGREGATION_METHODS = ['average', 'median', 'minimum', 'maximum']  # Aggregation methods to experiment with

# Visualization Settings
PLOT_TITLE_FONT_SIZE = 16                # Font size for plot titles
PLOT_LABEL_FONT_SIZE = 14                # Font size for plot labels
PLOT_LEGEND_FONT_SIZE = 12               # Font size for plot legends

# Debugging and Logging
VERBOSE = True                           # Set to True to print detailed logs during execution
SAVE_PLOTS = True                        # Set to True to save plots to files

# Other Settings
RANDOM_SEED = 42                         # Random seed for reproducibility

