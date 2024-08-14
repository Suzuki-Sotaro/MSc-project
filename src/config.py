# config.py

# Data loading parameters
DATA_PATH = './data/LMP.csv'
SELECTED_BUSES = [115, 116, 117, 118, 119, 121, 135, 139]

# Data preprocessing parameters
WINDOW_SIZE = 24  # 1 day
STEP_SIZE = 1  # Move 1 hour at a time

# config.py
GEM_K = 5  # この値は変更せずに保持
GEM_ALPHA = 0.01  # 0.1から0.01に変更
GEM_H = 20  # 10から20に変更

QQ_WINDOW_SIZE = 100  # 50から100に変更
QQ_H = 10  # 5から10に変更

METHOD_A_P_VALUES = [0.3, 0.5, 0.7]  # 値の範囲を狭める

METHOD_B_H_METHODS = ['mean', 'median']  # 'min'と'max'を除外
METHOD_B_MAD_THRESHOLD = 2.5  # 3.5から2.5に変更

# Performance evaluation parameters
DETECTION_TOLERANCE = 5  # Tolerance window for matching true and detected changes

# Visualization parameters
PLOT_FIGSIZE = (12, 6)

# Random seed for reproducibility
RANDOM_SEED = 42

# Experiment settings
N_EXPERIMENTS = 10  # Number of experiments to run for each configuration

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FILE = 'change_detection_experiments.log'

# Result storage
RESULTS_DIR = './results'

# Debug mode flag
DEBUG = False

# Add any other configuration parameters you need for your project here