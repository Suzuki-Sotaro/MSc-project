import pandas as pd
import numpy as np
from typing import Tuple, List

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

def load_lmp_data(file_path: str, selected_buses: List[int] = None) -> pd.DataFrame:
    """
    Load and preprocess the LMP data.
    
    Args:
    file_path (str): Path to the CSV file
    selected_buses (List[int], optional): List of bus numbers to select
    
    Returns:
    pd.DataFrame: Preprocessed data
    """
    print(f"Loading data from {file_path}...")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    print("\nFirst few rows of the loaded data:")
    print(df.head())
    
    # Select specific buses if provided
    if selected_buses:
        bus_columns = [f'Bus{i}' for i in selected_buses]
        df = df[['Week', 'Label'] + bus_columns]
        print(f"\nSelected buses: {selected_buses}")
        print(f"Data shape after selecting buses: {df.shape}")
    
    # Convert 'Week' to integer type
    df['Week'] = df['Week'].astype(int)
    
    # Print data info
    print("\nData Info:")
    df.info()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Print unique values in 'Label' column
    print("\nUnique values in 'Label' column:", df['Label'].unique())
    
    return df

def prepare_data_for_change_detection(df: pd.DataFrame, window_size: int, step_size: int) -> Tuple[np.ndarray, np.ndarray]:
    features = df.drop(['Week', 'Label'], axis=1).values
    
    # 移動平均フィルタを適用してノイズを減らす
    features_smoothed = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=features)
    
    windows = []
    labels = []
    for i in range(0, len(features_smoothed) - window_size + 1, step_size):
        windows.append(features_smoothed[i:i+window_size].flatten())
        labels.append(df['Label'].iloc[i+window_size-1])
    
    return np.array(windows), np.array(labels)

if __name__ == "__main__":
    # Example usage
    file_path = './data/LMP.csv'
    selected_buses = [115, 116, 117, 118, 119, 121, 135, 139]
    
    # Load data
    df = load_lmp_data(file_path, selected_buses)
    
    # Prepare data for change detection
    window_size = 24  # 1 day
    step_size = 1  # Move 1 hour at a time
    windows, labels = prepare_data_for_change_detection(df, window_size, step_size)
    
    # Print some additional information
    print("\nShape of the first window:", windows[0].shape)
    print("First few elements of the first window:", windows[0][:5])
    print("\nFirst few labels:", labels[:10])
    
    print("\nData loading and preparation completed.")