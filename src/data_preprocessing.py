# 以下はdata_preprocessing.pyのコード
import pandas as pd
import numpy as np
from typing import List, Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from the CSV file.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def extract_bus_data(data: pd.DataFrame, bus_numbers: List[int]) -> pd.DataFrame:
    """
    Extract data for specified bus numbers.
    
    Args:
    data (pd.DataFrame): Full dataset.
    bus_numbers (List[int]): List of bus numbers to extract.
    
    Returns:
    pd.DataFrame: Extracted data for specified buses.
    """
    bus_columns = [f'Bus{num}' for num in bus_numbers]
    return data[['Week', 'Label'] + bus_columns]

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the data using min-max normalization.
    
    Args:
    data (pd.DataFrame): Data to normalize.
    
    Returns:
    pd.DataFrame: Normalized data.
    """
    min_vals = data.min()
    max_vals = data.max()
    return (data - min_vals) / (max_vals - min_vals)

def get_attack_period(data: pd.DataFrame) -> Tuple[int, int]:
    """
    Get the start and end indices of the attack period.
    
    Args:
    data (pd.DataFrame): Data containing the Label column.
    
    Returns:
    Tuple[int, int]: Start and end indices of the attack period.
    """
    attack_start = data[data['Label'] == 1].index[0]
    attack_end = data[data['Label'] == 1].index[-1]
    return attack_start, attack_end

def preprocess_data(file_path: str, bus_numbers: List[int], num_samples: int = 855) -> Tuple[pd.DataFrame, int, int]:
    """
    Main function to preprocess the data.
    
    Args:
    file_path (str): Path to the CSV file.
    bus_numbers (List[int]): List of bus numbers to extract.
    num_samples (int): Number of samples to use from the end of the dataset.
    
    Returns:
    Tuple[pd.DataFrame, int, int]: Preprocessed data, attack start index, and attack end index.
    """
    # Load the data
    data = load_data(file_path)
    
    # Extract data for specified buses
    bus_data = extract_bus_data(data, bus_numbers)
    
    # Get the last num_samples
    bus_data = bus_data.tail(num_samples).reset_index(drop=True)
    
    # Normalize the data
    normalized_data = normalize_data(bus_data.drop(['Week', 'Label'], axis=1))
    
    # Add back the Week and Label columns
    normalized_data = pd.concat([bus_data[['Week', 'Label']], normalized_data], axis=1)
    
    # Get the attack period
    attack_start, attack_end = get_attack_period(normalized_data)
    
    return normalized_data, attack_start, attack_end

if __name__ == "__main__":
    file_path = './data/LMP.csv'
    bus_numbers = [115, 116, 117, 118, 119, 121, 135, 139]
    
    preprocessed_data, attack_start, attack_end = preprocess_data(file_path, bus_numbers)
    
    print("Preprocessed data shape:", preprocessed_data.shape)
    print("Attack period:", attack_start, "-", attack_end)
    print(preprocessed_data.head())
    print(preprocessed_data.tail())