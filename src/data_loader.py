import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the LMP dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the LMP data.
    
    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape {data.shape}")
    return data

def preprocess_data(data):
    """
    Preprocess the LMP data by handling missing values, normalizing, and splitting into features and labels.
    
    Args:
        data (pd.DataFrame): Raw LMP data.
    
    Returns:
        tuple: Tuple containing:
            - X (pd.DataFrame): Feature data (buses' LMP values).
            - y (pd.Series): Labels indicating change points.
    """
    print("Preprocessing data...")

    # Handle missing values
    if data.isnull().values.any():
        print("Missing values detected. Filling missing values with the mean of each column.")
        data.fillna(data.mean(), inplace=True)
    
    # Normalize the feature columns (3rd column onwards are buses' LMP values)
    X = data.iloc[:, 2:].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    y = data.iloc[:, 1]  # Assuming the second column is the label

    print("Data preprocessing complete.")
    print(f"Feature data shape: {X.shape}")
    print(f"Label data shape: {y.shape}")
    
    return X, y

def split_data(X, y, train_size=0.7):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Labels.
        train_size (float): Proportion of the data to be used for training.
    
    Returns:
        tuple: Tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training labels.
            - y_test (pd.Series): Testing labels.
    """
    print(f"Splitting data into training and testing sets with train size {train_size}...")

    split_idx = int(len(X) * train_size)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print("Data split complete.")
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main():
    # Path to the data file
    data_file_path = './data/LMP.csv'
    
    # Load the data
    data = load_data(data_file_path)
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Data loading and preprocessing pipeline completed successfully.")
    
if __name__ == "__main__":
    main()
