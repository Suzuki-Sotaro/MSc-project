import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists. If it does not, create it.
    
    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def plot_time_series(data, title="Time Series Data", xlabel="Time", ylabel="Value", legend=None):
    """
    Plot a time series with optional title, labels, and legend.
    
    Args:
        data (np.ndarray or pd.Series): The time series data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        legend (list or None): The legend labels for the plot.
    """
    plt.figure(figsize=(10, 5))
    if isinstance(data, np.ndarray):
        plt.plot(data)
    else:
        data.plot()
    
    if legend:
        plt.legend(legend)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_comparison(data_list, labels, title="Comparison Plot", xlabel="Time", ylabel="Value"):
    """
    Plot multiple time series for comparison on the same plot.
    
    Args:
        data_list (list of np.ndarray or list of pd.Series): List of time series data to plot.
        labels (list of str): List of labels for each time series.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(10, 5))
    
    for data, label in zip(data_list, labels):
        if isinstance(data, np.ndarray):
            plt.plot(data, label=label)
        else:
            data.plot(label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def print_summary_statistics(data, label="Data"):
    """
    Print summary statistics (mean, standard deviation, min, max) for the given data.
    
    Args:
        data (np.ndarray or pd.Series): The data to summarize.
        label (str): The label to describe the data (for output purposes).
    """
    mean_value = np.mean(data)
    std_value = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)
    
    print(f"Summary Statistics for {label}:")
    print(f"Mean: {mean_value:.4f}")
    print(f"Standard Deviation: {std_value:.4f}")
    print(f"Minimum: {min_value:.4f}")
    print(f"Maximum: {max_value:.4f}")

def save_plot_to_file(data, file_path, title="Plot", xlabel="X-axis", ylabel="Y-axis", legend=None):
    """
    Save a plot of the data to a specified file.
    
    Args:
        data (np.ndarray or pd.Series): The data to plot.
        file_path (str): The path where the plot should be saved.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        legend (list or None): The legend labels for the plot.
    """
    plt.figure(figsize=(10, 5))
    if isinstance(data, np.ndarray):
        plt.plot(data)
    else:
        data.plot()
    
    if legend:
        plt.legend(legend)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    ensure_directory_exists(os.path.dirname(file_path))
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved to {file_path}")

def compute_rolling_mean(data, window_size):
    """
    Compute the rolling mean of the given data with a specified window size.
    
    Args:
        data (np.ndarray or pd.Series): The data for which to compute the rolling mean.
        window_size (int): The window size for the rolling mean.
    
    Returns:
        np.ndarray: The rolling mean of the data.
    """
    if isinstance(data, np.ndarray):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    else:
        return data.rolling(window=window_size).mean()

def compute_rolling_std(data, window_size):
    """
    Compute the rolling standard deviation of the given data with a specified window size.
    
    Args:
        data (np.ndarray or pd.Series): The data for which to compute the rolling standard deviation.
        window_size (int): The window size for the rolling standard deviation.
    
    Returns:
        np.ndarray: The rolling standard deviation of the data.
    """
    if isinstance(data, np.ndarray):
        return np.sqrt(np.convolve((data - np.mean(data)) ** 2, np.ones(window_size) / window_size, mode='valid'))
    else:
        return data.rolling(window=window_size).std()

