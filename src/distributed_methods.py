import numpy as np
from typing import List, Tuple
from scipy.stats import median_abs_deviation

def method_a(local_decisions: np.ndarray, p: float) -> Tuple[List[int], np.ndarray]:
    """
    Implement Method A for distributed change detection.
    
    Args:
    local_decisions (np.ndarray): Binary decisions from each bus at each time step
    p (float): Percentage of buses required to declare a change
    
    Returns:
    Tuple[List[int], np.ndarray]: Detected change points and global decisions
    """
    print(f"Running Method A with p = {p}")
    n_timesteps, n_buses = local_decisions.shape
    print(f"Input shape: {local_decisions.shape} (timesteps, buses)")
    
    threshold = int(np.ceil(p * n_buses))
    print(f"Threshold for change detection: {threshold} buses")
    
    global_decisions = np.sum(local_decisions, axis=1) >= threshold
    change_points = np.where(global_decisions)[0].tolist()
    
    print(f"Detected {len(change_points)} change points")
    for cp in change_points:
        print(f"Change detected at time {cp}")
    
    return change_points, global_decisions

def method_b_average(local_statistics: np.ndarray, h_method: str, local_thresholds: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """
    Implement Method B with average aggregation for distributed change detection.
    
    Args:
    local_statistics (np.ndarray): Local statistics from each bus at each time step
    h_method (str): Method to determine global threshold ('mean', 'min', 'max', or 'median')
    local_thresholds (np.ndarray): Local thresholds for each bus
    
    Returns:
    Tuple[List[int], np.ndarray]: Detected change points and global statistics
    """
    print(f"Running Method B (Average) with h_method = {h_method}")
    n_timesteps, n_buses = local_statistics.shape
    print(f"Input shape: {local_statistics.shape} (timesteps, buses)")
    
    if h_method == 'mean':
        h = np.mean(local_thresholds)
    elif h_method == 'min':
        h = np.min(local_thresholds)
    elif h_method == 'max':
        h = np.max(local_thresholds)
    elif h_method == 'median':
        h = np.median(local_thresholds)
    else:
        raise ValueError("Invalid h_method. Choose 'mean', 'min', 'max', or 'median'.")
    
    print(f"Global threshold H = {h:.4f}")
    
    global_statistics = np.mean(local_statistics, axis=1)
    global_decisions = global_statistics > h
    change_points = np.where(global_decisions)[0].tolist()
    
    print(f"Detected {len(change_points)} change points")
    for cp in change_points:
        print(f"Change detected at time {cp}")
    
    return change_points, global_statistics

def method_b_median(local_statistics: np.ndarray, h_method: str, local_thresholds: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """
    Implement Method B with median aggregation for distributed change detection.
    
    Args:
    local_statistics (np.ndarray): Local statistics from each bus at each time step
    h_method (str): Method to determine global threshold ('mean', 'min', 'max', or 'median')
    local_thresholds (np.ndarray): Local thresholds for each bus
    
    Returns:
    Tuple[List[int], np.ndarray]: Detected change points and global statistics
    """
    print(f"Running Method B (Median) with h_method = {h_method}")
    n_timesteps, n_buses = local_statistics.shape
    print(f"Input shape: {local_statistics.shape} (timesteps, buses)")
    
    if h_method == 'mean':
        h = np.mean(local_thresholds)
    elif h_method == 'min':
        h = np.min(local_thresholds)
    elif h_method == 'max':
        h = np.max(local_thresholds)
    elif h_method == 'median':
        h = np.median(local_thresholds)
    else:
        raise ValueError("Invalid h_method. Choose 'mean', 'min', 'max', or 'median'.")
    
    print(f"Global threshold H = {h:.4f}")
    
    global_statistics = np.median(local_statistics, axis=1)
    global_decisions = global_statistics > h
    change_points = np.where(global_decisions)[0].tolist()
    
    print(f"Detected {len(change_points)} change points")
    for cp in change_points:
        print(f"Change detected at time {cp}")
    
    return change_points, global_statistics

def method_b_mad(local_statistics: np.ndarray, h_method: str, local_thresholds: np.ndarray, mad_threshold: float = 3.5) -> Tuple[List[int], np.ndarray]:
    """
    Implement Method B with MAD-based outlier removal for distributed change detection.
    
    Args:
    local_statistics (np.ndarray): Local statistics from each bus at each time step
    h_method (str): Method to determine global threshold ('mean', 'min', 'max', or 'median')
    local_thresholds (np.ndarray): Local thresholds for each bus
    mad_threshold (float): Threshold for MAD-based outlier detection
    
    Returns:
    Tuple[List[int], np.ndarray]: Detected change points and global statistics
    """
    print(f"Running Method B (MAD) with h_method = {h_method}, mad_threshold = {mad_threshold}")
    n_timesteps, n_buses = local_statistics.shape
    print(f"Input shape: {local_statistics.shape} (timesteps, buses)")
    
    if h_method == 'mean':
        h = np.mean(local_thresholds)
    elif h_method == 'min':
        h = np.min(local_thresholds)
    elif h_method == 'max':
        h = np.max(local_thresholds)
    elif h_method == 'median':
        h = np.median(local_thresholds)
    else:
        raise ValueError("Invalid h_method. Choose 'mean', 'min', 'max', or 'median'.")
    
    print(f"Global threshold H = {h:.4f}")
    
    global_statistics = []
    for t in range(n_timesteps):
        stats = local_statistics[t]
        median = np.median(stats)
        mad = median_abs_deviation(stats)
        
        lower_bound = median - mad_threshold * mad
        upper_bound = median + mad_threshold * mad
        
        non_outliers = stats[(stats >= lower_bound) & (stats <= upper_bound)]
        global_stat = np.mean(non_outliers)
        global_statistics.append(global_stat)
        
        if t % 100 == 0:
            print(f"Processed timestep {t}: removed {len(stats) - len(non_outliers)} outliers")
    
    global_statistics = np.array(global_statistics)
    global_decisions = global_statistics > h
    change_points = np.where(global_decisions)[0].tolist()
    
    print(f"Detected {len(change_points)} change points")
    for cp in change_points:
        print(f"Change detected at time {cp}")
    
    return change_points, global_statistics

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_timesteps, n_buses = 1000, 10
    
    # Generate synthetic local decisions for Method A
    local_decisions = np.random.randint(0, 2, size=(n_timesteps, n_buses))
    local_decisions[500:, :] = 1  # Simulate a change at t=500
    
    print("Testing Method A:")
    change_points_a, global_decisions_a = method_a(local_decisions, p=0.5)
    
    # Generate synthetic local statistics for Method B
    local_statistics = np.random.randn(n_timesteps, n_buses)
    local_statistics[500:, :] += 2  # Simulate a change at t=500
    local_thresholds = np.random.uniform(1, 3, size=n_buses)
    
    print("\nTesting Method B (Average):")
    change_points_b_avg, global_statistics_b_avg = method_b_average(local_statistics, h_method='mean', local_thresholds=local_thresholds)
    
    print("\nTesting Method B (Median):")
    change_points_b_med, global_statistics_b_med = method_b_median(local_statistics, h_method='mean', local_thresholds=local_thresholds)
    
    print("\nTesting Method B (MAD):")
    change_points_b_mad, global_statistics_b_mad = method_b_mad(local_statistics, h_method='mean', local_thresholds=local_thresholds)
    
    # You can add more visualization or analysis here if needed