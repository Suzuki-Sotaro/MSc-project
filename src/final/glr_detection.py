# below is the content of glr_detection.py
import numpy as np
import matplotlib.pyplot as plt

def glr_detect(data, theta0, sigma, h, nu_min=0):
    n = len(data)
    g = np.zeros(n)
    t_hat = 0
    
    for k in range(1, n):
        max_likelihood = -np.inf
        max_j = 0
        
        for j in range(1, k+1):
            mean_diff = np.mean(data[j-1:k]) - theta0
            nu = max(mean_diff, nu_min)
            
            likelihood = np.sum((nu/sigma**2) * (data[j-1:k] - theta0) - 0.5 * (nu**2/sigma**2))
            
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_j = j
        
        g[k-1] = max_likelihood
        
        if g[k-1] > h:
            t_hat = max_j
            break
    
    return t_hat, g

def plot_glr_results(data, change_point, glr_scores, detection_points, threshold_values, method_name, save_path=None):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    ax1.plot(data, label='Data')
    ax1.axvline(change_point, color='red', linestyle='--', label='True Change Point')

    for i, (threshold, glr_score) in enumerate(zip(threshold_values, glr_scores)):
        if detection_points[i] != -1:
            ax1.axvline(detection_points[i], color=f'C{i}', linestyle='--', label=f'Detected Change (h={threshold})')
        ax1.plot(glr_score, color=f'C{i}', label=f'GLR Scores (h={threshold})')
        ax1.axhline(threshold, color=f'C{i}', linestyle='--', label=f'Threshold (h={threshold})')

    ax1.set_xlabel('Sample')
    ax1.set_ylabel('GLR Score')
    ax1.set_title(f'{method_name} Change Detection')
    ax1.legend()

    plt.tight_layout()

    # Save plot to file if save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
