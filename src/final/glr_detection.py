# 以下はglr_detection.pyのコードです。
import numpy as np

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

