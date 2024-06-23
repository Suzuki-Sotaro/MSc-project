import numpy as np
import matplotlib.pyplot as plt

def glr_detect(data, theta0, sigma, h, theta_min=0):
    n = len(data)
    g = np.zeros(n)
    t_hat = 0
    
    for k in range(1, n):
        max_likelihood = -np.inf
        max_j = 0
        
        for j in range(1, k+1):
            delta_j = np.abs(np.mean(data[j-1:k]) - theta0)
            if delta_j < theta_min:
                delta_j = theta_min
            
            likelihood = np.sum((delta_j/sigma**2) * (data[j-1:k] - theta0) - 0.5 * (delta_j**2/sigma**2))
            
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_j = j
        
        g[k-1] = max_likelihood
        
        if g[k-1] > h:
            t_hat = max_j
            break
    
    return t_hat, g

# 使用例
theta0 = 0
theta1 = 1
sigma = 1
h = 5
n = 100
change_point = 50

data = np.concatenate((np.random.normal(theta0, sigma, change_point),
                       np.random.normal(theta1, sigma, n - change_point)))

t_hat, g = glr_detect(data, theta0, sigma, h, theta_min=0.5)

print(f"Estimated change point: {t_hat}")

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(data)
plt.axvline(change_point, color='r', linestyle='--', label='True change point')
plt.axvline(t_hat, color='g', linestyle='--', label='Estimated change point')
plt.legend()
plt.title('Data with change point')
plt.xlabel('Time')
plt.ylabel('Value')

plt.subplot(122)
plt.plot(g)
plt.axhline(h, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title('GLR decision function')
plt.xlabel('Time')
plt.ylabel('Likelihood ratio')

plt.tight_layout()
plt.show()