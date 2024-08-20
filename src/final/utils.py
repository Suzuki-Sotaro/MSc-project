import numpy as np

def calculate_far_ed(true_labels, predicted_labels, detection_time):
    far = np.sum((predicted_labels == 1) & (true_labels == 0)) / np.sum(true_labels == 0)
    
    if detection_time != -1 and np.any(true_labels == 1):
        true_change_point = np.argmax(true_labels)
        ed = max(0, detection_time - true_change_point)
    else:
        ed = np.inf
    
    return far, ed