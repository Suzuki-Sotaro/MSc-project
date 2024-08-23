# This the content of utils.py  
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_far_ed(true_labels, predicted_labels, detection_time):
    far = np.sum((predicted_labels == 1) & (true_labels == 0)) / np.sum(true_labels == 0)
    
    if detection_time != -1 and np.any(true_labels == 1):
        true_change_point = np.argmax(true_labels)
        ed = max(0, detection_time - true_change_point)
    else:
        ed = np.inf
    
    return far, ed

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1