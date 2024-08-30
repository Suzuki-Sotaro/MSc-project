# This the content of utils.py  
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_far_ed(true_labels, predicted_labels, change_point):
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Extract false positives (FP) and true negatives (TN) from the confusion matrix
    fp = cm[0, 1]  # False Positives: actual 0 but predicted 1
    tn = cm[0, 0]  # True Negatives: actual 0 and predicted 0

    # Calculate FAR (False Alarm Rate)
    if (tn + fp) == 0:
        far = 0.0  # No true negatives, FAR should be set to 0.0 or another appropriate value
    else:
        far = fp / (fp + tn)
    
    # Identify change points from true_labels and predicted_labels
    true_change_points = [i for i in range(1, len(true_labels)) if true_labels[i] != true_labels[i-1]]
    predicted_change_points = [i for i in range(1, len(predicted_labels)) if predicted_labels[i] != predicted_labels[i-1]]
    
    # Calculate ED (Expected Delay)
    delays = []
    for true_cp in true_change_points:
        # Find the closest predicted change point relative to the true change point
        closest_delay = float('inf')
        for predicted_cp in predicted_change_points:
            delay = predicted_cp - true_cp
            if delay >= 0 and delay < closest_delay:  # We only care about non-negative delays
                closest_delay = delay
            elif delay < 0 and abs(delay) < closest_delay:  # Handle early predictions with -1
                closest_delay = -1
        
        if closest_delay != float('inf'):
            delays.append(closest_delay)
    
    if delays:
        ed = sum(delays) / len(delays)  # Average of all delays
    else:
        ed = 0.0  # No valid delays, set ED to 0.0 or another appropriate value
    
    return far, ed

def evaluate_results(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return cm, accuracy, precision, recall, f1

def adjust_length(data_list, detection_list, label_list):
    data_len = len(data_list)
    detection_len = len(detection_list)
    label_len = len(label_list)

    # Detection と Label の長さが一致しているか確認
    if detection_len != label_len:
        raise ValueError("Detection and Label lists must have the same length")

    # Data 列が短い場合、Detection と Label を Data の長さに合わせる
    if data_len < detection_len:
        detection_list = detection_list[:data_len]
        label_list = label_list[:data_len]
    # Data 列が長い場合、Detection と Label にゼロを埋めて Data の長さに合わせる
    elif data_len > detection_len:
        detection_list.extend([0] * (data_len - detection_len))
        label_list.extend([0] * (data_len - detection_len))

    return detection_list, label_list
