# This the content of utils.py  
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_far_ed(true_labels, predicted_labels, detection_time):
    # Calculate False Alarm Rate (FAR)
    num_true_negatives = np.sum(true_labels == 0)
    if num_true_negatives == 0:
        far = 0.0  # No true negatives, FAR should be set to 0.0 or another appropriate value
    else:
        far = np.sum((predicted_labels == 1) & (true_labels == 0)) / num_true_negatives
    
    # Calculate Expected Delay (ED)
    if detection_time == -1:
        ed = np.nan  # No detection occurred
    else:
        ed = detection_time
    
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
