# 以下はperformance_metrics.pyのコードです。
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

class PerformanceMetrics:
    def __init__(self, true_labels, predicted_labels, detection_times=None):
        """
        Initialize the PerformanceMetrics class.

        Parameters:
        - true_labels: Ground truth labels (1 for change, 0 for no change).
        - predicted_labels: Predicted labels from the change detection algorithm.
        - detection_times: (Optional) List of detection times for calculating delay metrics.
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.detection_times = detection_times

    def calculate_classification_metrics(self):
        """
        Calculate classification metrics: Precision, Recall, F1-Score, and Accuracy.

        Returns:
        - metrics: A dictionary containing precision, recall, F1-score, and accuracy.
        """
        precision = precision_score(self.true_labels, self.predicted_labels)
        recall = recall_score(self.true_labels, self.predicted_labels)
        f1 = f1_score(self.true_labels, self.predicted_labels)
        accuracy = accuracy_score(self.true_labels, self.predicted_labels)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }

        return metrics

    def calculate_detection_delay(self):
        """
        Calculate the detection delay based on the detection times.

        Returns:
        - detection_delay: The average delay from the actual change points to the detected change points.
        """
        if self.detection_times is None:
            raise ValueError("Detection times are not provided.")

        true_change_times = np.where(self.true_labels == 1)[0]
        delays = []

        for true_time in true_change_times:
            detected_time = next((t for t in self.detection_times if t >= true_time), None)
            if detected_time is not None:
                delays.append(detected_time - true_time)

        detection_delay = np.mean(delays) if delays else None
        return detection_delay

    def calculate_false_alarm_rate(self):
        """
        Calculate the false alarm rate.

        Returns:
        - false_alarm_rate: The rate of false positives (incorrect change detections).
        """
        false_positives = np.sum((self.predicted_labels == 1) & (self.true_labels == 0))
        total_negatives = np.sum(self.true_labels == 0)
        false_alarm_rate = false_positives / total_negatives if total_negatives > 0 else 0.0
        return false_alarm_rate

# Example usage:
if __name__ == "__main__":
    # Example true and predicted labels (these should be replaced with real data in practice)
    true_labels = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    predicted_labels = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    detection_times = [2, 3, 4, 8, 10]  # Example detection times

    # Initialize PerformanceMetrics
    metrics_calculator = PerformanceMetrics(true_labels, predicted_labels, detection_times)

    # Calculate classification metrics
    classification_metrics = metrics_calculator.calculate_classification_metrics()
    print("Classification Metrics:")
    print(classification_metrics)

    # Calculate detection delay
    detection_delay = metrics_calculator.calculate_detection_delay()
    print("\nDetection Delay:")
    print(detection_delay)

    # Calculate false alarm rate
    false_alarm_rate = metrics_calculator.calculate_false_alarm_rate()
    print("\nFalse Alarm Rate:")
    print(false_alarm_rate)
