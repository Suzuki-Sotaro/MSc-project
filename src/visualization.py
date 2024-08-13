# 以下はvisualization.pyのコードです。
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

class Visualization:
    def __init__(self, results):
        """
        Initialize the Visualization class.
        
        Parameters:
        - results: The results dictionary containing classification metrics, detection delays, and false alarm rates.
        """
        self.results = results

    def plot_detection_performance(self):
        """
        Plot detection performance metrics such as precision, recall, and F1-score.
        """
        ks, alphas, hs = [], [], []
        precisions, recalls, f1_scores = [], [], []

        for (k, alpha, h), metrics in self.results.items():
            ks.append(k)
            alphas.append(alpha)
            hs.append(h)
            precisions.append(metrics['classification_metrics']['precision'])
            recalls.append(metrics['classification_metrics']['recall'])
            f1_scores.append(metrics['classification_metrics']['f1_score'])

        # Convert lists to numpy arrays for easier plotting
        ks = np.array(ks)
        alphas = np.array(alphas)
        hs = np.array(hs)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)

        # Create a scatter plot for Precision, Recall, and F1-Score
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        scatter1 = ax[0].scatter(ks, alphas, c=precisions, cmap='viridis', s=100)
        ax[0].set_title('Precision')
        ax[0].set_xlabel('k')
        ax[0].set_ylabel('alpha')
        fig.colorbar(scatter1, ax=ax[0], label='Precision')

        scatter2 = ax[1].scatter(ks, alphas, c=recalls, cmap='plasma', s=100)
        ax[1].set_title('Recall')
        ax[1].set_xlabel('k')
        ax[1].set_ylabel('alpha')
        fig.colorbar(scatter2, ax=ax[1], label='Recall')

        scatter3 = ax[2].scatter(ks, alphas, c=f1_scores, cmap='coolwarm', s=100)
        ax[2].set_title('F1-Score')
        ax[2].set_xlabel('k')
        ax[2].set_ylabel('alpha')
        fig.colorbar(scatter3, ax=ax[2], label='F1-Score')

        plt.show()

    def plot_roc_curve(self):
        """
        Plot ROC curves for the various configurations.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for (k, alpha, h), metrics in self.results.items():
            true_labels = metrics['true_labels']
            predicted_scores = metrics['predicted_scores']

            # Since predicted_scores are binary, we can use them directly
            fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, lw=2, label=f'k={k}, alpha={alpha}, h={h} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc='lower right')

        plt.show()


    def plot_parameter_sensitivity(self):
        """
        Plot the sensitivity of the detection performance to different parameters.
        """
        ks, alphas, hs = [], [], []
        false_alarm_rates, detection_delays = [], []

        for (k, alpha, h), metrics in self.results.items():
            ks.append(k)
            alphas.append(alpha)
            hs.append(h)
            false_alarm_rates.append(metrics['false_alarm_rate'])
            detection_delays.append(metrics['detection_delay'])

        # Convert lists to numpy arrays for easier plotting
        ks = np.array(ks)
        alphas = np.array(alphas)
        hs = np.array(hs)
        false_alarm_rates = np.array(false_alarm_rates)
        detection_delays = np.array(detection_delays)

        # Create a scatter plot for False Alarm Rate and Detection Delay
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        scatter1 = ax[0].scatter(ks, alphas, c=false_alarm_rates, cmap='inferno', s=100)
        ax[0].set_title('False Alarm Rate')
        ax[0].set_xlabel('k')
        ax[0].set_ylabel('alpha')
        fig.colorbar(scatter1, ax=ax[0], label='False Alarm Rate')

        scatter2 = ax[1].scatter(ks, alphas, c=detection_delays, cmap='cividis', s=100)
        ax[1].set_title('Detection Delay')
        ax[1].set_xlabel('k')
        ax[1].set_ylabel('alpha')
        fig.colorbar(scatter2, ax=ax[1], label='Detection Delay')

        plt.show()

# Example usage:
if __name__ == "__main__":
    # Assume results is a dictionary containing the experiment results
    results = {
        (5, 0.01, 10): {
            "classification_metrics": {"precision": 0.75, "recall": 0.85, "f1_score": 0.80, "accuracy": 0.90},
            "detection_delay": 5,
            "false_alarm_rate": 0.05,
            "true_labels": np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
            "predicted_scores": np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.2, 0.1, 0.4, 0.3, 0.2, 0.7])
        },
        (10, 0.05, 20): {
            "classification_metrics": {"precision": 0.85, "recall": 0.80, "f1_score": 0.82, "accuracy": 0.88},
            "detection_delay": 7,
            "false_alarm_rate": 0.03,
            "true_labels": np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
            "predicted_scores": np.array([0.2, 0.3, 0.4, 0.9, 0.95, 0.3, 0.2, 0.5, 0.4, 0.3, 0.8])
        }
    }

    # Initialize the Visualization class
    visualizer = Visualization(results)

    # Plot detection performance
    visualizer.plot_detection_performance()

    # Plot ROC curves
    visualizer.plot_roc_curve()

    # Plot parameter sensitivity
    visualizer.plot_parameter_sensitivity()
