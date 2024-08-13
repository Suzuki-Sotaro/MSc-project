# 以下はvisualization.pyのコードです。
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

class Visualizer:
    def __init__(self, results_file='experiment_results.json'):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        self.df = pd.DataFrame(self.results)

    def plot_performance_comparison(self, metric='F1 Score'):
        plt.figure(figsize=(12, 6))
        data = self.df[self.df[metric].notna()]  # Exclude rows with NaN values
        if len(data) > 0:
            sns.boxplot(x='method', y=metric, data=data)
            plt.title(f'{metric} Comparison Across Methods')
            plt.ylabel(metric)
            plt.xlabel('Method')
        else:
            plt.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
        plt.savefig(f'{metric.lower().replace(" ", "_")}_comparison.png')
        plt.close()

    def plot_parameter_sensitivity(self, method, parameter, metric='F1 Score'):
        method_df = self.df[self.df['method'] == method]
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=parameter, y=metric, data=method_df)
        plt.title(f'{metric} Sensitivity to {parameter} for {method}')
        plt.ylabel(metric)
        plt.xlabel(parameter)
        plt.savefig(f'{method.lower().replace(" ", "_")}_{parameter}_sensitivity.png')
        plt.close()

    def plot_heatmap(self, method, x_param, y_param, metric='F1 Score'):
        method_df = self.df[self.df['method'] == method]
        pivot_table = method_df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title(f'{metric} Heatmap for {method}')
        plt.ylabel(y_param)
        plt.xlabel(x_param)
        plt.savefig(f'{method.lower().replace(" ", "_")}_{metric.lower().replace(" ", "_")}_heatmap.png')
        plt.close()

    def plot_bus_performance(self, metric='F1 Score'):
        centralized_df = self.df[self.df['method'] == 'Centralized']
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='bus', y=metric, data=centralized_df)
        plt.title(f'{metric} Comparison Across Buses')
        plt.ylabel(metric)
        plt.xlabel('Bus')
        plt.xticks(rotation=45)
        plt.savefig(f'bus_{metric.lower().replace(" ", "_")}_comparison.png')
        plt.close()

    def plot_roc_curve(self):
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            fpr = method_df['False Alarm Rate'].mean()
            tpr = method_df['Detection Rate'].mean()
            roc_auc = auc([0, fpr, 1], [0, tpr, 1])
            
            plt.plot([0, fpr, 1], [0, tpr, 1], label=f'{method} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()

    def generate_all_plots(self):
        metrics = ['F1 Score', 'Average Detection Delay', 'False Alarm Rate', 'Detection Rate']
        methods = ['Centralized', 'Method A', 'Method B']
        parameters = ['window_size', 'k', 'alpha', 'h']

        for metric in metrics:
            self.plot_performance_comparison(metric)

        for method in methods:
            for parameter in parameters:
                self.plot_parameter_sensitivity(method, parameter)

        self.plot_heatmap('Method A', 'p', 'h')
        self.plot_heatmap('Method B', 'aggregation_method', 'h')

        self.plot_bus_performance()
        self.plot_roc_curve()

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.generate_all_plots()
    print("All plots have been generated and saved.")