import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class analytics:
    def __init__(self):
        # Load the data from the CSV files and store them as instance variables
        self.comparison_data = pd.read_csv('comparison.csv')
        self.self_comparison_data = pd.read_csv('comparison_self_comparison.csv')

    def plot_filtered_boxplot(self, model=None, translation_count=None, final_language=None, last_language=None, x_axis=None, metrics=None, output_path=None):
        """
        Plots a box plot for selected metrics with filtering options and a dynamic x-axis.

        Parameters:
            model (str): Filter by model (optional).
            translation_count (int): Filter by translation count (optional).
            final_language (str): Filter by final language (optional).
            last_language (str): Filter by last language (optional).
            x_axis (str): Column to use for the x-axis (default: 'Translation Count').
            metrics (list): List of metrics to include in the plot (default: ['BLEU', 'METEOR', 'ROUGE']).
            output_path (str): Path to save the plot (optional).
        """
        # Default metrics if none are provided
        if metrics is None:
            metrics = ['BLEU', 'METEOR', 'ROUGE']

        # Ensure metrics are sorted for consistent ordering
        metrics = sorted(metrics)

        # Define consistent colors for metrics
        metric_colors = {
            'BLEU': '#1f77b4',
            'METEOR': '#2ca02c',
            'ROUGE': '#ff7f0e'
        }

        metricString = "_"
        for metric in metrics:
            metricString = metricString + metric + '_'

        # Apply filters
        filtered_data = self.comparison_data
        filteredString = ''
        if model:
            filtered_data = filtered_data[filtered_data['Model'] == model]
            filteredString += f'Model: {model} |'
        if translation_count:
            filtered_data = filtered_data[filtered_data['Translation Count'] == translation_count]
            filteredString += f'Translation Count: {translation_count} |'
        if final_language:
            filtered_data = filtered_data[filtered_data['Final Language'] == final_language]
            filteredString += f'Final Language: {final_language} |'
        if last_language:
            filtered_data = filtered_data[filtered_data['Last Language'] == last_language]
            filteredString += f'Last Language: {last_language} |'

        if (filteredString != ''):
            filteredString = "Filtered by - " + filteredString

        # Melt the DataFrame to long format for the selected metrics
        melted_data = filtered_data.melt(
            id_vars=['Filename', 'Model', 'Translation Count', 'Final Language', 'Last Language'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )

        # Sort the data by the x_axis column
        melted_data = melted_data.sort_values(by=x_axis)

        # Create the box plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            x=x_axis, y='Score', hue='Metric', data=melted_data,
            hue_order=metrics,  # Ensure consistent metric order
            palette=metric_colors  # Ensure consistent colors
        )
        plt.title(f'Box Plot of Selected Metrics by {x_axis}')
        plt.xlabel(x_axis + ' ' + filteredString)
        plt.ylabel('Score')
        plt.legend(title='Metric')
        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path + '/' +
                        str(model) + '_' +
                        str(translation_count) + '_' +
                        str(final_language) + '_' +
                        str(last_language) + metricString +
                        f'boxplot_by_{x_axis}.png')
            plt.close()
        else:
            plt.show()

    def plot_filtered_boxplot_2(self, model1=None, translation_count1=None, last_language1=None,
                                model2=None, translation_count2=None, last_language2=None,
                                final_language=None, x_axis=None, metrics=None, output_path=None):
        """
        Plots a box plot for selected metrics with filtering options for self-comparison data.

        Parameters:
            model1 (str): Filter by Model1 (optional).
            translation_count1 (int): Filter by Translation Count1 (optional).
            last_language1 (str): Filter by Last Language1 (optional).
            model2 (str): Filter by Model2 (optional).
            translation_count2 (int): Filter by Translation Count2 (optional).
            last_language2 (str): Filter by Last Language2 (optional).
            final_language (str): Filter by Final Language (optional).
            x_axis (str): Column to use for the x-axis (default: None).
            metrics (list): List of metrics to include in the plot (default: ['BLEU', 'METEOR', 'ROUGE']).
            output_path (str): Path to save the plot (optional).
        """
        # Default metrics if none are provided
        if metrics is None:
            metrics = ['BLEU', 'METEOR', 'ROUGE']

        # Ensure metrics are sorted for consistent ordering
        metrics = sorted(metrics)

        # Define consistent colors for metrics
        metric_colors = {
            'BLEU': '#1f77b4',
            'METEOR': '#2ca02c',
            'ROUGE': '#ff7f0e'
        }

        metricString = "_"
        for metric in metrics:
            metricString = metricString + metric + '_'

        # Apply filters
        filtered_data = self.self_comparison_data
        filterString = ''
        if model1:
            filtered_data = filtered_data[filtered_data['Model1'] == model1]
            filterString += f'Model1: {model1} |'
        if translation_count1:
            filtered_data = filtered_data[filtered_data['Translation Count1'] == translation_count1]
            filterString += f'Translation Count1: {translation_count1} |'
        if last_language1:
            filtered_data = filtered_data[filtered_data['Last Language1'] == last_language1]
            filterString += f'Last Language1: {last_language1} |'
        if model2:
            filtered_data = filtered_data[filtered_data['Model2'] == model2]
            filterString += f'Model2: {model2} |'
        if translation_count2:
            filtered_data = filtered_data[filtered_data['Translation Count2'] == translation_count2]
            filterString += f'Translation Count2: {translation_count2} |'
        if last_language2:
            filtered_data = filtered_data[filtered_data['Last Language2'] == last_language2]
            filterString += f'Last Language2: {last_language2} |'
        if final_language:
            filtered_data = filtered_data[filtered_data['Final Language'] == final_language]
            filterString += f'Final Language: {final_language} |'

        if (filterString != ''):
            filterString = "Filtered by - " + filterString

        # Melt the DataFrame to long format for the selected metrics
        melted_data = filtered_data.melt(
            id_vars=['Model1', 'Translation Count1', 'Last Language1',
                     'Model2', 'Translation Count2', 'Last Language2', 'Final Language'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )

        # Debugging: Check if x_axis exists in the DataFrame
        if x_axis not in melted_data.columns:
            raise ValueError(f"Invalid x_axis value: {x_axis}. Available columns: {melted_data.columns}")

        # Sort the data by the x_axis column
        melted_data = melted_data.sort_values(by=x_axis)

        # Create the box plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            x=x_axis, y='Score', hue='Metric', data=melted_data,
            hue_order=metrics,  # Ensure consistent metric order
            palette=metric_colors  # Ensure consistent colors
        )
        plt.title(f'Translation Score by {x_axis}')
        plt.xlabel(x_axis + " " + filterString)
        plt.ylabel('Score')
        plt.legend(title='Metric')
        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path + '/' +
                        str(model1) + '_' + str(translation_count1) + '_' + str(last_language1) + '_' +
                        str(model2) + '_' + str(translation_count2) + '_' + str(last_language2) + '_' +
                        str(final_language) + metricString +
                        f'boxplot_by_{x_axis}.png')
            plt.close()
        else:
            plt.show()
