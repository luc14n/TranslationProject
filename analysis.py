import pandas as pd
import seaborn as sns

class analytics:
    def compareisonBoxPlots(self, var, ax):
        # Load the data from the comparison CSV
        data = pd.read_csv('comparison.csv')

        # Melt the data to make it suitable for Seaborn
        melted_data = data.melt(
            id_vars=['Filename', 'Model', 'Final Language', 'Last Language', 'Translation Count'],
            value_vars=['BLEU', 'METEOR', 'ROUGE'],
            var_name='Metric',
            value_name='Score'
        )

        # Create the boxplot in the provided subplot
        sns.boxplot(x=var, y='Score', hue='Metric', data=melted_data, palette='Set2', ax=ax)
        ax.set_title('Boxplot of BLEU, METEOR, and ROUGE Scores by ' + var)
        ax.set_xlabel(var)
        ax.set_ylabel('Score')

    def ComparisonSelfBoxPlots(self, var, ax):
        # Load the data from the self-comparison CSV
        data = pd.read_csv('comparison_self_comparison.csv')

        # Melt the data to make it suitable for Seaborn
        melted_data = data.melt(
            id_vars=['Filename1', 'Model1', 'Translation Count1', 'Last Language1',
                     'Filename2', 'Model2', 'Translation Count2', 'Last Language2', 'Final Language'],
            value_vars=['BLEU', 'METEOR', 'ROUGE'],
            var_name='Metric',
            value_name='Score'
        )

        # Create the boxplot in the provided subplot
        sns.boxplot(x=var, y='Score', hue='Metric', data=melted_data, palette='Set2', ax=ax)
        ax.set_title('Boxplot of BLEU, METEOR, and ROUGE Scores by ' + var)
        ax.set_xlabel(var)
        ax.set_ylabel('Score')

    def ComparisonSelfModel1BoxPlots(self, var, ax, div):
        # Load the data from the self-comparison CSV
        data = pd.read_csv('comparison_self_comparison.csv')

        subset = data[data['Model1'] == div]

        # Melt the data to make it suitable for Seaborn
        melted_data = subset.melt(
            id_vars=['Filename1', 'Model1', 'Translation Count1', 'Last Language1',
                     'Filename2', 'Model2', 'Translation Count2', 'Last Language2', 'Final Language'],
            value_vars=['BLEU', 'METEOR', 'ROUGE'],
            var_name='Metric',
            value_name='Score'
        )

        # Create the boxplot in the provided subplot
        sns.boxplot(x=var, y='Score', hue='Metric', data=melted_data, palette='Set2', ax=ax)
        ax.set_title('Boxplot of BLEU, METEOR, and ROUGE Scores by ' + var + ' for ' + div)
        ax.set_xlabel(var)
        ax.set_ylabel('Score')