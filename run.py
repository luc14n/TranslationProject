import matplotlib
matplotlib.use('TkAgg')

from translate import translatable
from compare import compareable
from itertools import permutations
from analysis import analytics
import matplotlib.pyplot as plt

# Create a new translatable object with the input file
translator = translatable('originals/poem_english.txt')

# Define the intermediate languages for permutation
intermediate_languages = ["arabic", "french", "mandarin"]

# Generate all sub-permutations of the intermediate languages
# for length in range(1, len(intermediate_languages) + 1):
#    for perm in permutations(intermediate_languages, length):
#        # Create the full language sequence starting and ending with English
#        languages = ["english"] + list(perm) + ["english"]
#        print(languages)
#        # Start the translation process
#        for source in [0,1,2,3,4]:  # 0 for OpenAI 3.5, 1, OpenAI 4, 2 for DeepSeek , 3 for Claude, 4 for Gemini
#            # Start translation for the current permutation and source
#            translator.startTranslation(languages, source=source)

# Translations are finished
comparator = compareable('originals', 'translations', 'comparison.csv')
# comparator.compare()
# comparator.selfComparison()


# Translations are finished
analysis = analytics()
comparison_vars = ['Model', 'Final Language', 'Translation Count']
self_comparison_vars = ['Model1', 'Translation Count1', 'Model2', 'Translation Count2', 'Final Language']

# Generate comparison plots
for var in comparison_vars:
    plt.figure(figsize=(12, 8))
    analysis.compareisonBoxPlots(var, plt.gca())  # Use the current axis
    plt.tight_layout()
    plt.show()

# Generate self-comparison plots
for var in self_comparison_vars:
    plt.figure(figsize=(12, 8))
    analysis.ComparisonSelfBoxPlots(var, plt.gca())  # Use the current axis
    plt.tight_layout()
    plt.show()

for model in ['gpt-3.5-turbo', 'gpt-4-turbo', 'deepseek-chat', 'claude-3-7-sonnet-20250219', 'gemini-2.0-flash']:
    plt.figure(figsize=(12, 8))
    analysis.ComparisonSelfModel1BoxPlots('Model2', plt.gca(), model)  # Use the current axis
    plt.tight_layout()
    plt.show()