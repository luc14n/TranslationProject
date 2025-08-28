if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    from translate import translatable
    from compare import compareable
    from itertools import permutations
    from analysis import analytics
    import matplotlib.pyplot as plt
    from concurrent.futures import ThreadPoolExecutor

    # Create a new translatable object with the input file
    translator = translatable('originals/poem_english.txt')

    # Define the intermediate languages for permutation
    intermediate_languages = ["arabic", "french", "mandarin", "spanish", "russian"]

    # Generate all sub-permutations of the intermediate languages
    #for length in range(1, len(intermediate_languages) + 1):
    #    for perm in permutations(intermediate_languages, length):
    #        # Create the full language sequence starting and ending with English
    #        languages = ["english"] + list(perm) + ["english"]
    #        print(languages)
    #
    #        # Use ThreadPoolExecutor to thread the source loop
    #        with ThreadPoolExecutor() as executor:
    #            futures = []
    #            for source in [0, 1, 2, 3, 4]:  # 0 for OpenAI 3.5, 1 for OpenAI 4, etc.
    #                futures.append(executor.submit(translator.startTranslation, languages, source))
    #
    #            # Wait for all threads to complete
    #            for future in futures:
    #                future.result()

    # Translations are finished
    #comparator = compareable('originals', 'translations', 'Comparison-Data/comparison.csv')
    #comparator.compare()
    #comparator.selfComparison()


    # Translations are finished
    analysis = analytics()



    # Process `comparison.csv`
    x_axis_variants = ['Translation Count', 'Last Language']
    unique_models = list(analysis.comparison_data['Model'].unique())
    unique_models.append(None)

    for model in unique_models:
        for x_axis in x_axis_variants:
            # Generate the plot for the current model and x-axis
            analysis.plot_filtered_boxplot(
                model=model,
                translation_count=None,
                final_language=None,
                last_language=None,
                x_axis=x_axis,
                metrics=['BLEU', 'METEOR', 'ROUGE'],
                output_path='graphs/comparison'
            )

    # Process `comparison_self_comparison.csv`
    x_axis_variants_self = ['Translation Count1', 'Last Language1', 'Model2', 'Translation Count2', 'Last Language2', 'Final Language']
    unique_models_self = list(analysis.self_comparison_data['Model1'].unique())
    unique_models_self.append(None)

    for model1 in unique_models_self:
        for x_axis in x_axis_variants_self:
            # Generate the plot for the current model and x-axis
            analysis.plot_filtered_boxplot_2(
                model1=model1,
                translation_count1=None,
                last_language1=None,
                model2=None,
                translation_count2=None,
                last_language2=None,
                final_language=None,
                x_axis=x_axis,
                metrics=['BLEU', 'METEOR', 'ROUGE'],
                output_path='graphs/self_comparison'
            )

    # Generating Boxpolts to compare translations of the same model across different languages
    for model1 in unique_models_self:
        for lang in analysis.self_comparison_data['Last Language1'].unique():
            # Generate the plot for the current model and x-axis
            analysis.plot_filtered_boxplot_2(
                model1=model1,
                translation_count1=None,
                last_language1=lang,
                model2=None,
                translation_count2=None,
                last_language2=None,
                final_language=None,
                x_axis='Final Language',
                metrics=['BLEU', 'METEOR', 'ROUGE'],
                output_path='graphs/lang_self_comparison'
            )