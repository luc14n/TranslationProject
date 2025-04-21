import os
import csv
import itertools
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

class compareable:
    def __init__(self, originals_path, translations_path, output_csv):
        # Store the path to the originals directory
        self.originals_path = originals_path

        # Store the file path to the translations
        self.translations_path = translations_path

        # Path to the output CSV file
        self.output_csv = output_csv

    def _load_file(self, file_path):
        # Helper method to load text from a file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def _get_original_poem(self, final_language):
        # Dynamically select the appropriate original poem based on the final language
        original_file = os.path.join(self.originals_path, f"poem_{final_language}.txt")
        if not os.path.exists(original_file):
            raise FileNotFoundError(f"Original poem for language '{final_language}' not found: {original_file}")
        return self._load_file(original_file)

    def compare(self):
        print("Comparing translations...")
        # Prepare the CSV file
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header row
            writer.writerow(["Filename", "Model", "Translation Count", "Final Language", "BLEU", "METEOR", "ROUGE"])

            # Iterate through all translation files in the directory
            for filename in os.listdir(self.translations_path):
                if filename.endswith('.txt'):
                    # Parse metadata from the filename
                    model, translation_count, final_language = self._parse_metadata(filename)
                    print([filename, model, translation_count, final_language])

                    # Load the appropriate original poem
                    original_poem = self._get_original_poem(final_language)

                    # Load the translation
                    translation_path = os.path.join(self.translations_path, filename)
                    translation = self._load_file(translation_path)

                    # Calculate metrics
                    bleu = self.BLEU(original_poem, translation)
                    meteor = self.METEOR(original_poem, translation)
                    rouge = self.ROUGE(original_poem, translation)

                    # Write the results to the CSV file
                    writer.writerow([filename, model, translation_count, final_language, f"{bleu:.4f}", f"{meteor:.4f}", f"{rouge:.4f}"])
                    print([filename, model, translation_count, final_language, f"{bleu:.4f}", f"{meteor:.4f}", f"{rouge:.4f}"])

    def _parse_metadata(self, filename):
        # Remove the file extension while preserving the last word
        base_name, _ = os.path.splitext(filename)

        # Split the base name on underscores
        parts = base_name.split("_")

        # Extract the model name (first part)
        model = parts[0]

        # Extract the languages (all parts after the model) and remove "to"
        languages = [part for part in parts[1:] if part != "to"]

        # Count the number of translations
        translation_count = len(languages) - 1  # Subtract 1 for the final language

        # Extract the final language (last part of the languages list)
        final_language = languages[-1]

        return model, translation_count, final_language

    def BLEU(self, original_poem, translation):
        # Calculate BLEU score with smoothing
        reference = [original_poem.split()]  # Tokenized reference
        candidate = translation.split()  # Tokenized candidate
        smoothie = SmoothingFunction().method1  # Use smoothing method1
        return sentence_bleu(reference, candidate, smoothing_function=smoothie)

    def METEOR(self, original_poem, translation):
        # Tokenize the original poem and translation
        original_tokens = original_poem.split()
        translation_tokens = translation.split()

        # Calculate METEOR score
        return meteor_score([original_tokens], translation_tokens)

    def ROUGE(self, original_poem, translation):
        # Calculate ROUGE-L score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(original_poem, translation)
        return scores['rougeL'].fmeasure

    def selfComparison(self):
        print("Performing self-comparison...")
        # Prepare the CSV file for self-comparison results
        output_csv = self.output_csv.replace(".csv", "_self_comparison.csv")
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header row
            writer.writerow(["Filename1", "Model1", "Translation Count1",
                             "Filename2", "Model2", "Translation Count2",
                             "Final Language", "BLEU", "METEOR", "ROUGE"])

            # Group files by final language
            translations_by_language = {}
            for filename in os.listdir(self.translations_path):
                if filename.endswith('.txt'):
                    _, _, final_language = self._parse_metadata(filename)
                    translations_by_language.setdefault(final_language, []).append(filename)

            # Compare each pair of translations within the same final language
            for final_language, files in translations_by_language.items():
                for file1, file2 in itertools.combinations(files, 2):
                    # Parse metadata for both files
                    model1, count1, _ = self._parse_metadata(file1)
                    model2, count2, _ = self._parse_metadata(file2)

                    # Load the translations
                    translation1 = self._load_file(os.path.join(self.translations_path, file1))
                    translation2 = self._load_file(os.path.join(self.translations_path, file2))

                    # Calculate metrics
                    bleu = self.BLEU(translation1, translation2)
                    meteor = self.METEOR(translation1, translation2)
                    rouge = self.ROUGE(translation1, translation2)

                    # Write the results to the CSV file
                    writer.writerow([file1, model1, count1, file2, model2, count2,
                                     final_language, f"{bleu:.4f}", f"{meteor:.4f}", f"{rouge:.4f}"])
                    print([file1, model1, count1, file2, model2, count2,
                           final_language, f"{bleu:.4f}", f"{meteor:.4f}", f"{rouge:.4f}"])