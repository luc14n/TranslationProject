import os
import json

from openai import OpenAI
import anthropic
from google import genai
from google.genai import types


class translatable:
    def __init__(self, userIn):
        self.text = open(userIn).read()
        self.currentText = self.text
        self.translatedText = ""
        self.currentLang = ""
        self.targetLang = ""
        self.storage_path = os.path.join(os.getcwd(), "translations")

        # Load API keys from config.json
        with open("config.json") as config_file:
            config = json.load(config_file)

        # Initialize API keys
        self.apiKeys = {
            0: config["OPENAI_API_KEY"],
            1: config["OPENAI_API_KEY"],
            2: config["DEEPSEEK_API_KEY"],
            3: config["ANTHROPIC_API_KEY"],
            4: config["GOOGLE_API_KEY"]
        }

    def translate(self, source):
        # Define a mapping for source to model and API base
        api_config = {
            0: {
                "api_base": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "temperature": 0.2
            },
            1: {
                "api_base": "https://api.openai.com/v1",
                "model": "gpt-4-turbo",
                "temperature": 0.2
            },
            2: {
                "api_base": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "temperature": 1.3
            },
            3: {
                "api_base": "https://api.anthropic.com",
                "model": "claude-3-7-sonnet-20250219",
                "temperature": 0.2
            },
            4: {
                "api_base": "https://api.google.com",
                "model": "gemini-2.0-flash",
                "temperature": 0.75
            }
        }

        # Validate the source
        if source not in api_config:
            raise ValueError(f"Invalid source: {source}")

        # Get the configuration for the selected source
        config = api_config[source]

        if source in [0,1,2]:
            # Initialize the OpenAI client
            # openai.api_base = config["api_base"]
            client = OpenAI(api_key=self.apiKeys[source], base_url=config["api_base"])

            # Make the API call using the new interface
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": f"You are a translator from {self.currentLang} to {self.targetLang}. "
                                                  f"Only output desired translation,"
                                                  f"only use the text in the input, "
                                                  f"do not copy a translation from elsewhere,"
                                                  f"perform a fresh metaphorical translation."},
                    {"role": "user", "content": self.currentText},
                ],
                max_tokens=1000,
                temperature=config["temperature"],
                stream=False
            )

            # Extract the translated text
            self.currentText = response.choices[0].message.content.strip()

        elif source in [3]:
            # Initialize the Anthropic client
            client = anthropic.Client(api_key=self.apiKeys[source], base_url=config["api_base"])

            # Make the API call using the new interface
            response = client.messages.create(
                model=config["model"],
                messages=[
                    {"role": "assistant", "content": f"You are a translator from {self.currentLang} to {self.targetLang}. "
                                                  f"Only output desired translation,"
                                                  f"only use the text in the input, "
                                                  f"do not copy a translation from elsewhere,"
                                                  f"perform a fresh metaphorical translation."},
                    {"role": "user", "content": self.currentText},
                ],
                max_tokens=1000,
                temperature=config["temperature"],
                stream=False
            )

            self.currentText = response.content[0].text.strip()

        elif source in [4]:
            # Initialize the Google client
            client = genai.Client(api_key=self.apiKeys[source])

            # Make the API call using the new interface
            response = client.models.generate_content(
                model=config["model"],
                contents=[self.currentText],
                config=types.GenerateContentConfig(
                    system_instruction=f"You are a translator from {self.currentLang} to {self.targetLang}. "
                                       f"Only output desired translation,"
                                       f"only use the text in the input, "
                                       f"do not copy a translation from elsewhere,"
                                       f"perform a fresh metaphorical translation.",
                    max_output_tokens=1000,
                    temperature=config["temperature"],
                )
            )

            # Extract the translated text
            self.currentText = response.text.strip()

    def startTranslation(self, languages, source):
        # Define the model name based on the source
        api_config = {
            0: "gpt-3.5-turbo",
            1: "gpt-4-turbo",
            2: "deepseek-chat",
            3: "claude-3-7-sonnet-20250219",
            4: "gemini-2.0-flash"
        }
        if source not in api_config:
            raise ValueError(f"Invalid source: {source}")
        model_name = api_config[source]

        filename = f"{model_name}_{languages[0]}"  # Start the filename with the first language and model
        for i in range(len(languages) - 1):
            self.currentLang = languages[i]
            self.targetLang = languages[i + 1]

            # Append the current target language to the filename
            filename += f"_to_{self.targetLang}"

            # Change for Efficiency
            # Check to see if translation file already exists
            if os.path.exists(f"{self.storage_path}/{filename}.txt"):
                print(f"Translation file {filename}.txt already exists. Skipping translation.")
                with open(f"{self.storage_path}/{filename}.txt", "r", encoding="utf-8") as file:
                    self.currentText = file.read()
            else:
                # Perform the translation
                self.translate(source)

                # Save the translation to a file
                file_path = os.path.join(f"{self.storage_path}", f"{filename}.txt")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(self.currentText)

                print(f"Translated from {self.currentLang} to {self.targetLang} successfully. Saved to {filename}.txt.")