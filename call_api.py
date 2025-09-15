import requests
import json
import os

# ----------------------
# URL of your running Gradio app
# ----------------------
url = "http://127.0.0.1:7860/api/predict/"

# ----------------------
# Ask user for audio file path dynamically
# ----------------------
audio_file_path = input("Enter path to audio file: ").strip()

if not os.path.isfile(audio_file_path):
    print("File not found. Please check the path.")
    exit()

# ----------------------
# Prepare file for POST
# ----------------------
files = {"data": open(audio_file_path, "rb")}

try:
    response = requests.post(url, files=files)
    response.raise_for_status()  # Raise error if request failed

    # Gradio Blocks returns JSON: {"data": [output]}
    result = response.json().get("data", [])[0]

    # Pretty-print the result
    print("\nTranscription Result:")
    print(json.dumps(result, indent=4))

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"Error parsing response: {e}")
