import json
import os

# Folder containing the JSON files
folder_path = './root_data'
REPUBLICAN_COLOR = "#DD2929"
REPUBLICAN_STRING = "R"

# Dictionary to store the data from all files
all_data = {}
questions = {}

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        # Open and load each JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if "stateExitPollData" in data:
                polls = data["stateExitPollData"]["polls"]

                for poll in polls:
                    question = poll["question"]
                    questions[question] = poll
            #     # Store data with filename as key (without the extension)
            #     all_data[filename] = data

print(questions.keys())
