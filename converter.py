import json
import os

# Adjust the path to load dataset.json from the data folder
with open("data/dataset.json", encoding="utf-8") as f:
    data = json.load(f)

all_tags = [intent["tag"] for intent in data["intents"]]

train_intent = []

for intent in data["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        cats = {t: 0 for t in all_tags}
        cats[tag] = 1
        train_intent.append((pattern, {"cats": cats}))

# Ensure the data folder exists (it does, but good practice)
os.makedirs("data", exist_ok=True)

# Save converted data back inside data folder
with open("data/intent_data.py", "w", encoding="utf-8") as f:
    f.write("TRAIN_DATA_INTENT = [\n")
    for text, cats in train_intent:
        f.write(f"    ({repr(text)}, {repr(cats)}),\n")
    f.write("]\n")

print("Conversion complete! Training data saved to data/intent_data.py")
