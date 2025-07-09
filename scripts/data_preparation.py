# import json
# import torch
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset
# from nltk.tokenize import TreebankWordTokenizer
# tokenizer = TreebankWordTokenizer()

# from collections import Counter
# import nltk

# nltk.download('punkt')

# # === Load dataset ===
# with open("data/dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # === Extract sentences and labels ===
# sentences = []
# labels = []

# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         sentences.append(pattern.lower())
#         labels.append(intent["tag"])

# # === Tokenize and build vocabulary ===
# all_words = []
# for sentence in sentences:
#     tokens = tokenizer.tokenize(sentence)

#     all_words.extend(tokens)

# vocab = sorted(set(all_words))
# word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 = padding

# # === Encode sentences ===
# def encode_sentence(sentence):
#     return [word2idx.get(token, 0) for token in tokenizer.tokenize(sentence.lower())]

# X = [encode_sentence(s) for s in sentences]
# max_len = max(len(x) for x in X)

# def pad_sequence(seq, max_len):
#     return seq + [0] * (max_len - len(seq))

# X = [pad_sequence(seq, max_len) for seq in X]

# # === Encode labels ===
# le = LabelEncoder()
# y = le.fit_transform(labels)

# # === PyTorch Dataset ===
# class IntentDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.long)
#         self.y = torch.tensor(y, dtype=torch.long)

#     def __getitem__(self, index):
#         return self.X[index], self.y[index]

#     def __len__(self):
#         return len(self.X)

# # === Split dataset ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train_dataset = IntentDataset(X_train, y_train)
# test_dataset = IntentDataset(X_test, y_test)

# # Exportable components
# __all__ = [
#     "train_dataset", "test_dataset", "word2idx", "le", "max_len", "encode_sentence", "pad_sequence"
# ]


import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
import os # Import os for directory creation

# --- NLTK Downloads (Run once if you don't have them) ---
# Ensure NLTK data is downloaded. Catch LookupError for missing resources.
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading NLTK data (punkt, wordnet, omw-1.4)... This may take a moment.")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK data downloaded.")
except Exception as e:
    print(f"An unexpected error occurred during NLTK data check: {e}")

tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer() # Initialize the stemmer

def stem(word):
    """
    Stem a word to its root form.
    """
    return stemmer.stem(word.lower())

# === Global variables for data preparation ===
# These will be populated after loading and processing the dataset
word2idx = {}
le = LabelEncoder()
max_len = 0
train_dataset = None
test_dataset = None
all_intents_data = None # To store the raw intents data for responses

def encode_sentence(sentence):
    """Converts a sentence to a sequence of word indices using the global word2idx."""
    # Tokenize and stem the sentence, then map to indices
    return [word2idx.get(stem(token), 0) for token in tokenizer.tokenize(sentence.lower())]

def pad_sequence(seq, target_max_len):
    """Pads a sequence with zeros up to target_max_len."""
    return seq + [0] * (target_max_len - len(seq))

# === PyTorch Dataset Class ===
class IntentDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

# === Function to load and prepare data ===
def load_and_prepare_data(dataset_path="data/dataset.json"):
    global word2idx, le, max_len, train_dataset, test_dataset, all_intents_data

    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    # --- Load dataset with robust error handling ---
    # Now strictly loads from the file. If the file is empty or invalid, it will raise an error.
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            if not file_content.strip(): # Check if file is empty
                raise json.JSONDecodeError(f"File '{dataset_path}' is empty.", file_content, 0)
            data = json.loads(file_content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Error: '{dataset_path}' contains invalid JSON or is empty. "
            f"Please ensure your dataset.json is correctly formatted. Original error: {e}",
            e.doc, e.pos
        ) from e
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file '{dataset_path}' not found. "
            "Please ensure you have a valid dataset.json file in the 'data' directory."
        )

    all_intents_data = data # Store the full intents data for response retrieval

    sentences = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern.lower())
            labels.append(intent["tag"])

    # === Tokenize and build vocabulary ===
    all_words_list = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        stemmed_tokens = [stem(token) for token in tokens] # Apply stemming
        all_words_list.extend(stemmed_tokens)

    vocab = sorted(list(set(all_words_list)))
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 = padding
    print(f"Vocabulary size: {len(word2idx)}")

    # === Encode sentences ===
    X = [encode_sentence(s) for s in sentences]
    max_len = max(len(x) for x in X) if X else 1 # Ensure max_len is at least 1
    print(f"Max sentence length: {max_len}")

    X = [pad_sequence(seq, max_len) for seq in X]

    # === Encode labels ===
    le.fit(labels) # Fit LabelEncoder on all unique tags
    y = le.transform(labels)
    print(f"Number of unique intents (classes): {len(le.classes_)}")

    # === Split dataset ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

    train_dataset = IntentDataset(X_train, y_train)
    test_dataset = IntentDataset(X_test, y_test)

    print("Data preparation complete.")

# Call the data preparation function when this module is imported
load_and_prepare_data()
