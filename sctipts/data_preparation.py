import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')

# === Load dataset ===
with open("data/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === Extract sentences and labels ===
sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern.lower())
        labels.append(intent["tag"])

# === Tokenize and build vocabulary ===
all_words = []
for sentence in sentences:
    tokens = word_tokenize(sentence)
    all_words.extend(tokens)

vocab = sorted(set(all_words))
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 = padding

# === Encode sentences ===
def encode_sentence(sentence):
    return [word2idx.get(token, 0) for token in word_tokenize(sentence.lower())]

X = [encode_sentence(s) for s in sentences]
max_len = max(len(x) for x in X)

def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

X = [pad_sequence(seq, max_len) for seq in X]

# === Encode labels ===
le = LabelEncoder()
y = le.fit_transform(labels)

# === PyTorch Dataset ===
class IntentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = IntentDataset(X_train, y_train)
test_dataset = IntentDataset(X_test, y_test)

# Exportable components
__all__ = [
    "train_dataset", "test_dataset", "word2idx", "le", "max_len", "encode_sentence", "pad_sequence"
]
