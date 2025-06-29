import spacy
from spacy.training.example import Example
from data.intent_data import TRAIN_DATA_INTENT
import random

def train_textcat():
    nlp = spacy.blank("en")

    # Add textcat pipe with default config (multilabel by default)
    textcat = nlp.add_pipe("textcat_multilabel")  # Use multilabel classifier

    # Add labels to textcat
    labels = list(next(iter(TRAIN_DATA_INTENT))[1]["cats"].keys())
    for label in labels:
        textcat.add_label(label)

    optimizer = nlp.begin_training()

    for i in range(20):
        random.shuffle(TRAIN_DATA_INTENT)
        losses = {}
        for text, annotations in TRAIN_DATA_INTENT:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
        print(f"Iteration {i + 1}, Losses: {losses}")

    nlp.to_disk("model/intent_model")
    print("Training complete and model saved to 'model/intent_model'")

if __name__ == "__main__":
    train_textcat()
