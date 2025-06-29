import spacy
import random
from responses import responses  # import the responses dictionary

nlp = spacy.load("model/intent_model")

def predict_intent(text):
    doc = nlp(text)
    return doc.cats

def get_response(intent):
    if intent in responses:
        return random.choice(responses[intent])
    else:
        return "Sorry, I didn't understand that."

if __name__ == "__main__":
    print("Welcome! Ask me about house renting. Type 'exit' to quit.")
    while True:
        text = input("You: ")
        if text.lower() in ("exit", "quit"):
            break
        cats = predict_intent(text)
        intent = max(cats, key=cats.get)
        print("Bot:", get_response(intent))
