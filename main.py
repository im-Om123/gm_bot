import torch
import json
import os
import sys
import random
import numpy as np # Import numpy for le.classes_ conversion

# --- Force PyTorch to use CPU by hiding CUDA devices ---
# This must be set BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = ''
# ---------------------------------------------------

# Append parent directory to sys.path to allow imports from 'scripts'
# This assumes main.py is in 'your_project_root/scripts/'
# If main.py is in 'your_project_root/', remove this line or adjust path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components from data_preparation and model
from scripts.data_preparation import encode_sentence, pad_sequence, stem, tokenizer, le, all_intents_data
from scripts.model import IntentModel
from sklearn.preprocessing import LabelEncoder # Explicitly import LabelEncoder for loading

# Define paths
MODEL_FILE = "models/intent_model.pth"
DATASET_FILE = "data/dataset.json"

# Ensure models directory exists (though train.py should create it)
os.makedirs("models", exist_ok=True)

# --- Load Model and Data for Inference ---
model = None
word2idx = {}
max_len = 0
responses = {}
confidence_threshold = 0.75 # Adjust this value as needed
loaded_le = LabelEncoder() # Create a new LabelEncoder instance for loading

def load_chatbot_components():
    global model, word2idx, max_len, responses, loaded_le, all_intents_data

    print(f"Loading chatbot components from {MODEL_FILE} and {DATASET_FILE}...")
    try:
        # Load model data
        # Map location to CPU to avoid loading CUDA tensors on a non-CUDA machine
        model_data = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        word2idx = model_data["word2idx"]
        loaded_le.classes_ = np.array(model_data["le_classes"]) # Load classes back into LabelEncoder
        max_len = model_data["max_len"]
        embed_dim = model_data["embed_dim"]
        hidden_dim = model_data["hidden_dim"]
        output_dim = model_data["output_dim"]

        # Initialize model with loaded hyperparameters
        model = IntentModel(len(word2idx), embed_dim, hidden_dim, output_dim).to(torch.device('cpu'))
        model.load_state_dict(model_data["model_state"])
        model.eval() # Set model to evaluation mode

        # Load responses from the dataset file
        # Use the all_intents_data loaded by data_preparation.py
        if all_intents_data:
            responses = {intent['tag']: intent.get('responses', ["Sorry, I don't understand."])
                         for intent in all_intents_data["intents"]}
        else:
            # Fallback if all_intents_data somehow wasn't loaded (shouldn't happen with current setup)
            print("Warning: all_intents_data not available. Attempting to load responses directly from dataset.json.")
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
                responses = {intent['tag']: intent.get('responses', ["Sorry, I don't understand."])
                             for intent in temp_data["intents"]}

        print("Chatbot components loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILE}' or dataset file '{DATASET_FILE}' not found.")
        print("Please ensure you have trained the model first by running `python scripts/train.py`.")
        sys.exit(1) # Exit if essential files are missing
    except Exception as e:
        print(f"An error occurred while loading chatbot components: {e}")
        sys.exit(1)

def predict_intent(text):
    """Predicts the intent of the given text."""
    # Ensure word2idx and max_len are available
    if not word2idx or max_len == 0:
        print("Error: Chatbot components not fully loaded. Cannot predict intent.")
        sys.exit(1)

    # Tokenize and stem the input sentence
    tokens = tokenizer.tokenize(text.lower())
    stemmed_tokens = [stem(token) for token in tokens]

    # Convert tokens to indices using the loaded word2idx
    encoded_seq = [word2idx.get(token, 0) for token in stemmed_tokens]

    # Pad the sequence to max_len
    padded_seq = pad_sequence(encoded_seq, max_len)
    
    # Convert to PyTorch tensor and reshape for batching (batch_size=1)
    input_tensor = torch.tensor([padded_seq], dtype=torch.long).to(torch.device('cpu'))
    
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(input_tensor)
        
        # Get probabilities by applying softmax
        probabilities = torch.softmax(output, dim=1)
        
        # Get the predicted class index and its probability
        max_prob, predicted_index = torch.max(probabilities, dim=1)
        
        # Convert predicted index back to original tag string using the loaded_le
        intent_tag = loaded_le.inverse_transform([predicted_index.item()])[0]
        
        return intent_tag, max_prob.item()

# --- Main Chatbot Loop ---
def run_chatbot():
    print("PropertyBot: Hi! Ask me about house rent, location, or details. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("PropertyBot: Goodbye!")
            break
        
        intent, confidence = predict_intent(user_input)
        
        if confidence > confidence_threshold:
            # If confident, pick a random response for the predicted intent
            response_options = responses.get(intent, ["I didn't get that."])
            print("PropertyBot:", random.choice(response_options))
        else:
            # If not confident enough
            print("PropertyBot: I didn't get that. Could you please rephrase or ask something else?")

# --- Main Execution Block ---
if __name__ == "__main__":
    # First, ensure data is prepared (this will also load/create dataset.json)
    # The load_and_prepare_data function in data_preparation.py is called upon import,
    # so we just need to ensure the module is imported.
    # This also populates the global variables like word2idx, le, max_len, etc.

    # Then, load the trained model and other components for the chatbot
    load_chatbot_components()

    # Finally, run the interactive chatbot
    run_chatbot()