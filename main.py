# main.py
import torch
import json
import os
import sys
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Force PyTorch to use CPU by hiding CUDA devices ---
# This ensures it runs on any machine, even without a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = ''
# ---------------------------------------------------

# Append parent directory to sys.path to allow imports from 'scripts'
# IMPORTANT: Adjust this path if your directory structure is different.
# If main.py is at the root of your project, you might not need this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components from your scripts
from scripts.data_preparation import encode_sentence, pad_sequence, stem, tokenizer, le, all_intents_data
from scripts.model import IntentModel

# Define paths
MODEL_FILE = "models/intent_model.pth"
DATASET_FILE = "data/dataset.json"
CONFIDENCE_THRESHOLD = 0.75 # Adjust this value as needed

# --- Global Variables for Chatbot Components (Loaded ONCE) ---
# These will be populated in the global scope when the script is first run by Flask.
model = None
word2idx = {}
max_len = 0
responses = {}
loaded_le = LabelEncoder()

# --- Load all chatbot components at the start of the application ---
def initialize_chatbot():
    """
    Loads the model, data, and other necessary components once.
    This function is called immediately when this script is imported.
    """
    global model, word2idx, max_len, responses, loaded_le

    print(f"Loading chatbot components from {MODEL_FILE} and {DATASET_FILE}...")
    try:
        # Load model data
        model_data = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        word2idx = model_data["word2idx"]
        loaded_le.classes_ = np.array(model_data["le_classes"])
        max_len = model_data["max_len"]
        embed_dim = model_data["embed_dim"]
        hidden_dim = model_data["hidden_dim"]
        output_dim = model_data["output_dim"]

        # Initialize model with loaded hyperparameters
        model = IntentModel(len(word2idx), embed_dim, hidden_dim, output_dim).to(torch.device('cpu'))
        model.load_state_dict(model_data["model_state"])
        model.eval() # Set model to evaluation mode

        # Load responses
        # The all_intents_data should be a dictionary with an 'intents' key
        if 'intents' in all_intents_data:
            responses = {intent['tag']: intent.get('responses', ["Sorry, I don't understand."])
                         for intent in all_intents_data["intents"]}
        else:
            print("Warning: 'intents' key not found in all_intents_data. Using fallback.")
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
                responses = {intent['tag']: intent.get('responses', ["Sorry, I don't understand."])
                             for intent in temp_data["intents"]}

        print("Chatbot components loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: Model file or dataset file not found: {e}")
        print("Please ensure you have trained the model first.")
        model = None # Set model to None to handle this gracefully
    except Exception as e:
        print(f"An error occurred while loading chatbot components: {e}")
        model = None # Set model to None to handle this gracefully

# Call the initialization function immediately when the module is imported
initialize_chatbot()

# --- Main Function for Flask App ---
def get_bot_response(user_input):
    """
    Predicts the intent of the user's message and returns a response.
    This function is designed to be called by the Flask application.
    """
    # Check if the model was loaded successfully.
    if model is None:
        return "Sorry, the bot is currently unavailable due to a server configuration error."

    try:
        # Tokenize and stem the input sentence
        tokens = tokenizer.tokenize(user_input.lower())
        stemmed_tokens = [stem(token) for token in tokens]

        # Convert tokens to indices using the loaded word2idx
        encoded_seq = [word2idx.get(token, 0) for token in stemmed_tokens]

        # Pad the sequence to max_len
        padded_seq = pad_sequence(encoded_seq, max_len)

        # Convert to PyTorch tensor and reshape for batching (batch_size=1)
        input_tensor = torch.tensor([padded_seq], dtype=torch.long).to(torch.device('cpu'))

        with torch.no_grad(): # Disable gradient calculation for inference
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            max_prob, predicted_index = torch.max(probabilities, dim=1)
            intent_tag = loaded_le.inverse_transform([predicted_index.item()])[0]

        if max_prob.item() > CONFIDENCE_THRESHOLD:
            # Get a random response for the predicted intent
            response_options = responses.get(intent_tag, ["I didn't get that."])
            return random.choice(response_options)
        else:
            # If not confident enough
            return "I didn't get that. Could you please rephrase or ask something else?"

    except Exception as e:
        # Catch any unexpected errors during the prediction process
        print(f"An error occurred during bot response inference: {e}")
        return "I'm sorry, I'm having trouble understanding that. Could you please rephrase?"

