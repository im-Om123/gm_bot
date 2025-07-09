import torch
from scripts.model import IntentModel
from scripts.data_preparation import word2idx, le, max_len, encode_sentence, pad_sequence

# Model parameters
embed_dim = 64
hidden_dim = 32
output_dim = len(le.classes_)

# Load model
model = IntentModel(len(word2idx), 64, 32, len(le.classes_))
model.load_state_dict(torch.load("models/intent_model.pth"))
model.eval()

# Prediction function
def predict_intent(text):
    encoded = encode_sentence(text)
    padded = pad_sequence(encoded, max_len)
    input_tensor = torch.tensor([padded])
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_tag = le.inverse_transform([predicted_index])[0]
        return predicted_tag
