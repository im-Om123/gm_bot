# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from scripts.data_preparation import train_dataset, word2idx
# from scripts.model import IntentModel
# import os
# os.makedirs("models", exist_ok=True)
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# # Hyperparameters
# embed_dim = 64
# hidden_dim = 32
# output_dim = len(torch.le.classes_)
# batch_size = 16
# epochs = 200
# learning_rate = 0.001

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model = IntentModel(len(word2idx), embed_dim, hidden_dim, output_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

# os.makedirs("models", exist_ok=True)

# for epoch in range(epochs):
#     total_loss = 0
#     model.train()
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     if (epoch + 1) % 10 == 0 or epoch == 0:
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# torch.save(model.state_dict(), "models/intent_model.pth")
# print("✅ Model trained and saved to models/intent_model.pth")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# --- Force PyTorch to use CPU by hiding CUDA devices ---
# This must be set BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = ''
# ---------------------------------------------------

# Append parent directory to sys.path to allow imports from 'scripts'
# This assumes train.py is in 'your_project_root/scripts/'
# If train.py is in 'your_project_root/', remove this line or adjust path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components from data_preparation and model
from scripts.data_preparation import train_dataset, word2idx, le, max_len, load_and_prepare_data, all_intents_data
from scripts.model import IntentModel

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Define hyperparameters
# These should match what's used in the model definition and saved for inference
embed_dim = 128
hidden_dim = 64
# Correctly get output_dim from the LabelEncoder's classes
output_dim = len(le.classes_)
batch_size = 32
epochs = 500 # Adjust as needed
learning_rate = 0.001

# Set device to CPU
device = torch.device("cpu")
print(f"Using device for training: {device}")

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialize model, optimizer, and criterion
model = IntentModel(len(word2idx), embed_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("--- Starting Training ---")
for epoch in range(epochs):
    model.train() # Set model to training mode
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad() # Clear gradients
        outputs = model(X_batch) # Forward pass
        loss = criterion(outputs, y_batch) # Calculate loss
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 50 == 0 or epoch == 1: # Print more frequently at start, then every 50 epochs
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the trained model and all necessary data for inference
model_data = {
    "model_state": model.state_dict(),
    "word2idx": word2idx,
    "le_classes": le.classes_.tolist(), # Save classes as list
    "max_len": max_len,
    "embed_dim": embed_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
}

model_save_path = "models/intent_model.pth"
torch.save(model_data, model_save_path)
print(f"✅ Model trained and saved to {model_save_path}")
print("--- Training Complete ---")

