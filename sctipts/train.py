import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from scripts.model import IntentModel
from scripts.data_preparation import train_dataset, word2idx, le

# Hyperparameters
embed_dim = 64
hidden_dim = 32
output_dim = len(le.classes_)
batch_size = 16
epochs = 200
learning_rate = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = IntentModel(len(word2idx), embed_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/intent_model.pth")
print("✅ Model trained and saved to models/intent_model.pth")
