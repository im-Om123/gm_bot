import torch
import torch.nn as nn

class IntentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, padding_idx=0):
        super(IntentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=padding_idx)
        
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        x = hidden.squeeze(0)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

