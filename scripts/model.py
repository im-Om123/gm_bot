import torch
import torch.nn as nn

class IntentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, padding_idx=0):
        super(IntentModel, self).__init__()
        # Embedding layer: converts word indices to dense vectors
        # vocab_size + 1 because 0 is used for padding
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=padding_idx)
        
        # GRU layer: processes sequential data, captures context
        # batch_first=True means input tensor will be (batch_size, sequence_length, features)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        # Apply embedding layer
        x = self.embedding(x)
        # x shape: (batch_size, sequence_length, embed_dim)

        # Pass through GRU. output contains hidden states for each time step,
        # hidden is the final hidden state (or last hidden state for each layer)
        # We only need the final hidden state for classification
        _, hidden = self.gru(x)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # For a single-layer, unidirectional GRU, it's (1, batch_size, hidden_dim)
        
        # Squeeze the first dimension (num_layers * num_directions)
        x = hidden.squeeze(0)
        # x shape: (batch_size, hidden_dim)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x shape: (batch_size, output_dim) - raw scores for each class
        return x

