import torch
import torch.nn as nn

class IntentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(IntentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)              # [batch_size, seq_len, embed_dim]
        x = x.mean(dim=1)                  # [batch_size, embed_dim] → average over sequence
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
