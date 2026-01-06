import torch.nn as nn
from config import *
class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=EMBEDDING_DIM, 
                          hidden_size=HIDDEN_SIZE, 
                          num_layers=1, 
                          batch_first=True)
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, 
                                out_features=vocab_size)
        
    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        output, _ = self.rnn(embed)  # [batch_size, seq_length, embedding_dim] -> [batch_size, seq_length, hidden_size]
        last_hidden_state = output[:, -1, :]  # 取最后一个时间步的输出 [batch_size, hidden_size]
        output = self.linear(last_hidden_state) # [batch_size, vocab_size]
        return output