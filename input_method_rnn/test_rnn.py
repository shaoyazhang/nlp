from torch import nn
import torch 

rnn = nn.RNN(input_size=3, hidden_size=4, batch_first=True, num_layers=2, bidirectional=True)

# input.shape: (batch_size, seq_length, input_size)
input = torch.rand(2, 4, 3)

# 前向传播
output, hn = rnn(input)

# output.shape: (batch_size, seq_length, num_directions * hidden_size)
# hn.shape: (num_layers * num_directions, batch_size, hidden_size)
print("Output shape:", output.shape)  
print("Hidden state shape:", hn.shape)