# this class is an RNN model for the final project ASR for the an4 dataset

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_units=128):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_dim, rnn_units, bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(2 * rnn_units, output_dim)  # 2 * rnn_units due to bidirectional LSTM

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.output_layer(rnn_output)
        return output
