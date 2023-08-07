import torch
import torch.nn as nn


class ASRModel(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_units=128):
        super(ASRModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, rnn_units, bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(2 * rnn_units, output_dim)  # 2 * rnn_units due to bidirectional LSTM

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.output_layer(rnn_output)
        return output


# Example usage:
input_dim = 40  # Dimensionality of the acoustic features (e.g., MFCCs)
output_dim = 30  # Number of characters or tokens in the vocabulary (including blank symbol)
asr_model = ASRModel(input_dim, output_dim)

# Print the model summary
print(asr_model)
