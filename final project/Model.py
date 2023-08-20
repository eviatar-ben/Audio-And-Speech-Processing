import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np


class ResCNN(nn.Module):
    def __init__(self, n_cnn_layers, n_class, n_feats, stride=2, dropout=0.1):
        super(ResCNN, self).__init__()
        n_feats = n_feats // 2 + 1
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride,
                             padding=3 // 2)  # cnn for extracting heirachal features
        self.rescnn_layers = nn.Sequential(*[
            ResBlock(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, n_class)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        return x

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResBlock(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResBlock, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.relu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class ResCNNTransformer(nn.Module):
    def __init__(self, n_cnn_layers, n_class, n_feats, dropout=0.1):
        super(ResCNNTransformer, self).__init__()
        n_feats = n_feats // 2 + 1
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)  # cnn for extracting heirachal features
        self.rescnn_layers = nn.Sequential(*[
            ResBlock(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.transformer = nn.Transformer(nhead=4, num_encoder_layers=6, num_decoder_layers=6, dropout=dropout, d_model=n_feats * 32)
        self.fully_connected = nn.Linear(n_feats * 32, n_class)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.transformer(x, x)
        x = self.fully_connected(x)
        return x

class RNN(nn.Module):
    def __init__(self, n_cnn_layers, n_class, n_feats, dropout=0.1, hidden_size=128, num_layers=2):
        super(RNN, self).__init__()
        n_feats = n_feats // 2 + 1
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)
        self.rescnn_layers = nn.Sequential(*[
            ResBlock(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.rnn = nn.RNN(input_size=n_feats * 32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fully_connected = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x, _ = self.rnn(x)  # Use only the output, disregard hidden states
        x = self.fully_connected(x)
        return x

class DeepSpeechModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(DeepSpeechModel, self).__init__()
        n_feats = n_feats // 2 + 1
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResBlock(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


def init_model(hparams):
    """
    initialize a model based on the hyperparameters
    """
    if hparams['model_name'] == 'res_cnn':
        model = ResCNN(n_cnn_layers=hparams['n_cnn_layers'], n_class=hparams['n_class'],
                       n_feats=hparams['n_feats'], stride=hparams['stride'], dropout=hparams['dropout'])
    elif hparams['model_name'] == 'transformer':
        model = ResCNNTransformer(n_cnn_layers=hparams['n_cnn_layers'], n_class=hparams['n_class'],
                                  n_feats=hparams['n_feats'], dropout=hparams['dropout'])
    elif hparams['model_name'] == 'rnn':
        model = RNN(n_cnn_layers=hparams['n_cnn_layers'], n_class=hparams['n_class'],
                    n_feats=hparams['n_feats'], dropout=hparams['dropout'])
    elif hparams['model_name'] == 'deep_speech':
        model = DeepSpeechModel(n_cnn_layers=hparams['n_cnn_layers'], n_rnn_layers=hparams['n_rnn_layers'],
                                rnn_dim=hparams['rnn_dim'], n_class=hparams['n_class'],
                                n_feats=hparams['n_feats'], stride=hparams['stride'], dropout=hparams['dropout'])
    else:
        raise NotImplementedError
    return model
