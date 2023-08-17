import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import wandb
from jiwer import wer

import Model
import preprocess
WB = True

class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, batch_iterator, criterion, optimizer, scheduler, epoch, iter_meter, batch_size):
    model.train()
    # with experiment.train():
    data_len = len(batch_iterator)
    for batch_idx, _data in enumerate(batch_iterator):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, torch.from_numpy(input_lengths), torch.from_numpy(label_lengths))
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        if batch_idx % 50 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / data_len, loss.item()))
            if WB:
                wandb.log({"train_loss": loss.item()})


def test(model, device, batch_iterator, criterion, epoch, iter_meter):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    text_transform = preprocess.TextTransform()

    with torch.no_grad():
        for batch_idx, _data in enumerate(batch_iterator):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, torch.from_numpy(input_lengths), torch.from_numpy(label_lengths))
            test_loss += loss.item() / len(batch_iterator)

            decoded_preds, decoded_targets = preprocess.GreedyDecoder(text_transform , output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
            if WB:
                wandb.log({"test_loss": loss.item()})

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    # experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    # experiment.log_metric('cer', avg_cer, step=iter_meter.get())
    # experiment.log_metric('wer', avg_wer, step=iter_meter.get())
    if WB:
                wandb.log({"wer":avg_wer})


    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def train_test_valid(hparams, batch_iterator):
    train_batch_iterator, test_batch_iterator, val_batch_iterator = batch_iterator[0], batch_iterator[1], batch_iterator[2], 
    epochs = hparams['epochs']
    batch_size = hparams['batch_size']

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    model = Model.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    # print(model)
    # print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=len(batch_iterator),  # todo: check if this is correct
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, train_batch_iterator, criterion, optimizer, scheduler, epoch, iter_meter, batch_size)
        test(model, device, test_batch_iterator, criterion, epoch, iter_meter)
        pass
