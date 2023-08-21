import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from jiwer import wer
import Model
import Utils
from HyperParameters import WB


def train(model, device, batch_iterator, criterion, optimizer, scheduler, epoch):
    model.train()
    data_len = len(batch_iterator)
    for batch_idx, _data in enumerate(batch_iterator):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)  # using log_softmax instead of softmax for numerical stability
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, torch.from_numpy(input_lengths), torch.from_numpy(label_lengths))
        loss.backward()

        optimizer.step()
        scheduler.step()

        if batch_idx % 50 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / data_len, loss.item()))
            if WB:
                wandb.log({"train_loss": loss.item()})
                decoded_preds, decoded_targets = Utils.greedy_decoder(output.transpose(0, 1), labels,
                                                                      label_lengths)
                wer_sum = 0
                for j in range(len(decoded_preds)):
                    wer_sum += wer(decoded_targets[j], decoded_preds[j])

                wandb.log({"train_wer": wer_sum / len(decoded_preds)})


def validation(model, device, val_loader, criterion, epoch):
    print('\nevaluating...')
    model.eval()
    val_loss = 0
    val_wer = []
    with torch.no_grad():
        for i, _data in enumerate(val_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, torch.from_numpy(input_lengths), torch.from_numpy(label_lengths))
            val_loss += loss.item() / len(val_loader)

            decoded_preds, decoded_targets = Utils.greedy_decoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                val_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_wer = sum(val_wer) / len(val_wer)

    print(
        'val set: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(val_loss, avg_wer))

    # print a sample of the val data and decoded predictions against the true labels
    if epoch % 10 == 0:
        print('Ground Truth -> Decoded Prediction')
        for i in range(10):
            print('{} -> {}'.format(decoded_targets[i], decoded_preds[i]))

    if WB:
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_wer": avg_wer})


def train_and_validation(hparams, batch_iterators):
    train_loader = batch_iterators[0]
    val_loader = batch_iterators[1]
    epochs = hparams['epochs']

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model.init_model(hparams).to(device)

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=len(train_loader),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch)

        validation(model, device, val_loader, criterion, epoch)


def deep_speech_train_and_validation(hparams, batch_iterators):
    train_loader = batch_iterators[0]
    val_loader = batch_iterators[1]
    epochs = hparams['epochs']

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = hparams['n_feats']
    if hparams['delta_delta']:
        feats *= 3
    elif hparams['delta']:
        feats *= 2

    model = Model.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], feats, hparams['stride'], hparams['dropout']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=len(train_loader),  # todo: check if this is correct
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
        validation(model, device, val_loader, criterion, epoch)
