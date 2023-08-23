import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from jiwer import wer, cer
import Model
import Utils
from HyperParameters import WB


def train(model, device, batch_iterator, criterion, optimizer, scheduler, epoch):
    total_train_loss = 0
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

        total_train_loss += loss.item()

        if batch_idx % 50 == 0 or batch_idx == data_len - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / data_len, loss.item()))

        # log to wandb once every epoch:
        if batch_idx == data_len - 1 and WB:
            wandb.log({"train_loss": total_train_loss / data_len}, step=epoch)
            decoded_preds, decoded_targets = Utils.greedy_decoder(output.transpose(0, 1), labels,
                                                                  label_lengths)
            wer_sum = 0
            cer_sum = 0
            for j in range(len(decoded_preds)):
                wer_sum += wer(decoded_targets[j], decoded_preds[j])
                cer_sum += cer(decoded_targets[j], decoded_preds[j])

            wandb.log({"train_wer (on last batch)": wer_sum / len(decoded_preds)}, step=epoch)
            wandb.log({"train_cer (on last batch)": cer_sum / len(decoded_preds)}, step=epoch)


def validation(model, device, val_loader, criterion, epoch):
    print('\nevaluating...')
    model.eval()
    val_loss = 0
    val_wer = []
    val_cer = []
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
                val_cer.append(cer(decoded_targets[j], decoded_preds[j]))

    avg_wer = sum(val_wer) / len(val_wer)
    avg_cer = sum(val_cer) / len(val_cer)

    print(
        'val set: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(val_loss, avg_wer))

    # print a sample of the val data and decoded predictions against the true labels
    if epoch % 10 == 0:
        print('Ground Truth -> Decoded Prediction')
        for i in range(10):
            print('{} -> {}'.format(decoded_targets[i], decoded_preds[i]))

    if WB:
        wandb.log({"val_loss": val_loss}, step=epoch)
        wandb.log({"val_wer": avg_wer}, step=epoch)
        wandb.log({"val_cer": avg_cer}, step=epoch)


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

<<<<<<< HEAD

def test_epoch(model, device, test_loader, criterion, epoch):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_wer = []
    test_cer = []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, torch.from_numpy(input_lengths), torch.from_numpy(label_lengths))
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = Utils.greedy_decoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))

    avg_wer = sum(test_wer) / len(test_wer)
    avg_cer = sum(test_cer) / len(test_cer)

    print(
        'test set: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(test_loss, avg_wer))

    # print a sample of the test data and decoded predictions against the true labels
    if epoch % 10 == 0:
        print('Ground Truth -> Decoded Prediction')
        for i in range(10):
            print('{} -> {}'.format(decoded_targets[i], decoded_preds[i]))

    if WB:
        wandb.log({"test_loss": test_loss}, step=epoch)
        wandb.log({"test_wer": avg_wer}, step=epoch)
        wandb.log({"test_cer": avg_cer}, step=epoch)


def test(hparams, test_batch_iterator, model, save=True):
    epochs = hparams['epochs']

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CTCLoss(blank=28).to(device)

    for epoch in range(1, epochs + 1):
        # break
        test_epoch(model, device, test_batch_iterator, criterion, epoch)

    if save:
        torch.save(model.state_dict(), 'data/model.pt')
    return model
