import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from Utils import TextTransform
from HyperParameters import deep_speech_hparams
import torchvision.transforms as transforms
import random
from HyperParameters import delta, delta_delta


class BatchIterator:
    def __init__(self, x, y, input_lengths, label_lengths, batch_size, shuffle=True, augmentation=None):
        self.x = x
        self.y = y
        self.input_lengths = input_lengths
        self.label_lengths = label_lengths
        self.batch_size = batch_size
        self.num_samples = len(x)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.shuffle = shuffle
        self.current_batch = 0
        self.augmentation = augmentation

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.num_samples)
            self.x = self.x[indices]
            self.y = self.y[indices]
            self.input_lengths = np.asarray(self.input_lengths)[indices]
            self.label_lengths = np.asarray(self.label_lengths)[indices]
        self.current_batch = 0
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_batch < self.num_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
            batch_x = self.x[start_idx:end_idx]
            if self.augmentation:
                batch_x, batch_input_lengths = self.augmentation(batch_x, self.input_lengths[start_idx:end_idx])
            batch_y = self.y[start_idx:end_idx]
            batch_input_lengths = self.input_lengths[start_idx:end_idx]
            batch_label_lengths = self.label_lengths[start_idx:end_idx]

            self.current_batch += 1

            return batch_x, batch_y, batch_input_lengths, batch_label_lengths
        else:
            raise StopIteration


def load_wavs_data(load_again=False, save=False, path=r".\an4",feat_type = 'mfcc', n_feats=13):
    text_transform = TextTransform()

    if load_again:
        all_spectrogram = torch.load("data/all_spectrogram.pt")
        all_labels = torch.load("data/all_labels.pt")
        all_input_lengths = torch.load("data/all_input_lengths.pt")
        all_label_lengths = torch.load("data/all_label_lengths.pt")
        return all_spectrogram, all_labels, all_input_lengths, all_label_lengths

    valid_file = ["test", "train", "val"]
    all_spectrogram = {"test": [], "train": [], "val": []}
    all_labels = {"test": [], "train": [], "val": []}
    all_input_lengths = {"test": [], "train": [], "val": []}
    all_label_lengths = {"test": [], "train": [], "val": []}
    for dir in os.listdir(path):
        if dir in valid_file:
            spectrogram = []
            labels = []
            input_lengths = []
            label_lengths = []
            for root2, dirs2, files2 in os.walk(os.path.join(path, dir)):
                for file in files2:
                    if file.endswith(".txt"):
                        # change suffix to wav
                        wav = file.replace(".txt", ".wav")
                        # print(os.path.join(root2, file))
                        # change last dir to wav instead of txt
                        root2_wav = root2.replace("txt", "wav")

                        # load wav:
                        waveform, sample_rate = torchaudio.load(os.path.join(root2_wav, wav))
                        # mfcc:
<<<<<<< HEAD
                        if feat_type == 'mfcc':
                            feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)
                        elif feat_type == 'mel_spectrogram':
                            feats = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)
                        elif feat_type == 'gammatone':
                            feats = torchaudio.transforms.Gammatone(sample_rate=sample_rate, n_mels=n_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)
=======
                        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)(waveform)
                        mfcc = mfcc.squeeze(0)
                        mfcc = mfcc.transpose(0, 1)
                        if delta:
                            mfcc_delta = torchaudio.transforms.ComputeDeltas()(mfcc)
                            # mfcc_delta_delta = torchaudio.transforms.ComputeDeltas()(mfcc_delta)
                            mfcc = torch.cat((mfcc, mfcc_delta), dim=1)
>>>>>>> 36175764f82f32e6964e4f9d23c92ce32015bdd9
                        # add mfcc to y:
                        spectrogram.append(feats)

                        # load txt:
                        with open(os.path.join(root2, file), 'r') as f:
                            text = f.read()
                            int_text = text_transform.text_to_int(text.lower())
                        # add text to labels:
                        label = torch.Tensor(int_text)
                        labels.append(label)
                        input_lengths.append(feats.shape[0] // 2)
                        label_lengths.append(len(label))
                    # print(file)

            # todo maybe the padding should be done for all the data together (train test and val)
            spectrogram = nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).unsqueeze(1).transpose(2, 3)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

            all_spectrogram[dir] = spectrogram
            all_labels[dir] = labels
            all_input_lengths[dir] = input_lengths
            all_label_lengths[dir] = label_lengths

    if save:
        torch.save(all_spectrogram, "data/all_spectrogram.pt")
        torch.save(all_labels, "data/all_labels.pt")
        torch.save(all_input_lengths, "data/all_input_lengths.pt")
        torch.save(all_label_lengths, "data/all_label_lengths.pt")
    return all_spectrogram, all_labels, all_input_lengths, all_label_lengths


<<<<<<< HEAD
def get_batch_iterator(data_type, batch_size=deep_speech_hparams["batch_size"],feat_type='mfcc',n_feats=13, augmentations=False):
    if data_type not in ["test", "train", "val"]:
        raise ValueError("data_type must be one of [test, train, val]")
    all_spectrogram, all_labels, all_input_lengths, all_label_lengths = load_wavs_data(load_again=False, save=True,feat_type=feat_type, n_feats=n_feats)
=======
def get_batch_iterator(batch_size=deep_speech_hparams["batch_size"], augmentations=False):
    all_spectrogram, all_labels, all_input_lengths, all_label_lengths = load_wavs_data(load_again=False, save=True)
>>>>>>> 36175764f82f32e6964e4f9d23c92ce32015bdd9

    test_iterator = BatchIterator(all_spectrogram["test"], all_labels["test"],
                                  all_input_lengths["test"], all_label_lengths["test"], batch_size,
                                  augmentation=apply_augmentations if augmentations else None)
    train_iterator = BatchIterator(all_spectrogram["train"], all_labels["train"],
                                   all_input_lengths["train"], all_label_lengths["train"], batch_size,
                                   augmentation=apply_augmentations if augmentations else None)
    val_iterator = BatchIterator(all_spectrogram["val"], all_labels["val"],
                                 all_input_lengths["val"], all_label_lengths["val"], batch_size,
                                 augmentation=apply_augmentations if augmentations else None)
    return train_iterator, test_iterator, val_iterator


def apply_augmentations(data, input_lengths, augmentation_prob=0.5):
    augmented_data = []
    freq_mask_param = 27
    time_mask_param = 80
    # warp_transform = transforms.RandomApply([torchaudio.transforms.TimeStretch()],
    #                                         p=augmentation_prob)
    freq_mask_transform = transforms.RandomApply([torchaudio.transforms.FrequencyMasking(freq_mask_param)],
                                                 p=augmentation_prob)
    time_mask_transform = transforms.RandomApply([torchaudio.transforms.TimeMasking(time_mask_param)],
                                                 p=augmentation_prob)

    for idx, item in enumerate(data):
        if random.random() < augmentation_prob:
            # rate = random.uniform(0.7, 1.3)
            # item = warp_transform(item, fixed_rate=rate)
            item = freq_mask_transform(item)
            item = time_mask_transform(item)
        augmented_data.append(item)

    # convert the list to a tensor
    augmented_data = torch.stack(augmented_data)

    return augmented_data, input_lengths
