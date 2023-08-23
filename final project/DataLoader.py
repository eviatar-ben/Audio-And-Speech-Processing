import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from Utils import TextTransform
from HyperParameters import deep_speech_hparams
import torchvision.transforms as transforms
import random
import cv2
import librosa

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


def load_wavs_data(load_again=False, save=False, path=r".\an4",feat_type = 'mfcc', n_feats=13, stretch_train=False):
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
                        # if stretch_train set to true, add augmented data to train set with augmentation probability of 0.2
                        s_wavform = torch.from_numpy(librosa.effects.time_stretch(waveform.squeeze(0).numpy(), rate=random.uniform(0.5, 1.5)))
                        #
                        # mfcc:
                        if feat_type == 'mfcc':
                            feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)

                            s_feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_feats)(s_wavform)
                            s_feats = s_feats.squeeze(0)
                            s_feats = s_feats.transpose(0, 1)

                        elif feat_type == 'mel_spectrogram':
                            feats = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)

                            s_feats = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_feats)(s_wavform)
                            s_feats = s_feats.squeeze(0)
                            s_feats = s_feats.transpose(0, 1)

                        elif feat_type == 'mfcc_with_delta':
                            assert n_feats % 2 == 0
                            n_mfcc_feats = n_feats // 2
                            feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)
                            delta = torchaudio.transforms.ComputeDeltas()(feats)
                            feats = torch.cat((feats, delta), 1)

                            s_feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc_feats)(s_wavform)
                            s_feats = s_feats.squeeze(0)
                            s_feats = s_feats.transpose(0, 1)
                            delta = torchaudio.transforms.ComputeDeltas()(s_feats)
                            s_feats = torch.cat((s_feats, delta), 1)

                        elif feat_type == 'mfcc_with_delta_delta':
                            assert n_feats % 3 == 0
                            n_mfcc_feats = n_feats // 3
                            feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc_feats)(waveform)
                            feats = feats.squeeze(0)
                            feats = feats.transpose(0, 1)
                            delta = torchaudio.transforms.ComputeDeltas()(feats)
                            delta_delta = torchaudio.transforms.ComputeDeltas()(delta)
                            feats = torch.cat((feats, delta, delta_delta), 1)

                            s_feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc_feats)(s_wavform)
                            s_feats = s_feats.squeeze(0)
                            s_feats = s_feats.transpose(0, 1)
                            delta = torchaudio.transforms.ComputeDeltas()(s_feats)
                            delta_delta = torchaudio.transforms.ComputeDeltas()(delta)
                            s_feats = torch.cat((s_feats, delta, delta_delta), 1)

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

                        if stretch_train and dir == 'train' and random.random() < 0.4:
                            spectrogram.append(s_feats)
                            labels.append(label)
                            input_lengths.append(s_feats.shape[0] // 2)
                            label_lengths.append(len(label))

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


def get_batch_iterator(data_type, batch_size=deep_speech_hparams["batch_size"], feat_type='mfcc', n_feats=13, deletion_augmentations=False, stretch_augmentation=False):
    if data_type not in ["test", "train", "val"]:
        raise ValueError("data_type must be one of [test, train, val]")
    all_spectrogram, all_labels, all_input_lengths, all_label_lengths = load_wavs_data(load_again=False, save=True, feat_type=feat_type, n_feats=n_feats, stretch_train=stretch_augmentation)

    batch_iterator = BatchIterator(all_spectrogram[data_type], all_labels[data_type],
                                   all_input_lengths[data_type], all_label_lengths[data_type], batch_size, augmentation=apply_augmentations if deletion_augmentations else None)

    return batch_iterator


def apply_augmentations(data, input_lengths, augmentation_prob=0.5):
    augmented_data = []
    freq_mask_param = 27
    time_mask_param = 80
    freq_mask_transform = torchaudio.transforms.FrequencyMasking(freq_mask_param)
    time_mask_transform =torchaudio.transforms.TimeMasking(time_mask_param)

    for idx, item in enumerate(data):
        if random.random() < augmentation_prob:
            item = freq_mask_transform(item)
            item = time_mask_transform(item)
        augmented_data.append(item)

    # convert the list to a tensor
    augmented_data = torch.stack(augmented_data)

    return augmented_data, input_lengths
