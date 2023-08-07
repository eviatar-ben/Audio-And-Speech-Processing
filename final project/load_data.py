import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn


class BatchIterator:
    def __init__(self, x, y, input_lengths, label_lengths, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.input_lengths = input_lengths
        self.label_lengths = label_lengths
        self.batch_size = batch_size
        self.num_samples = len(x)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.shuffle = shuffle
        self.current_batch = 0

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
            batch_y = self.y[start_idx:end_idx]
            batch_input_lengths = self.input_lengths[start_idx:end_idx]
            batch_label_lengths = self.label_lengths[start_idx:end_idx]

            self.current_batch += 1

            return batch_x, batch_y, batch_input_lengths, batch_label_lengths
        else:
            raise StopIteration


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


def load_wavs_data(load_again=False, save=False,
                   path=r"C:\Users\eviatar\PycharmProjects\Audio-And-Speech-Processing\final project\an4"):
    text_transform = TextTransform()

    if not load_again:
        all_spectrogram = torch.load("all_spectrogram.pt")
        all_labels = torch.load("all_labels.pt")
        all_input_lengths = torch.load("all_input_lengths.pt")
        all_label_lengths = torch.load("all_label_lengths.pt")
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
                        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)(waveform)
                        mfcc = mfcc.squeeze(0)
                        mfcc = mfcc.transpose(0, 1)  # todo: check if transpose is needed
                        # add mfcc to y:
                        spectrogram.append(mfcc)

                        # load txt:
                        with open(os.path.join(root2, file), 'r') as f:
                            text = f.read()
                            int_text = text_transform.text_to_int(text.lower())
                        # add text to labels:
                        label = torch.Tensor(int_text)
                        labels.append(label)
                        input_lengths.append(mfcc.shape[0] // 2)
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
        torch.save(all_spectrogram, "all_spectrogram.pt")
        torch.save(all_labels, "all_labels.pt")
        torch.save(all_input_lengths, "all_input_lengths.pt")
        torch.save(all_label_lengths, "all_label_lengths.pt")
    return all_spectrogram, all_labels, all_input_lengths, all_label_lengths


def get_batch_iterator(data_type, batch_size=10):
    if data_type not in ["test", "train", "val"]:
        raise ValueError("data_type must be one of [test, train, val]")
    all_spectrogram, all_labels, all_input_lengths, all_label_lengths = load_wavs_data(load_again=False, save=False)
    return BatchIterator(all_spectrogram[data_type], all_labels[data_type],
                         all_input_lengths[data_type], all_label_lengths[data_type], batch_size)
