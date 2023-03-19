import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import string

chars = string.digits+string.ascii_lowercase
idx2char, char2idx = dict(), dict()
for idx,char in enumerate(chars):
    idx2char[idx] = char
    char2idx[char] = idx

class BinaryAlphaDigitsDataset(torch.utils.data.Dataset):
    def __init__(self, path_documents='data/rawdata/binaryalphadigs.mat', mode="all", restrict_labels=False):
        # load data
        data = scipy.io.loadmat(path_documents)['dat']

        # restrict labels, if False, every decimal number and ascii letters are used
        if restrict_labels:
            index = []
            for label in restrict_labels:
                index.append(char2idx[label])
            index = np.array(index)
            data = data[index]

        x, y = data.shape
        labels = np.arange(x).T.reshape(-1, 1).repeat(y, axis=-1)
        
        # restrict data to train or test set
        assert mode in ["train", "test", "all"]
        
        if mode != "all":
            n_sample = data.shape[1]
            train_data, test_data = data[:int(0.8*n_sample)], data[int(0.8*n_sample):]
            train_labels, test_labels = labels[:int(0.8*n_sample)], labels[int(0.8*n_sample):]
            data = train_data if mode =="train" else test_data
            labels = train_labels if mode=="train" else test_labels

        self.data = np.stack(data.reshape(-1))
        self.labels = np.stack(labels.reshape(-1))

        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "data": self.data[index],
            "labels": self.labels[index]
        }