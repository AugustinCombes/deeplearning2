import scipy.io
import numpy as np
import torch


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path_documents='data/rawdata/mnist_all.mat', mode='train', size=None):

        raw_data = scipy.io.loadmat(path_documents)

        assert mode in ['train', 'test']

        if mode == 'train':
            data = np.concatenate(
                [raw_data[f'train{i}'] for i in range(10)], axis=0)
            labels = np.concatenate(
                [np.full(raw_data[f'train{i}'].shape[0], i) for i in range(10)], axis=0)

        elif mode == 'test':
            data = np.concatenate(
                [raw_data[f'test{i}'] for i in range(10)], axis=0)
            labels = np.concatenate(
                [np.full(raw_data[f'test{i}'].shape[0], i) for i in range(10)], axis=0)

        data = data.reshape(-1, 28, 28)

        # Binarize data
        data[data <= 127] = 0
        data[data > 127] = 1

        # Option to use only a subset of the data 
        if size is not None : 
            data = data[0:size, :, :]
            labels = labels[0:size]

        self.data = data
        self.labels = labels
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "data": self.data[index],
            "labels": self.labels[index]
        }
