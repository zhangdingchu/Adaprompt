from abc import ABCMeta, abstractmethod
import torch
import torch.utils.data as data
import numpy as np
import logging


__all__ = ['StreamDataset']

class StreamDataset(metaclass=ABCMeta):
    NAME = "NULL" 

    def __init__(self, root="./MyDATA", batch_size=64, seed=998244353):
        self.root, self.seed = root, seed
        self.batch_size = batch_size
        self.rand = np.random.RandomState(seed=seed)

    @abstractmethod
    def run(self, ):pass
    
    @abstractmethod
    def download(self,): pass

    @abstractmethod
    def num_classes(self,): pass

    @staticmethod
    def load_dataset(dataset):
        test_loader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        x_test, y_test = [], []
        for i, (x, y) in enumerate(test_loader):
            x_test.append(x)
            y_test.append(y)
        x_test_tensor = torch.cat(x_test)
        y_test_tensor = torch.cat(y_test)
        return x_test_tensor, y_test_tensor