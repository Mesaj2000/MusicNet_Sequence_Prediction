import os
import random
from glob import glob
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util import *

class Composer_Agnostic_Note_Prediction(Dataset):
    def __init__(self, train=True, small=False):
        data_dir = TRAIN_DIR if train else TEST_DIR
        
        data_paths = sorted(glob(f'{data_dir}data/*.npy'), 
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        label_paths = [path.replace('data', 'labels') for path in data_paths]

        if small:
            data_paths = data_paths[:500]
            label_paths = label_paths[:500]

        self.data = [np.load(path) for path in data_paths]
        self.labels = [np.load(path) for path in label_paths]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])


class Composer_Classifier(Dataset):
    def __init__(self, train=True):
        data_dir = TRAIN_DIR if train else TEST_DIR
        
        all_data = {int(i): glob(f'{data_dir}{i}/*.npy') 
                    for i in os.listdir(data_dir)}
        
        samples_per_class = min([len(samples) for samples in all_data.values()])

        selected_data = {i: random.sample(samples, samples_per_class)
                         for i, samples in all_data.items()}

        num_labels = len(selected_data)

        self.data = [(np.load(sample), torch.tensor([1.0 if j == i else 0.0 for j in range(num_labels)])) 
                    for i, samples in selected_data.items() 
                    for sample in samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



if __name__ == "__main__":
    dataset = Composer_Classifier(train=False)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    for i, (data, label) in enumerate(dataloader):
        print(i, data, label, '\n', sep='  ')

        if i >= 2:
            break