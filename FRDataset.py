import torch
import numpy as np
import os
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import os


def rename_files(path):
    classes = os.listdir(path)
    for classs in classes:
        for file in os.listdir(path + '/' + classs):
            if file.endswith('jpg'):
                os.rename((path + '/' + classs + '/' + file),(path + '/' + classs + '/' + classs + "_" + file))


def parse_species(fname):
    parts = fname.split('_')
    return parts[0]


def open_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FRDataset(Dataset):
    def __init__(self, path, transform):
        super(FRDataset, self).__init__()
        self.path = path
        self.files = []
        self.classes = [fname for fname in os.listdir(path)]
        for classs in self.classes:
            for file in os.listdir(path + '/' + classs):
                if file.endswith('jpg'):
                    self.files.append(file)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fname = self.files[i]
        species = parse_species(fname)
        fpath = os.path.join(self.path, species, fname)
        img = self.transform(open_image(fpath))
        class_idx = self.classes.index(species)
        return img, class_idx


if __name__ == '__main__':
   pass

