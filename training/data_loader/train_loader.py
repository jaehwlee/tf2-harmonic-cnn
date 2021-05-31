import numpy as np
import math
from tensorflow.keras.utils import Sequence
import os


class TrainLoader(Sequence):
    def __init__(self, root, batch_size=16, input_length=80000):
        self.root = root
        self.input_length = input_length
        self.get_songlist()
        self.binary = np.load(os.path.join(self.root, "mtat", "binary.npy"))
        self.batch_size = batch_size

        
    def __getitem__(self, idx):
        npy_list = []
        tag_list = []
        
        for i in range(self.batch_size):
            file_index = idx * self.batch_size + i
            npy, tag_binary = self.get_npy(file_index)
            npy_list.append(npy)
            tag_list.append(tag_binary)

        npy_list = np.array(npy_list)
        tag_list = np.array(tag_list)
        return npy_list, tag_list

    
    def get_songlist(self):
        self.fl = np.load(os.path.join(self.root, "mtat", self.split))

        
    def get_npy(self, index):
        ix, fn = self.fl[index].split("\t")
        npy_path = os.path.join(self.root, "mtat", "npy", fn.split("/")[1][:-3]) + "npy"
        npy = np.load(npy_path)
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx : random_idx + self.input_length])
        tag_binary = self.binary[int(ix)]
        return npy.astype('float32'), tag_binary.astype('float32')

    
    def __len__(self):
        return int(np.floor(len(self.fl) / self.batch_size))
