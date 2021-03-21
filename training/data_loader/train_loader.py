import numpy as np
import math
from tensorflow.keras.utils import Sequence
import os


class TrainLoader(Sequence):
    def __init__(self, root, split, batch_size=16, input_length=59049, shuffle=False):
        self.root = root
        self.input_length = input_length
        self.split = split + ".npy"
        self.get_songlist()
        self.binary = np.load(os.path.join(self.root, "mtat", "binary.npy"))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        # indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        npy_list = []
        tag_list = []
        for i in range(self.batch_size):
            file_index = idx * self.batch_size + i
            if file_index >= 15247:
                npy_list.append(np.zeros((59049,)))
                tag_list.append(np.zeros((50,)))
                continue

            ix, fn = self.fl[file_index].split("\t")
            npy_path = (
                os.path.join(self.root, "mtat", "npy", fn.split("/")[1][:-3]) + "npy"
            )
            npy = np.load(npy_path)
            # scaler = max(max(npy), np.abs(min(npy)))
            # npy /= scaler

            random_idx = int(
                np.floor(np.random.random(1) * (len(npy) - self.input_length))
            )
            x = np.array(npy[random_idx : random_idx + self.input_length])
            tag_binary = self.binary[int(ix)]
            npy_list.append(x)
            tag_list.append(tag_binary)

        npy_list = np.array(npy_list)
        tag_list = np.array(tag_list)
        return npy_list, tag_list

    def get_songlist(self):
        self.fl = np.load(os.path.join(self.root, "mtat", self.split))
        # print(len(self.fl))

    def get_npy(self, index):
        ix, fn = self.fl[index].split("\t")
        npy_path = os.path.join(self.root, "mtat", "npy", fn.split("/")[1][:-3]) + "npy"
        npy = np.load(npy_path)
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx : random_idx + self.input_length])
        tag_binary = self.binary[int(ix)]
        return npy, tag_binary

    def on_epoch_end(self):
        self.indices = np.arange(len(self.fl))
        # if self.shuffle == True:
        # np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.fl) / self.batch_size)
