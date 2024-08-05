import numpy as np
from random import shuffle
import threading

class Numpy_DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        """
        Args:
            dataset (CityscapesDataset): The dataset from which to load data.
            batch_size (int): How many samples per batch to load.
            shuffle (bool): Set to True to have the data reshuffled at every epoch.
            num_workers (int): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = list(range(len(dataset)))
        self.lock = threading.Lock()
        self.index = 0

        if self.shuffle:
            shuffle(self.indices)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            shuffle(self.indices)
        return self

    def __next__(self):
        with self.lock:
            if self.index >= len(self.indices):
                raise StopIteration

            batch_indices = self.indices[self.index:self.index + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            self.index += self.batch_size

            # Separate images and labels into two lists
            images, labels = zip(*batch)

            # Convert lists to numpy arrays
            images = np.stack(images)
            labels = np.stack(labels)

            return images, labels





