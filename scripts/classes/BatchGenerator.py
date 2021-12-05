# create the BatchGenerator class
# code based on: 
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes
from tensorflow import keras
import numpy as np
import os

class BatchGenerator(keras.utils.Sequence):
    
    def __init__(
        self, 
        list_filehashes, 
        labels,
        data_dir,
        window_size,
        n_channels,
        batch_size=32,  
        n_classes=2, 
        shuffle=True
    ):
        """
        Generate batches of data for training.
    
        Args:
            data (str): the path to a directory of npy files
            labels (dict): a dictionary of filehashes and their corresponding labels
            list_filehashes (list): the list of filehashes to use
            window_size (int): the number of timesteps to use as a window
            n_channels (int): the number of features in the data
            batch_size (int): the number of samples per batch. Defaults to 32.
            n_classes (int): the number of classes for the target. Defaults to 2.
            shuffle (bool): whether to shuffle the data. Defaults to True.
        """
        self.data_dir = data_dir
        self.labels = labels
        self.batch_size = batch_size
        self.window_size = window_size
        self.n_channels = n_channels
        self.list_IDs = list_filehashes
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, window_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.window_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # get X_tmp and then cut down to the window size
            X_tmp = np.load(os.path.join(self.data_dir, ID + '.npy'))

            X[i,] = X_tmp[:self.window_size,:]

            # Store class
            y[i] = self.labels[ID]

        return X, y

