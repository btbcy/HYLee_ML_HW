import numpy as np


class StandardScaler:

    def __init__(self):
        self.__mean = None
        self.__std = None

    def mean_(self):
        return self.__mean

    def var_(self):
        return self.__std ** 2

    def fit(self, X):
        """ Compute the mean and std to be used for later scaling.

        Args:
            X (ndarray): shape (n_samples, ):

        """
        self.__mean = X.mean()
        self.__std = X.std()

    def transform(self, X):
        """ Do standard saler transfrom

        Args:
            X (ndarray): shape (n_samples, ):

        Returns:
            ndarray: shape (n_samples, ), the same as input data.

        """
        data = X.copy()
        new_data = (data - self.__mean) / self.__std
        return new_data
