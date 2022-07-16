import os
import numpy as np
from sklearn.model_selection import train_test_split
import sys

class Dataloader(object):
    def __init__(self, data, target, train_test_ratio, size_of_test_set, random_state, isTest = False):
        self.data = data
        self.target = target
        self.train_test_ratio = train_test_ratio
        self.isTest = isTest
        self.samples_indices = None
        self.size_of_test_set = size_of_test_set
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        np.random.seed(random_state)

        self.randomize_indices = np.random.permutation(target.index)
        # print(self.randomize_indices)
        # sys.exit()

    def binary_analysis(self, target):
        index_of_zeros = np.where(target == 0)
        index_of_ones = np.where(target == 1)

        return index_of_zeros[0], index_of_ones[0]
    
    def get_data_distribution(self):
        index_of_zeros, index_of_ones = self.binary_analysis(self.target)
        return len(index_of_ones) / (len(index_of_zeros) + len(index_of_ones))
    
    def get_sample_indices(self):
        return self.samples_indices

    def get_wrong_pred_runs(self, y_pred, y_test):
        wrong_predicitions_indices = (y_pred != y_test)
        # print((y_pred != y_test))
        # print(y_test)
        # print(y_pred)
        runs = self.test_idx[wrong_predicitions_indices]
        # adjust for change of index
        # print(runs)
        if len(runs) != 0:
            # print("here", runs+1)
            return runs + 1
        return runs

    def load_train_test_data(self, num_samples = None):
        # randomly sample subset of dataset
        sample_indices = self.randomize_indices[:num_samples]
        # print(len(sample_indices))
        # print(sample_indices)

        if num_samples != None:
            self.samples_indices = sample_indices
            # self.samples_indices = np.random.choice(np.arange(len(self.target)), size = num_samples, replace = False)
            # print(self.target)
            # print(self.target.loc[62])
            self.target = self.target.loc[self.samples_indices]
            self.data = self.data.loc[self.samples_indices]
        else:
            self.samples_indices = np.arange(len(self.target))
            # target = self.target
            # data = self.data

        # X_train, X_test, y_train, y_test = 
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size = self.train_test_ratio, shuffle = False)
        # print(len(samples_indices))
        self.train_idx = y_train.index
        self.test_idx = y_test.index
        # print(self.train_idx)

        # X_train, X_rem, y_train, y_rem = train_test_split(self.data, self.target, test_size = self.size_of_test_set)
        # X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
        # self.valid_idx = y_valid.index
        # self.train_idx = y_train.index
        # self.test_idx = y_test.index

        # if self.isTest:
        #     print("Test Dist is", len(index_of_ones) / (len(index_of_ones)+len(index_of_zeros)))

        return X_train, X_test, y_train, y_test
        # return X_train, X_valid, X_test, y_train, y_valid, y_test 

    def load_balanced_train_test(self):
        pass