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

        # self.randomize_indices = np.random.permutation(target.index)
        # print(target.index)
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
        runs = self.test_idx[wrong_predicitions_indices]
        if len(runs) != 0:
            return runs + 1
        return runs

    def check_if_two_classes(self):
        return (sum(self.target[self.train_idx] == 0) > 0) and \
                (sum(self.target[self.train_idx] == 1) > 0) and \
                (sum(self.target[self.test_idx] == 0) > 0) and \
                (sum(self.target[self.test_idx] == 1) > 0)

    def load_train_test_data(self, num_samples = None):
        # keep sampling until we get two classes in both train and test
        loop_count = 0
        while True:
            # randomly sample subset of dataset
            sample_indices = np.random.permutation(self.target.index)[:num_samples]

            if num_samples != None:
                self.samples_indices = sample_indices
                self.target = self.target.loc[self.samples_indices]
                self.data = self.data.loc[self.samples_indices]
                # print(type(self.data))

                self.data = self.data.replace(np.nan, 0)
                # df.replace(np.nan, 0)

                # print(type(self.data))
            else:
                self.samples_indices = np.arange(len(self.target))

            X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size = self.train_test_ratio, shuffle = False)
            self.train_idx = y_train.index
            self.test_idx = y_test.index

            if self.check_if_two_classes():
                break
            # if cannot sample for two classes
            elif loop_count > 50:
                print("Unable to get a representative sample")
                sys.exit()

            loop_count += 1
        
        # make sure that there are more than 1 class
        assert(sum(self.target[self.train_idx] == 0) > 0)
        assert(sum(self.target[self.train_idx] == 1) > 0)
        assert(sum(self.target[self.test_idx] == 0) > 0)
        assert(sum(self.target[self.test_idx] == 1) > 0)

        return X_train, X_test, y_train, y_test

    def load_balanced_train_test(self):
        pass