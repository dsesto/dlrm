# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Description: generate inputs and targets for the dlrm benchmark
# The inputs and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i)  Criteo Kaggle Display Advertising Challenge Dataset
#    https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#    ii) Criteo Terabyte Dataset
#    https://labs.criteo.com/2013/12/download-terabyte-click-logs


from __future__ import absolute_import, division, print_function, unicode_literals

import sys
# others
from os import path

# numpy
import numpy as np
# pytorch
import torch
from torch.utils.data import Dataset

import data_utils


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets
class CriteoDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_ind_range,
        sub_sample_rate,
        randomize,
        split="train",
        raw_path="",
        pro_data="",
        memory_map=False,
        dataset_multiprocessing=False,
    ):
        # dataset
        # num_target_features = 1   # single target
        num_dense_features = 13  # 13 dense features
        # num_sparse_features = 26  # 26 sparse features
        # num_total_features = num_target_features + num_dense_features + num_sparse_features
        if dataset == "kaggle":
            days = 7
            out_file = "kaggleAdDisplayChallenge_processed"
        elif dataset == "terabyte":
            days = 24
            out_file = "terabyte_processed"
        else:
            raise (ValueError("Data set option is not supported"))
        self.max_ind_range = max_ind_range
        self.memory_map = memory_map

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
        self.npzfile = self.d_path + (
            (self.d_file + "_day") if dataset == "kaggle" else self.d_file
        )
        self.trafile = self.d_path + (
            (self.d_file + "_fea") if dataset == "kaggle" else "fea"
        )

        # check if pre-processed data is available
        data_ready = True
        if memory_map:
            for i in range(days):
                reo_data = self.npzfile + "_{0}_reordered.npz".format(i)
                if not path.exists(str(reo_data)):
                    data_ready = False
        else:
            if not path.exists(str(pro_data)):
                data_ready = False

        # pre-process data if needed
        # WARNNING: when memory mapping is used we get a collection of files
        if data_ready:
            print("Reading pre-processed data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            print("Reading raw data=%s" % (str(raw_path)))
            file = data_utils.getCriteoAdData(
                raw_path,
                out_file,
                max_ind_range,
                sub_sample_rate,
                days,
                split,
                randomize,
                dataset == "kaggle",
                memory_map,
                dataset_multiprocessing,
            )

        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]
        # print(self.offset_per_file)

        # setup data
        if memory_map:
            # setup the training/testing split
            self.split = split
            if split == "none" or split == "train":
                self.day = 0
                self.max_day_range = days if split == "none" else days - 1
            elif split == "test" or split == "val":
                self.day = days - 1
                num_samples = (
                    self.offset_per_file[days] - self.offset_per_file[days - 1]
                )
                self.test_size = int(np.ceil(num_samples / 2.0))
                self.val_size = num_samples - self.test_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")

            # load unique counts
            with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                self.counts = data["counts"]
            self.num_dense_features = num_dense_features  # X_int.shape[1]
            self.n_emb = len(self.counts)
            print("Sparse features= %d, Dense features= %d" % (self.n_emb, self.num_dense_features))

            # Load the test data
            # Only a single day is used for testing
            if self.split == "test" or self.split == "val":
                # only a single day is used for testing
                fi = self.npzfile + "_{0}_reordered.npz".format(self.day)
                with np.load(fi) as data:
                    self.X_int = data["X_int"]  # continuous  feature
                    self.X_cat = data["X_cat"]  # categorical feature
                    self.y = data["y"]  # target

        else:
            # load and preprocess data
            with np.load(file) as data:
                X_int = data["X_int"]  # continuous  feature
                X_cat = data["X_cat"]  # categorical feature
                y = data["y"]  # target
                self.counts = data["counts"]
            self.num_dense_features = X_int.shape[1]  # num_dense_features
            self.n_emb = len(self.counts)
            print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.num_dense_features))

            # create reordering
            indices = np.arange(len(y))

            if split == "none":
                # randomize all data
                if randomize == "total":
                    indices = np.random.permutation(indices)
                    print("Randomized indices...")

                X_int[indices] = X_int
                X_cat[indices] = X_cat
                y[indices] = y

            else:
                indices = np.array_split(indices, self.offset_per_file[1:-1])

                # randomize train data (per day)
                if randomize == "day":  # or randomize == "total":
                    for i in range(len(indices) - 1):
                        indices[i] = np.random.permutation(indices[i])
                    print("Randomized indices per day ...")

                train_indices = np.concatenate(indices[:-1])
                test_indices = indices[-1]
                test_indices, val_indices = np.array_split(test_indices, 2)

                print("Defined %s indices..." % (split))

                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(train_indices)
                    print("Randomized indices across days ...")

                # create training, validation, and test sets
                if split == "train":
                    self.X_int = [X_int[i] for i in train_indices]
                    self.X_cat = [X_cat[i] for i in train_indices]
                    self.y = [y[i] for i in train_indices]
                elif split == "val":
                    self.X_int = [X_int[i] for i in val_indices]
                    self.X_cat = [X_cat[i] for i in val_indices]
                    self.y = [y[i] for i in val_indices]
                elif split == "test":
                    self.X_int = [X_int[i] for i in test_indices]
                    self.X_cat = [X_cat[i] for i in test_indices]
                    self.y = [y[i] for i in test_indices]

            print("Split data according to indices...")

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        if self.memory_map:
            if self.split == "none" or self.split == "train":
                # check if need to swicth to next day and load data
                if index == self.offset_per_file[self.day]:
                    # print("day_boundary switch", index)
                    self.day_boundary = self.offset_per_file[self.day]
                    fi = self.npzfile + "_{0}_reordered.npz".format(self.day)
                    # print('Loading file: ', fi)
                    with np.load(fi) as data:
                        self.X_int = data["X_int"]  # continuous  feature
                        self.X_cat = data["X_cat"]  # categorical feature
                        self.y = data["y"]  # target
                    self.day = (self.day + 1) % self.max_day_range

                i = index - self.day_boundary
            elif self.split == "test" or self.split == "val":
                # only a single day is used for testing
                i = index + (0 if self.split == "test" else self.test_size)
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")
        else:
            i = index

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    def _default_preprocess(self, X_int, X_cat, y):
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
        if self.max_ind_range > 0:
            X_cat = torch.tensor(X_cat % self.max_ind_range, dtype=torch.long)
        else:
            X_cat = torch.tensor(X_cat, dtype=torch.long)
        y = torch.tensor(y.astype(np.float32))

        return X_int, X_cat, y

    def __len__(self):
        if self.memory_map:
            if self.split == "none":
                return self.offset_per_file[-1]
            elif self.split == "train":
                return self.offset_per_file[-2]
            elif self.split == "test":
                return self.test_size
            elif self.split == "val":
                return self.val_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train nor test.")
        else:
            return len(self.y)


def collate_wrapper_criteo_offset(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    y = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    # NOTE: (dsesto)
    # Offset of sparse (categorical) features for the EmbeddingBag
    sparse_features_offsets = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    # Indices of sparse (categorical) features for the EmbeddingBag
    sparse_features_indices = [X_cat[:, i] for i in range(featureCnt)]

    return X_int, torch.stack(sparse_features_offsets), torch.stack(sparse_features_indices), y


def make_criteo_data_and_loaders(args, offset_to_length_converter=False):
    train_data = CriteoDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing,
    )

    test_data = CriteoDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "test",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing,
    )

    collate_wrapper_criteo = collate_wrapper_criteo_offset

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_data, train_loader, test_data, test_loader
