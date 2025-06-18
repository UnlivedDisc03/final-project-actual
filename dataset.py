import os
import torch
import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
from config import Config

# #below code used from https://y-t-g.github.io/tutorials/yolo-class-balancing/
# """MIT License
# Copyright (c) 2024 Mohammed Yasin
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

class Dataset:
    def __init__(self):
        self.config = Config()
        self.change_path_in_dataset_yaml()
        self.push_dataset()

    def change_path_in_dataset_yaml(self):
        cwd = os.getcwd()
        path = os.path.join(cwd, 'dataset')
        self.dataset_path = os.path.join(cwd, 'dataset', 'dataset.yaml')

        with open(self.dataset_path, 'w+') as f:
            f.write(fr"""names:
  - b_fully_ripened
  - b_half_ripened
  - b_green
  - l_fully_ripened
  - l_half_ripened
  - l_green
nc: 6
path: {path}
train: train\images
val: val\images
""")
            print("Dataset File Updated.")

    class YOLOWeightedDataset(YOLODataset):
        def __init__(self, *args, mode="train", **kwargs):
            #Initialize the WeightedDataset.
            #Args: class_weights (list or numpy array): A list or array of weights corresponding to each class.

            super().__init__(*args, **kwargs)

            self.train_mode = "train" in self.prefix

            self.count_instances()
            class_weights = np.sum(self.counts) / self.counts

            # Aggregation function
            config = Config()
            if config.agg_function == "sum":
                self.agg_func = np.sum
            elif config.agg_function == "mean":
                self.agg_func = np.mean
            elif config.agg_function == "max":
                self.agg_func = np.max
            elif config.agg_function == "median":
                self.agg_func = np.median

            self.class_weights = np.array(class_weights)
            self.weights = self.calculate_weights()
            self.probabilities = self.calculate_probabilities()

        #1
        def count_instances(self):
            """
            Count the number of instances per class

            Returns:
                dict: A dict containing the counts for each class.
            """
            self.counts = [0 for i in range(len(self.data["names"]))]
            for label in self.labels:
                cls = label['cls'].reshape(-1).astype(int)
                for id in cls:
                    self.counts[id] += 1

            self.counts = np.array(self.counts)
            self.counts = np.where(self.counts == 0, 1, self.counts)
            print("counts: " +str(self.counts))

        def calculate_weights(self):
            """
            Calculate the aggregated weight for each label based on class weights.

            Returns:
                list: A list of aggregated weights corresponding to each label.
            """
            weights = []
            for label in self.labels:
                cls = label['cls'].reshape(-1).astype(int)

                # Give a default weight to background class
                if cls.size == 0:
                    weights.append(1)
                    continue

                # Take mean of weights
                # You can change this weight aggregation function to aggregate weights differently
                weight = self.agg_func(self.class_weights[cls])
                weights.append(weight)
                #print("Weights: " +str(weights))
            return weights

        def calculate_probabilities(self):
            """
            Calculate and store the sampling probabilities based on the weights.

            Returns:
                list: A list of sampling probabilities corresponding to each label.
            """
            total_weight = sum(self.weights)
            probabilities = [w / total_weight for w in self.weights]
            #print("Sampling Probabilities for each label: "+str(probabilities))
            return probabilities

        def __getitem__(self, index):
            """
            Return transformed label information based on the sampled index.
            """
            # Don't use for validation
            if not self.train_mode:
                return self.transforms(self.get_image_and_label(index))
            else:
                index = np.random.choice(len(self.labels), p=self.probabilities)
                return self.transforms(self.get_image_and_label(index))

    def push_dataset(self):
        if self.config.weighted_dataset:
            build.YOLODataset = self.YOLOWeightedDataset #weighted dataset
            print("Selected weighted dataset")
        else:
            build.YOLODataset = YOLODataset #normal dataset
            print("Selected Standard Dataset")

#dataset = Dataset()