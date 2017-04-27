"""Functions for downloading from internet or local file, and reading MNIST data."""
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, extract_images, extract_labels
from tensorflow.examples.tutorials.mnist import input_data


class DataSets(object):
    pass

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
VALIDATION_SIZE = 5000


def load_minst(src=None, path=None, one_hot=False):
    mnist = DataSets()
    if src:
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)
    if path:
        if path[-1] != '/':
            path += '/'
        train_images = extract_images(path + TRAIN_IMAGES)
        train_labels = extract_labels(path + TRAIN_LABELS, one_hot=one_hot)
        test_images = extract_images(path + TEST_IMAGES)
        test_labels = extract_labels(path + TEST_LABELS, one_hot=one_hot)

        validation_images = train_images[:VALIDATION_SIZE]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_images = train_images[VALIDATION_SIZE:]
        train_labels = train_labels[VALIDATION_SIZE:]

        mnist.train = DataSet(train_images, train_labels)
        mnist.validation = DataSet(validation_images, validation_labels)
        mnist.test = DataSet(test_images, test_labels)
    return mnist
