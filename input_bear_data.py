# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 17:25:21 2016

@author: David_Zhang
"""
import numpy
def extract_images(data):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  data = data.reshape(data.shape[0], data.shape[1])
  return data
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes), dtype = numpy.uint8)
  #our labels have label 10, so we minus 1
  labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
  return labels_one_hot
class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 2000
    else:
      assert images.shape[0] == labels.shape[0]
      self._num_examples = images.shape[0]
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(2400)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
def read_data_sets(images, labels, test_images, test_labels, fake_data=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets
    #here we using 1500 training datas, 200 validation datas and 300 test datas
  TRAIN_SIZE = 15000
  Train_Images = extract_images(images)
  Train_Labels = dense_to_one_hot(labels)
  test_images = extract_images(test_images)
  test_labels = dense_to_one_hot(test_labels)
  train_images = Train_Images[:TRAIN_SIZE]
  train_labels = Train_Labels[:TRAIN_SIZE]
  validation_images = Train_Images[TRAIN_SIZE:]
  validation_labels = Train_Labels[TRAIN_SIZE:]
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets