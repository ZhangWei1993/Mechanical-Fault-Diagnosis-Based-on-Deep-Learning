# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 17:21:17 2016
this code disorders the input images and their corresponding labels 
@author: David_Zhang
"""
import scipy.io as sio 
import numpy as np
image_path = 'images.mat'
image_D = sio.loadmat(image_path)
label_path = 'labels.mat'  
label_D = sio.loadmat(label_path)
images = image_D['images']
labels = label_D['labels']
test_image_path = 'test_images.mat'
test_image_D = sio.loadmat(test_image_path)
test_label_path = 'test_labels.mat'  
test_label_D = sio.loadmat(test_label_path)
images = image_D['images']
labels = label_D['labels']
test_images = test_image_D['test_images']
test_labels = test_label_D['test_labels']
def disorder_images(images, labels):
    assert images.shape[0] == labels.shape[0]
    num_examples = images.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    return images,labels
images,labels =  disorder_images(images, labels)
test_images,test_labels = disorder_images(test_images, test_labels)
np.save("images.npy",images)
np.save("labels.npy",labels)
np.save("test_images.npy",test_images)
np.save("test_labels.npy",test_labels)