import os
import glob

from nilearn import plotting
#%matplotlib inline
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(style="darkgrid")

from nilearn.image import mean_img
from nilearn.plotting import plot_anat

import nibabel as nib

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop

from keras import backend as k

from datetime import datetime

from grammar_helper import createModelForNeuralNetwork, getLearningOptFromNetwork


# data path
data_folder_path = '/media/gpin/datasets/AMBAC/acerta_whole/classification/*.nii'
data_paths = glob.glob(data_folder_path) # list of each nii path as string

# mask path
data_mask_path = '/media/gpin/datasets/AMBAC/acerta_whole/mask_group_whole.nii'

#Get labels
data_classification_path = '/media/gpin/datasets/AMBAC/acerta_whole/y.csv'
labels = pd.read_csv(data_classification_path, sep=",")
target = labels['Label']

x_coord = 60
y_coord = 73
z_coord = 61

# Specify number of filtersper layer
filters = 16

# Specify shape of convolution kernel
kernel_size = (3, 3)

# Specify number of output categories
#n_classes = 2

ind = '(layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
networkArchitecture = ind



def load_data_from_nii_files(file_paths):
    loaded_images = []
    for img_path in sorted(file_paths):
        # print(data_file)
        image = nib.load(img_path).get_data()
        # print(type(data))
        # data = data.squeeze()
        # print(data.shape)
        loaded_images.append(image)
        # data_all.append(data.reshape(x_coord, y_coord, z_coord))
    return loaded_images

data_all = load_data_from_nii_files(data_paths)

# Load mask file
def load_mask_file(data_mask_path):
    mask_file = nib.load(data_mask_path).get_data()
    print(mask_file.shape)
    # Verify if the whole brain mask is boolean
    mask_file = mask_file.astype(bool)
    return mask_file

mask = load_mask_file(data_mask_path)

#print(len(data_all))

#Preprocessing
#Apply the mask to all the data files
#Maintaining the original structure of the image file
def image_preprocessing(imgs_to_apply_mask, mask):
    masked_imgs =[]
    for img in imgs_to_apply_mask:
        masked_imgs.append(np.array(img * mask))

    images = np.asarray(masked_imgs)
    #print(images.shape)
    return images

images = image_preprocessing(data_all, mask)

#Next, rescale the data with using max-min normalisation technique
def apply_zscore(images):
    mean = np.mean(images)
    std = np.std(images)
    #print(ma, mi)
    zscored_imgs = (images - mean) / (std)
    return zscored_imgs

images = apply_zscore(images)

#Let's verify the minimum and maximum value of the data which should be 0.0 and 1.0 after rescaling it!
#print(np.min(images), np.max(images))



# Create list of indices and shuffle them
N = images.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

def get_dataset_index_split(indices):
    #  Cut the dataset at 80% to create the training and test set
    training_images_percentage = 0.8
    training_img_quantity = int(training_images_percentage * N)
    train_indices = indices[:training_img_quantity]
    test_indices = indices[training_img_quantity:]
    return train_indices, test_indices

train_indices, test_indices = get_dataset_index_split(indices)

def split_the_data_into_training_and_test_sets(images, train_indices, test_indices):
    X_train = images[train_indices, ...]
    X_test = images[test_indices, ...]

    print(X_train.shape, X_test.shape)
    return X_train, X_test

X_train, X_test = split_the_data_into_training_and_test_sets(images, train_indices, test_indices)

# Create outcome variable
y_train = target[train_indices]
y_test  = target[test_indices]
print(y_train.shape)
print(y_test)


# Create a sequential model

# Get shape of input data
data_shape = tuple(X_train.shape[1:])
print(data_shape)



model = createModelForNeuralNetwork(networkArchitecture, data_shape)

k.clear_session()
# model = Sequential()
#
# model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=data_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
#
# model.add(Conv2D(filters * 2, kernel_size, activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
#
# model.add(Conv2D(filters * 4, kernel_size, activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
#
# model.add(Flatten())
#
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(1, activation='sigmoid'))

#optimizer
# learning_rate = 1e-5
# adam = Adam(lr=learning_rate)
# sgd = SGD(lr=learning_rate)
#
optimizer = getLearningOptFromNetwork(networkArchitecture)
loss = 'binary_crossentropy'

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# model.compile(loss=loss,
#               optimizer=adam, # swap out for sgd
#               metrics=['accuracy'])

model.summary()



#Fitting the Model

#The next step is now of course to fit our model to the training data.
#In our case we have two parameters that we can work with:

#First: How many iterations of the model fitting should be computed
nEpochs = 100  # Increase this value for better results (i.e., more training)

#Second: How many elements (volumes) should be considered at once for the updating of the weights?

batch_size = 16   # Increasing this value might speed up fitting




# TensorBoard callback
#LOG_DIRECTORY_ROOT = 'logdir'
#now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
#tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
#callbacks = [tensorboard]



# Let's test the model:

fit = model.fit(X_train, y_train, epochs=nEpochs, batch_size=batch_size)


#Evaluating the model
evaluation = model.evaluate(X_test, y_test)
print('Loss in Test set:        %.02f' % (evaluation[0]))
print('Accuracy in Test set:    %.02f' % (evaluation[1] * 100)) 
