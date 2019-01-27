import glob
import time
import gc

import nibabel as nib
import numpy as np
import pandas as pd

from keras.utils import to_categorical

from grammar_helper import createModelForNeuralNetwork, getLearningOptFromNetwork
from writeFileHelper import writeLog

##################################################
# pre-defined configuration
##################################################
# data path
data_folder_path = '/media/gpin/datasets/AMBAC/acerta_whole/classification/*.nii'
data_paths = glob.glob(data_folder_path) # list of each nii path as string

# mask path
data_mask_path = '/media/gpin/datasets/AMBAC/acerta_whole/mask_group_whole.nii'

# labels
data_classification_path = '/media/gpin/datasets/AMBAC/acerta_whole/y.csv'
labels = pd.read_csv(data_classification_path, sep=",")

input_shape = (60, 73, 61)

# Specify number of filters per layer
filters = 16

# How many iterations of the model fitting should be computed
epochs = 100  # Increase this value for better results (i.e., more training)

# How many elements (volumes) should be considered at once for the updating of the weights?
batch_size = 16   # Increasing this value might speed up fitting

# percentage of images from the set to train
training_images_percentage = 0.8


##################################################
# methods
##################################################
def load_data_from_nii_files(file_paths):
    loaded_images = []
    for img_path in sorted(file_paths):
        image = nib.load(img_path).get_data()
        loaded_images.append(image)
    return loaded_images


def load_mask_file(data_mask_path):
    mask_file = nib.load(data_mask_path).get_data()
    print(mask_file.shape)
    # Verify if the whole brain mask is boolean
    mask_file = mask_file.astype(bool)
    return mask_file


def apply_mask_to_data_files(imgs_to_apply_mask, mask):
    # Maintaining the original structure of the image file
    masked_imgs =[]
    for img in imgs_to_apply_mask:
        masked_imgs.append(np.array(img * mask))

    masked_imgs_converted_to_array = np.asarray(masked_imgs)
    return masked_imgs_converted_to_array


def apply_zscore(images):
    # rescale the data using zscore standardization technique
    mean = np.mean(images)
    standart_deviation = np.std(images)
    zscored_imgs = (images - mean) / standart_deviation
    return zscored_imgs


def get_shuffled_index_list(index_list_size):
    # Create list of indices and shuffle them
    indices = np.arange(index_list_size)
    np.random.shuffle(indices)
    return indices


def split_train_and_test_index_set(indices, img_quantity):
    training_img_quantity = int(training_images_percentage * img_quantity)
    train_indices = indices[:training_img_quantity]
    test_indices = indices[training_img_quantity:]
    return train_indices, test_indices


def split_data_into_training_and_test_sets(images, train_indices, test_indices):
    X_train = images[train_indices, ...]
    X_test = images[test_indices, ...]
    return X_train, X_test


def create_outcome_variables(label_list, train_indices, test_indices):
    target = label_list['Label']
    y_train = target[train_indices]
    y_test  = target[test_indices]
    return y_train, y_test


def reshape_outcome_variables_to_categorical(y_train_as_3D, y_test_as_3D):
    # We need to reformat the shape of our outcome variables,
    # y_train and y_test, because Keras needs the labels as a 2D array.
    y_train = to_categorical(y_train_as_3D)
    y_test = to_categorical(y_test_as_3D)
    return y_train, y_test


def memory_clean(model):
    del model
    gc.collect()
    return


def log_history(history):
    acc = "acc = "+str(history.history['acc'])
    val_acc = "val_acc = "+str(history.history['val_acc'])
    loss = "loss = "+str(history.history['loss'])
    val_loss = "val_loss = "+str(history.history['val_loss'])
    writeLog(acc)
    writeLog(val_acc)
    writeLog(loss)
    writeLog(val_loss)
    return


def log_execution_time(start_time, end_time):
    diff = int(end_time - start_time)
    minutes, seconds = diff // 60, diff % 60
    writeLog("Model took minutes to train " + str(minutes) + ':' + str(seconds).zfill(2))
    return


##################################################
# process
##################################################

def runNeuralNetwork(networkArchitecture):
    writeLog("starting neuralNetwork_assuncao process for: " + networkArchitecture)

    # load data
    data_all = load_data_from_nii_files(data_paths)
    mask = load_mask_file(data_mask_path)

    # preprocessing
    masked_imgs = apply_mask_to_data_files(data_all, mask)
    images = apply_zscore(masked_imgs)

    img_quantity = images.shape[0]
    indices = get_shuffled_index_list(img_quantity)

    train_indices, test_indices = split_train_and_test_index_set(indices, img_quantity)
    X_train, X_test = split_data_into_training_and_test_sets(images, train_indices, test_indices)

    y_train_as_3D, y_test_as_3D = create_outcome_variables(labels, train_indices, test_indices)
    y_train, y_test = reshape_outcome_variables_to_categorical(y_train_as_3D, y_test_as_3D)

    data_shape = tuple(X_train.shape[1:])

    model = createModelForNeuralNetwork(networkArchitecture, data_shape)

    optimizer = getLearningOptFromNetwork(networkArchitecture)
    loss = 'categorical_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # train the model
    start = time.time()

    # Let's test the model:
    model_info = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    evaluation = model.evaluate(X_test, y_test)

    end = time.time()
    log_execution_time(start, end)
    log_history(model_info)

    loss_result = (evaluation[0])
    accuracy_result = (evaluation[1] * 100)

    writeLog('Loss in Test set:        %.02f' % loss_result)
    writeLog('Accuracy in Test set:    %.02f' % accuracy_result)

    memory_clean(model)

    return accuracy_result


##################################################
# test
##################################################
ind = '(layer:conv num-filters:16 filter-shape:3 stride:1 padding:valid act:relu bias:False batch-normalisation:True merge-input:False) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:32 filter-shape:3 stride:1 padding:valid act:relu bias:True batch-normalisation:True merge-input:False) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:64 filter-shape:3 stride:1 padding:same act:relu bias:True batch-normalisation:True merge-input:False) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:fc act:relu num-units:256 bias:True) (layer:fc act:relu num-units:512 bias:True) (layer:fc act:relu num-units:64 bias:True) (layer:fc act:softmax num-units:2 bias:True) (learning:gradient-descent learning-rate:0.01)'

runNeuralNetwork(ind)