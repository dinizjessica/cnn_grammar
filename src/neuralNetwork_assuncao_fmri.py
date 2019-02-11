import glob
import time
import gc

import nibabel as nib
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from keras import backend as k
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
# from keras.optimizers import Adam, SGD, RMSprop

from grammar_helper import createModelForNeuralNetwork, getLearningOptFromNetwork
from writeFileHelper import writeLog

from os import path

##################################################
# pre-defined configuration
##################################################

# data set folder path
fmri_dataset_path = path.abspath(path.join(__file__, "../../datasets/fmri/"))

# data path
# data_folder_path = '/media/gpin/datasets/AMBAC/data_aug/*.nii'
data_folder_path = fmri_dataset_path+'/whole/*.nii'
data_paths = glob.glob(data_folder_path) # list of each nii path as string

# mask path
#data_mask_path = '/media/gpin/datasets/AMBAC/acerta_whole/mask_group_whole.nii'

# labels
# data_classification_path = '/media/gpin/datasets/AMBAC/y_aug_backup.csv'
data_classification_path = fmri_dataset_path+'/y_aug_backup.csv'
labels = pd.read_csv(data_classification_path, sep=";")

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
    
def transform_list_to_array(imgs_to_transform):
    imgs_transformed, images = [], []
    for img in imgs_to_transform:
        imgs_transformed.append(np.array(img))
        #print(img.shape)
    images = np.asarray(imgs_transformed)
    return images


#def load_mask_file(data_mask_path):
#    mask_file = nib.load(data_mask_path).get_data()
#    print(mask_file.shape)
    # Verify if the whole brain mask is boolean
#    mask_file = mask_file.astype(bool)
#    return mask_file


#def apply_mask_to_data_files(imgs_to_apply_mask, mask):
    # Maintaining the original structure of the image file
#    masked_imgs =[]
#    for img in imgs_to_apply_mask:
#        masked_imgs.append(np.array(img * mask))

#    masked_imgs_converted_to_array = np.asarray(masked_imgs)
#    return masked_imgs_converted_to_array


#def apply_zscore(images):
    # rescale the data using zscore standardization technique
#    mean = np.mean(images)
#    standart_deviation = np.std(images)
#    zscored_imgs = (images - mean) / standart_deviation
#    return zscored_imgs

def get_shuffled_index_list(index_list_size):
    # Create list of indices and shuffle them, but maintain balanced classes
    indexes = np.arange(index_list_size)
    indexes_dis = indexes[:144]
    indexes_con = indexes[144:]
    np.random.shuffle(indexes_dis)
    np.random.shuffle(indexes_con)
    return indexes_dis, indexes_con


def split_train_and_test_index_set(indexes_dis, indexes_con, img_quantity):
    training_img_quantity = int(training_images_percentage * img_quantity)
    train_indexes_dis, train_indexes_con = indexes_dis[:training_img_quantity], indexes_con[:training_img_quantity]
    test_indexes_dis, test_indexes_con = indexes_dis[training_img_quantity:], indexes_con[training_img_quantity:]
    print(test_indexes_dis, test_indexes_con)
    train_indexes = np.concatenate((train_indexes_dis, train_indexes_con), axis=None)
    test_indexes = np.concatenate((test_indexes_dis, test_indexes_con), axis=None)
    print(train_indexes, test_indexes)
    return train_indexes, test_indexes


def split_data_into_training_and_test_sets(images, train_indexes, test_indexes):
    X_train = images[train_indexes, ...]
    X_test = images[test_indexes, ...]
    return X_train, X_test


def create_outcome_variables(label_list, train_indexes, test_indexes):
    target = label_list['Labels']
    y_train = target[train_indexes]
    y_test  = target[test_indexes]
    return y_train, y_test


def reshape_outcome_variables_to_categorical(y_train_as_3D, y_test_as_3D):
    # We need to reformat the shape of our outcome variables,
    # y_train and y_test, because Keras needs the labels as a 2D array.
    y_train = to_categorical(y_train_as_3D)
    y_test = to_categorical(y_test_as_3D)
    return y_train, y_test


def memory_clean(model):
    del model
    k.clear_session()
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


# def show_activation(layer_name, model, X_train):
#     # Aggregate the layers
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])
#
#     layer_output = layer_dict[layer_name].output
#
#     fn = k.function([model.input], [layer_output])
#
#     inp = X_train[0:1]
#
#     this_hidden = fn([inp])[0]
#
#     # plot the activations, 8 filters per row
#     plt.figure(figsize=(16, 8))
#     nFilters = this_hidden.shape[-1]
#     nColumn = 8 if nFilters >= 8 else nFilters
#     for i in range(nFilters):
#         plt.subplot(nFilters / nColumn, nColumn, i + 1)
#         plt.imshow(this_hidden[0, :, :, i], cmap='magma', interpolation='nearest')
#         plt.axis('off')
#     return


# 0.01 -> do 0 ate o 5th
# 0.1 -> do 6th ate o 250th
# 0.01 -> do 251st ate o 375th
# 0.001 -> do 376th ate o 400th
def step_decay(epoch):
    if (epoch <= 5 or (epoch > 250 and epoch <= 375)):
        return 0.00001
    elif epoch > 5 and epoch <= 80:
        return 0.0001
    elif epoch > 80:
        return 0.001

##################################################
# process
##################################################

def runNeuralNetwork(networkArchitecture, use_step_decay=False):
    writeLog("starting neuralNetwork_assuncao process for: " + networkArchitecture)

    # load data
    data_all = load_data_from_nii_files(data_paths)
    images = transform_list_to_array(data_all)
    #print(images.shape)
    #mask = load_mask_file(data_mask_path)

    # preprocessing
    #masked_imgs = apply_mask_to_data_files(data_all, mask)
    #images = apply_zscore(masked_imgs)

    img_quantity = images.shape[0]
    indexes_dis, indexes_con = get_shuffled_index_list(img_quantity)
    img_quantity_balanced = indexes_dis.shape[0]
    train_indexes, test_indexes = split_train_and_test_index_set(indexes_dis, indexes_con, img_quantity_balanced)
    X_train, X_test = split_data_into_training_and_test_sets(images, train_indexes, test_indexes)

    y_train_as_3D, y_test_as_3D = create_outcome_variables(labels, train_indexes, test_indexes)
    y_train, y_test = reshape_outcome_variables_to_categorical(y_train_as_3D, y_test_as_3D)

    data_shape = tuple(X_train.shape[1:])
    k.clear_session()
    model = createModelForNeuralNetwork(networkArchitecture, data_shape)

    # learning_rate = 1e-5
    # adam = Adam(lr=learning_rate)
    optimizer = getLearningOptFromNetwork(networkArchitecture)
    loss = 'binary_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # import pdb; pdb.set_trace() # debug

    # train the model
    start = time.time()

    # Let's test the model:
    if (use_step_decay):
        # alterar o learning rate em determinados pontos
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]

        model_info = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    else:
        model_info = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    evaluation = model.evaluate(X_test, y_test)

    end = time.time()
    log_execution_time(start, end)
    log_history(model_info)

    loss_result = (evaluation[0])
    accuracy_result = (evaluation[1] * 100)

    writeLog('Loss in Test set:        %.02f' % loss_result)
    writeLog('Accuracy in Test set:    %.02f' % accuracy_result)
    
    #show_activation('conv2d_1', model, X_train)

    memory_clean(model)

    return accuracy_result


##################################################
# test
##################################################
ind = '(layer:conv num-filters:32 filter-shape:4 stride:1 padding:same act:linear bias:False batch-normalisation:False merge-input:True) (layer:fc act:relu num-units:512 bias:True layer:fc act:linear num-units:128 bias:False) (layer:fc act:softmax num-units:2 bias:True) (learning:adam learning-rate:0.0001)'

runNeuralNetwork(ind)
