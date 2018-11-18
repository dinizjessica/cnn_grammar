import time
import numpy as np
import gc

from keras.callbacks import LearningRateScheduler, Callback

from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10

from writeFileHelper import writeLog, writeArray
from keras import backend as K 

from grammar_helper import createModelForNeuralNetwork, getLearningOptFromNetwork, step_decay

#####################################
# pre-defined configuration
#####################################

input_shape=(32,32,3)

#####################################

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

# #####################################

# # 0.01 -> do 0 ate o 5th
# # 0.1 -> do 6th ate o 250th 
# # 0.01 -> do 251st ate o 375th
# # 0.001 -> do 376th ate o 400th

# def step_decay(epoch):
#     if (epoch <= 5 or (epoch > 250 and epoch <= 375)):
#         return 0.01
#     elif epoch > 5 and epoch <= 250:
#         return 0.1
#     elif epoch > 375:
#         return 0.001

# #####################################

def runNeuralNetwork(networkArchitecture, epochs=400, batch_size=128, useDataAugmentation=False):
    writeLog("starting neuralNetwork_assuncao process for: " + networkArchitecture)
    # load cifar data
    (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
    # split the data on test (test_features_test) and validation (test_features_val)
    test_features_val, test_features_test, test_labels_val, test_labels_test = train_test_split(test_features, test_labels, test_size=0.2, random_state=42) 

    train_features = train_features.astype("float")/255.0
    test_features_val = test_features_val.astype("float")/255.0
    test_features_test = test_features_test.astype("float")/255.0

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels_val = lb.transform(test_labels_val)
    test_labels_test = lb.transform(test_labels_test)

    # to free memory 
    if K.backend() == 'tensorflow':
        K.clear_session() 

    # Create the model according to the networkArchitecture
    model = createModelForNeuralNetwork(networkArchitecture, input_shape)

    optimizer = getLearningOptFromNetwork(networkArchitecture)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    start = time.time()

    # alterar o learning rate em determinados pontos
    lrate = LearningRateScheduler(step_decay) 
    callbacks_list = [lrate]

    if (useDataAugmentation):
        # adding data augmentation
        datagen = ImageDataGenerator(zoom_range=0.2,
                                     horizontal_flip=True)

        model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = batch_size),
                                         steps_per_epoch = train_features.shape[0]//batch_size, 
                                         epochs = epochs, 
                                         validation_data = (test_features_val, test_labels_val), 
                                         callbacks=callbacks_list,
                                         verbose=0)

    else:
        model_info = model.fit(train_features, train_labels, 
                               batch_size=batch_size, 
                               epochs=epochs, 
                               validation_data = (test_features_val, test_labels_val), 
                               callbacks=callbacks_list,
                               verbose=0)
    
    writeLog("[INFO] evaluating network...")
    # predictions = model.predict(test_features_val, batch_size=batch_size)

    end = time.time()

    logRunTime(start,end)
    logHistoryLog(model_info)

    # compute test accuracy
    accuracyValue = accuracy(test_features_test, test_labels_test, model)
    writeLog("Accuracy on test data is: " + str(accuracyValue))

    memoryClean(model)

    return accuracyValue;

#####################################
def memoryClean(model):
    del model
    gc.collect()
    return;
#####################################
def logHistoryLog(history):
    acc = "acc = "+str(history.history['acc'])
    val_acc = "val_acc = "+str(history.history['val_acc'])
    loss = "loss = "+str(history.history['loss'])
    val_loss = "val_loss = "+str(history.history['val_loss'])
    writeLog(acc)
    writeLog(val_acc)
    writeLog(loss)
    writeLog(val_loss)
    return;
#####################################
def logRunTime(startTime, endTime):
    diff = int(endTime - startTime  )
    minutes, seconds = diff // 60, diff % 60
    writeLog("Model took minutes to train " + str(minutes) + ':' + str(seconds).zfill(2))
    return;
#####################################

####################################
# tests
####################################

# writeLog('####################################')
# writeLog('STARTING')
# writeLog('####################################')
# ind = '(layer:conv num-filters:128 filter-shape:2 stride:3 padding:same act:sigmoid bias:False batch-normalisation:True merge-input:False) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:fc act:sigmoid num-units:512 bias:False) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.001)'
# ind = '(layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
# ind = '(layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
# runNeuralNetwork(ind)
