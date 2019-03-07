from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, Callback

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from neuralNetworkHelper import getNumberOfClasses, getQuantityOfFilesInAFolder
from writeFileHelper import writeLog
from grammar_helper import createModelForNeuralNetwork, getLearningOptFromNetwork

import gc
import time
import os


def runNeuralNetwork(networkArchitecture, data_dir, epochs=100, batch_size=32, img_width=120, img_height=120):
    writeLog("starting neuralNetwork_assuncao_outrasBases process for: " + networkArchitecture)

    #####################################
    # basic configuration - dont change
    #####################################
    input_shape = getInputShape(img_width, img_height)
    train_data_dir = data_dir+'/train'
    validation_data_dir = data_dir+'/validation'
    num_classes = getNumberOfClasses(train_data_dir)
    nb_train_samples =  getQuantityOfFilesInAFolder(train_data_dir)             # dividido igualmente entre as classes
    nb_validation_samples = getQuantityOfFilesInAFolder(validation_data_dir)    # dividido igualmente entre as classes
    # print("nb_train_samples: " + str(nb_train_samples) + "; nb_validation_samples: " + str(nb_validation_samples))
    # print("num_classes: "+str(num_classes))
    best_weights_filepath = 'best_weights.hdf5'
    #####################################


    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       fill_mode='nearest',
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    # class_mode='categorical' ???

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # to free memory 
    if K.backend() == 'tensorflow':
        K.clear_session()

    model = createModelForNeuralNetwork(networkArchitecture, input_shape, numClasses=num_classes)

    optimizer = getLearningOptFromNetwork(networkArchitecture)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # remove saved best weight if it exists
    if os.path.exists(best_weights_filepath):
        os.remove(best_weights_filepath)

    # alterar o learning rate em determinados pontos
    # lrate = LearningRateScheduler(step_decay)
    early_stopping = EarlyStopping(monitor='val_acc', patience=7, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    callbacks_list = [early_stopping, saveBestModel]  # [lrate]

    # model training
    start = time.time()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=nb_validation_samples // batch_size,
                                  callbacks=callbacks_list #[earlyStopping, saveBestModel]
                                  )

    #reload best weights
    model.load_weights(best_weights_filepath)

    writeLog("[INFO] evaluating network...")
    scores = model.evaluate_generator(validation_generator, nb_validation_samples)

    end = time.time()
    logRunTime(start, end)
    logHistoryLog(history)

    # writeLog("[INFO] evaluating network...")
    # predictions = model.predict(test_features_val, batch_size=batch_size)

    accuracy = scores[1]
    writeLog("Accuracy on test data is: " + str(accuracy))

    memoryClean(train_generator,validation_generator,train_datagen,test_datagen,model)

    return accuracy

#####################################
# 0.01 -> do 0 ate o 5th
# 0.1 -> do 6th ate o 250th 
# 0.01 -> do 251st ate o 375th
# 0.001 -> do 376th ate o 400th

def step_decay(epoch):
    if (epoch <= 5 or (epoch > 250 and epoch <= 375)):
        return 0.01
    elif epoch > 5 and epoch <= 250:
        return 0.1
    elif epoch > 375:
        return 0.001
#####################################
def getInputShape(img_width, img_height):
    # img rgb => 3 channels => depth 3
    if K.image_data_format() == 'channels_first':
        return (3, img_width, img_height)
    else:
        return (img_width, img_height, 3)
#####################################
def logRunTime(startTime, endTime):
    diff = int(endTime - startTime  )
    minutes, seconds = diff // 60, diff % 60
    writeLog("Model took minutes to train " + str(minutes) + ':' + str(seconds).zfill(2))
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
def memoryClean(train_generator,validation_generator,train_datagen,test_datagen,model):
    del train_generator
    del validation_generator
    del train_datagen
    del test_datagen
    del model
    gc.collect()
    return;
#####################################



############################################
# Testes
############################################
# writeLog('####################################')
# writeLog('STARTING')
# writeLog('####################################')
# data_dir = '/Users/jdiniz/Documents/android/visao computacional/mestrado/dataset2test'
# epochs = 5
# batch_size = 32
# img_width, img_height = 150, 150
# img_width, img_height = 120, 120

# ind = '(layer:conv num-filters:128 filter-shape:2 stride:3 padding:same act:sigmoid bias:False batch-normalisation:True merge-input:False) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:fc act:sigmoid num-units:512 bias:False) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.001)'
# ind = '(layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
# ind = '(layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
# runNeuralNetwork(ind, data_dir, epochs, batch_size)

