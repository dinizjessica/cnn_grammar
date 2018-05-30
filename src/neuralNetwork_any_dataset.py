from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from neuralNetworkHelper import getConvQuant, hasPool, getLayerQuantity, getFCquantity
from writeFileHelper import writeLog, writeModelSummaryLog
from keras import backend as K 


#####################################
# pre-defined configuration
#####################################

data_dir = '/Users/jessicadiniz/lab-vision/atividade10/pre-process-data'
train_data_dir = data_dir+'/train'
validation_data_dir = data_dir+'/validation'
nb_train_samples = 100                  # 50 bening + 50 malignant
nb_validation_samples = 60              # 30 bening + 30 malignant 

epochs = 1
batch_size = 32
img_width, img_height = 150, 150

#####################################

def createModelForNeuralNetwork(networkArchitecture, input_shape, addDropout, addBatchNormalization):

    layerQuantity = getLayerQuantity(networkArchitecture)
    convQuantity = getConvQuant(networkArchitecture)
    pool = hasPool(networkArchitecture)
    fcQuantity = getFCquantity(networkArchitecture)

    model = Sequential()

    filterLenght = 32    
    for layer in range(layerQuantity):
        for conv in range(convQuantity):

            model.add(Conv2D(filterLenght, (3, 3), activation='relu', padding='same', input_shape=input_shape))
            if  (((conv+layer)%2)==1): filterLenght = filterLenght * 2 #duplica o tamanho do filtro a cada duas camadas convolutivas

            if addBatchNormalization:
                model.add(BatchNormalization())

        if pool:
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if addDropout:
                model.add(Dropout(0.25))
    
    model.add(Flatten())
    for fc in range(fcQuantity):
        model.add(Dense(64))
        model.add(Activation('relu'))

    if addDropout:
        model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(model.summary())
    writeModelSummaryLog(model)
    return model;

#####################################

def getBestModel(model, train_generator, validation_generator):
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    best_weights_filepath = 'best_weights.hdf5'
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    # train model
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=nb_validation_samples // batch_size,
                                  callbacks=[earlyStopping, saveBestModel])

    # history = model.fit(x_tr, y_trbatch_size=batch_size, nb_epoch=n_epochs,
    #           verbose=1, validation_data=(x_va, y_va), callbacks=[earlyStopping, saveBestModel])
    
    writeHistoryLog(history)

    #reload best weights
    model.load_weights(best_weights_filepath)
    return model

#####################################

def writeHistoryLog(history):
    # import pdb; pdb.set_trace();
    acc = "acc = "+str(history.history['acc'])
    val_acc = "val_acc = "+str(history.history['val_acc'])
    loss = "loss = "+str(history.history['loss'])
    val_loss = "val_loss = "+str(history.history['val_loss'])
    writeLog(acc)
    writeLog(val_acc)
    writeLog(loss)
    writeLog(val_loss)
    print(acc)
    print(val_acc)
    print(loss)
    print(val_loss)
    return;

#####################################

def runNeuralNetwork(networkArchitecture, addDropout=False, addBatchNormalization=False):
    writeLog("starting process for: " + networkArchitecture)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # img rgb => 3 channels => depth 3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # to free memory 
    if K.backend() == 'tensorflow':
        K.clear_session()

    model = createModelForNeuralNetwork(networkArchitecture, input_shape, addDropout, addBatchNormalization)
    model = getBestModel(model, train_generator, validation_generator)

    scores = model.evaluate_generator(validation_generator, nb_validation_samples)

    accuracy = scores[1]

    accMsg = "Accuracy on test data is: " + str(accuracy)
    print(accMsg)
    writeLog(accMsg)

    return scores[1]

#####################################

def runBests(bests, addDropout, addBatchNormalization):
    for arch in bests:
        firstRun = runNeuralNetwork(arch, addDropout, addBatchNormalization)
        # secondRun = runNeuralNetworkCifar(arch, addDropout, addBatchNormalization)
        # thirdRun = runNeuralNetworkCifar(arch, addDropout, addBatchNormalization)
        # result = (firstRun + secondRun + thirdRun)/3

        writeLog(arch + " => addDropout: " + str(addDropout) + "; addBatchNormalization: " + str(addBatchNormalization))
        writeLog(arch + " => firstRun: " + str(firstRun))
        # writeLog(arch + " => firstRun: " + str(firstRun) + "; secondRun: " + str(secondRun) + "; thirdRun: " + str(thirdRun))
        # writeLog(arch + " => media: " + result)



############################################
# Testes com dropout e batch normalization
############################################
# bests = ['(((conv*2)pool)*3)fc*2', '(((conv*2)pool)*3)fc*1', '(((conv*2)pool)*3)fc*0']
bests = ['(((conv*1)pool)*1)fc*1']

runBests(bests, True, True)



