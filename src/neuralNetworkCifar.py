import time
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10

from neuralNetworkHelper import getConvQuant, hasPool, getLayerQuantity, getFCquantity
from writeFileHelper import writeLog, writeModelSummaryLog
from keras import backend as K 


def createModelForNeuralNetwork(networkArchitecture, 
                                input_shape, 
                                num_classes, 
                                addDropout, 
                                addBatchNormalization):

    layerQuantity = getLayerQuantity(networkArchitecture)
    convQuantity = getConvQuant(networkArchitecture)
    pool = hasPool(networkArchitecture)
    fcQuantity = getFCquantity(networkArchitecture)

    model = Sequential()
    filterLenght = 32
    for layer in range(layerQuantity):
        for conv in range(convQuantity):
            #if (conv > 0) and (((conv+layer)%2)==0): filterLenght = filterLenght * 2 #duplica o tamanho do filtro a cada duas camadas convolutivas
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
        model.add(Dense(128))
        model.add(Activation('relu'))

    if addDropout:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    writeModelSummaryLog(model)
    return model;

#####################################

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

#####################################

def runNeuralNetworkCifar(networkArchitecture, 
                          addDropout=False, 
                          addBatchNormalization=False,
                          useDataAugmentation=False):

    writeLog("starting process for: " + networkArchitecture)
    # load cifar data
    (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
    # split the data on test (test_features_test) and validation (test_features_val)
    test_features_val, test_features_test, test_labels_val, test_labels_test = train_test_split(test_features, test_labels, test_size=0.2, random_state=42) 

    num_classes = len(np.unique(train_labels))

    train_features = train_features.astype("float")/255.0
    test_features_val = test_features_val.astype("float")/255.0
    test_features_test = test_features_test.astype("float")/255.0

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels_val = lb.transform(test_labels_val)
    test_labels_test = lb.transform(test_labels_test)

    input_shape=(32,32,3)
    batch_size = 128
    epochs = 10

    # to free memory 
    if K.backend() == 'tensorflow':
        K.clear_session()
            

    # Create the model according to the networkArchitecture
    model = createModelForNeuralNetwork(networkArchitecture, input_shape, num_classes, addDropout, addBatchNormalization)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    start = time.time()

    if (useDataAugmentation):
        # adding data augmentation
        datagen = ImageDataGenerator(zoom_range=0.2,
                                     horizontal_flip=True)

        model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = batch_size),
                                         steps_per_epoch = train_features.shape[0]//batch_size, 
                                         epochs = epochs, 
                                         validation_data = (test_features_val, test_labels_val), 
                                         verbose=0)

    else:
        model_info = model.fit(train_features, train_labels, 
                               batch_size=batch_size, 
                               epochs=epochs, 
                               validation_data = (test_features_val, test_labels_val), 
                               verbose=0)

    end = time.time()
    timeMsg = "Model took seconds to train " + str((end - start))
    print(timeMsg)
    writeLog(timeMsg)
    # compute test accuracy
    accuracyValue = accuracy(test_features_test, test_labels_test, model)
    accMsg = "Accuracy on test data is: " + str(accuracyValue)
    print(accMsg)
    writeLog(accMsg)
    return accuracyValue;

#####################################

def runBests(bests, addDropout, addBatchNormalization, useDataAugmentation):
    for arch in bests:
        firstRun = runNeuralNetworkCifar(arch, addDropout, addBatchNormalization, useDataAugmentation)
        secondRun = runNeuralNetworkCifar(arch, addDropout, addBatchNormalization, useDataAugmentation)
        thirdRun = runNeuralNetworkCifar(arch, addDropout, addBatchNormalization, useDataAugmentation)
        result = (firstRun + secondRun + thirdRun)/3

        writeLog(arch + " => addDropout: " + str(addDropout) + "; addBatchNormalization: " + str(addBatchNormalization) + "; useDataAugmentation: " + str(useDataAugmentation))
        writeLog(arch + " => firstRun: " + str(firstRun) + "; secondRun: " + str(secondRun) + "; thirdRun: " + str(thirdRun))
        writeLog(arch + " => media: " + result)



############################################
# Testes com dropout e batch normalization
############################################
bests = ['(((conv*2)pool)*3)fc*2', '(((conv*2)pool)*3)fc*1', '(((conv*2)pool)*3)fc*0']

# with data augmentation
runBests(bests, True, True, False)
# runBests(bests, True, False, True)
# runBests(bests, False, True, True)
# runBests(bests, False, False, True)

# # without data augmentation
# runBests(bests, True, True, False)
# runBests(bests, True, False, False)
# runBests(bests, False, True, False)
# runBests(bests, False, False, False)



# (((conv*2)pool)*3)fc*2 => Accuracy on test data is: 79.3
# (((conv*2)pool)*3)fc*1 => Accuracy on test data is: 79.05
# (((conv*2)pool)*3)fc*1 => Accuracy on test data is: 79.35
# (((conv*2)pool)*3)fc*1 => Accuracy on test data is: 79.25
# (((conv*2)pool)*3)fc*1 => Accuracy on test data is: 79.10000000000001
# (((conv*2)pool)*3)fc*0 => Accuracy on test data is: 78.8



