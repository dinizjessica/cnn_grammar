from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from neuralNetworkHelper import getConvQuant, hasPool, getLayerQuantity, getFCquantity
from random import *

redeExamplo1 = '(((conv*2)vazio)*3)fc*2'
redeExamplo2 = '(((conv*1)pool)*1)fc*5'

img_width, img_height = 150, 150

data_dir = '/Users/jessicadiniz/lab-vision/atividade10/pre-process-data'#pre-process-data'
train_data_dir = data_dir+'/train'
validation_data_dir = data_dir+'/validation'
nb_train_samples = 100                  # 50 bening + 50 malignant
nb_validation_samples = 60             # 30 bening + 30 malignant 
epochs = 1
batch_size = 32



train_datagen = ImageDataGenerator(
    rescale=1. / 255,
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


def getBestModel(model, train_generator):
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
    print("acc = ", history.history['acc'])
    print("val_acc = ", history.history['val_acc'])
    print("loss = ", history.history['loss'])
    print("val_loss = ", history.history['val_loss'])

    #reload best weights
    model.load_weights(best_weights_filepath)
    return model

def createModelForNeuralNetwork(networkArchitecture, input_shape):
    layerQuantity = getLayerQuantity(networkArchitecture)
    convQuantity = getConvQuant(networkArchitecture)
    pool = hasPool(networkArchitecture)
    fcQuantity = getFCquantity(networkArchitecture)

    model = Sequential()
    for layer in range(layerQuantity):
        for conv in range(convQuantity):
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        if pool:
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    for fc in range(fcQuantity):
        model.add(Dense(64))
        model.add(Activation('relu'))

    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(model.summary())
    return model

def runNeuralNetwork(networkArchitecture):
    print(networkArchitecture)
    
    return uniform(1, 10)

    # img rgb => 3 channels => depth 3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    model = createModelForNeuralNetwork(networkArchitecture, input_shape)
    model = getBestModel(model, train_generator)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    # mcp = ModelCheckpoint(model_chk_path, monitor="val_acc",
    #                       save_best_only=True, save_weights_only=False)

    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=nb_train_samples // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=nb_validation_samples // batch_size)

    # model.save_weights('try.h5')

    scores = model.evaluate_generator(validation_generator, nb_validation_samples)
    print("Accuracy = ",scores[1])
    return scores[1]

# print("acc = ", history.history['acc'])
# print("val_acc = ", history.history['val_acc'])
# print("loss = ", history.history['loss'])
# print("val_loss = ", history.history['val_loss'])

# import pdb; pdb.set_trace();

# redeExamplo2 = '(((conv*2)pool)*2)fc*1'
# runNeuralNetwork(redeExamplo2)

