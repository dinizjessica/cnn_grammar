import time
# import matplotlib.pyplot as plt

# import matplotlib as mpl
# mpl.use('TkAgg')
# from matplotlib import pyplot as plt



import numpy as np
# % matplotlib inline
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
# from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split


def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
test_features_val, test_features_test, test_labels_val, test_labels_test = train_test_split(test_features, test_labels, test_size=0.33, random_state=42)

num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))

# import pdb; pdb.set_trace()
train_features = train_features.astype("float")/255.0
# test_features = test_features.astype("float")/255.0
test_features_val = test_features_val.astype("float")/255.0
test_features_test = test_features_test.astype("float")/255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
# test_labels = lb.transform(test_labels)
test_labels_val = lb.transform(test_labels_val)
test_labels_test = lb.transform(test_labels_test)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Define the model
model = Sequential()
model.add(Conv2D(48, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
# start = time.time()
# model_info = model.fit(train_features, train_labels, 
#                        batch_size=128, nb_epoch=200, 
#                        validation_data = (test_features, test_labels), 
#                        verbose=0)
# end = time.time()

# print("Model took %0.2f seconds to train", (end - start))
# # compute test accuracy
# print("Accuracy on test data is: %0.2f", accuracy(test_features, test_labels, model))

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2, 
                             horizontal_flip=True)


# train the model
start = time.time()

batch_size = 128
model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = batch_size),
                                 steps_per_epoch = train_features.shape[0]//batch_size, 
                                 epochs = 2, 
                                 validation_data = (test_features_val, test_labels_val), 
                                 verbose=0)


end = time.time()
print("Model took seconds to train ",(end - start))
# compute test accuracy
print("Accuracy on test data is: ",accuracy(test_features_test, test_labels_test, model))
