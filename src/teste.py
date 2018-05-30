import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10
# from neuralNetworkHelper import getConvQuant, hasPool, getLayerQuantity, getFCquantity
# from writeFileHelper import writeLog, writeModelSummaryLog
from keras import backend as K 



# individuo = '(layer:pool-max [kernel-size,int,1,1,5] [stride,int,1,1,3] padding:valid)(layer:conv[num-filters,int,1,32,256][filter-shape,int,1,1,5][stride,int,1,1,3] padding:valid act:sigmoid bias:False batch-normalisation:True merge-input:True)(layer:pool-max [kernel-size,int,1,1,5] [stride,int,1,1,3] padding:valid) (layer:fc act:relu [num-units,int,1,128,2048] bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent [lr,float,1,0.0001,0.1])'
# individuo = '(layer:conv [num-filters,int,1,32,256][filter-shape,int,1,1,5][stride,int,1,1,3] padding:same act:sigmoid bias:True batch-normalisation:True merge-input:False) (layer:fc act:relu [num-units,int,1,128,2048] bias:False) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent [lr,float,1,0.0001,0.1])'
individuo = '(layer:conv [num-filters,int,1,32,256][filter-shape,int,1,1,5][stride,int,1,1,3] padding:same act:relu bias:True batch-normalisation:True merge-input:True) (layer:pool-max [kernel-size,int,1,1,5] [stride,int,1,1,3] padding:valid) (layer:pool-max [kernel-size,int,1,1,5] [stride,int,1,1,3] padding:valid) (layer:pool-max [kernel-size,int,1,1,5] [stride,int,1,1,3] padding:valid) (layer:fc act:sigmoid [num-units,int,1,128,2048] bias:False) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent [lr,float,1,0.0001,0.1])'

individuo = individuo.replace(')','')

rede_str = individuo.split('(')
rede_str.remove('') #removendo o primeiro elemento que fica vazio por conta do split (

def convolution(layerStr, model):
	padding = layerStr[2].split(':')[1]		# padding: one of "valid" or "same"
	activation = layerStr[3].split(':')[1]	# activation: 
	bias = getBiasValue(layerStr[4])
	normalisation = layerStr[5]				# ?? 
	mergeInput = layerStr[6]				# ??

	filterLenght = 32  # ver como aumentar
	input_shape=(32,32,3)

	model.add(Conv2D(filterLenght, (3, 3), activation=activation, padding=padding, input_shape=input_shape))
	return model;

def poolAvg(layerStr, model):
	padding = layerStr[3].split(':')[1]

	model.add(AveragePooling2D(pool_size=(2, 2), padding=padding))
	return model;

def poolMax(layerStr, model):
	padding = layerStr[3].split(':')[1]

	model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))
	return model;

def classification(layerStr, model):
	activation = layerStr[1].split(':')[1]
	bias = getBiasValue(layerStr[3])

	model.add(Flatten())
	model.add(Dense(128, activation=activation))#, use_bias=bias))
	return model;

def softmax(layerStr, model):
	activation = 'softmax'
	bias = True		
	numUnits = 10	

	num_classes = 10
	model.add(Dense(num_classes, activation=activation, use_bias=bias))
	return model;

def learning(layerStr, model):
	learning = 'gradient-descent'

	return;

def getBiasValue(biasStr):
	if biasStr.split(':')[1] == 'False':	# use_bias: True or False
		return False
	return True

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

model = Sequential()

for i in range(len(rede_str)):
	layer_parts = rede_str[i].split(' ')

	# <convolution> 
	if (layer_parts[0] == 'layer:conv'):
		convolution(layer_parts, model)

	# <pooling> avg
	elif (layer_parts[0] == 'layer:pool-avg'):
		poolAvg(layer_parts, model)

	# <pooling> max
	elif (layer_parts[0] == 'layer:pool-max'):
		poolMax(layer_parts, model)

	# <classification> 
	elif (layer_parts[0] == 'layer:fc' and layer_parts[1] != 'act:softmax'):
		classification(layer_parts, model)

	# <softmax>
	elif (layer_parts[0] == 'layer:fc act:softmax'):
		softmax(layer_parts, model)

	# <learning>
	elif (layer_parts[0] == 'learning:gradient-descent'):
		learning(layer_parts, model)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
epochs = 70
useDataAugmentation = False

print(model.summary())

model_info = model.fit(train_features, train_labels, 
                               batch_size=batch_size, 
                               epochs=epochs, 
                               validation_data = (test_features_val, test_labels_val), 
                               verbose=0)

accuracyValue = accuracy(test_features_test, test_labels_test, model)

import pdb; pdb.set_trace()


# (layer:conv [num-filters,int,1,32,256][filter-shape,int,1,1,5][stride,int,1,1,3] padding:same act:sigmoid bias:True batch-normalisation:True merge-input:False
# (layer:fc act:relu [num-units,int,1,128,2048] bias:False) 
# (layer:fc act:softmax num-units:10 bias:True) 
# (learning:gradient-descent [lr,float,1,0.0001,0.1])

# (layer:conv [num-filters,int,1,32,256][filter-shape,int,1,1,5][stride,int,1,1,3] padding:valid act:linear bias:True batch-normalisation:False merge-input:True) 
# (layer:fc act:linear [num-units,int,1,128,2048] bias:True) 
# (layer:fc act:softmax num-units:10 bias:True) 
# (learning:gradient-descent [lr,float,1,0.0001,0.1])

# (layer:conv [num-filters,int,1,32,256][filter-shape,int,1,1,5][stride,int,1,1,3] padding:same act:relu bias:True batch-normalisation:True merge-input:False) 
# (layer:fc act:sigmoid [num-units,int,1,128,2048] bias:False) 
# (layer:fc act:softmax num-units:10 bias:True) 
# (learning:gradient-descent [lr,float,1,0.0001,0.1])




