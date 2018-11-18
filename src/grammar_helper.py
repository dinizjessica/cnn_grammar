import re
import os

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Merge, BatchNormalization, Concatenate
from keras.optimizers import SGD
from keras.models import Sequential


def getConvOrPoolLayerArray(redeString):
	redeArray = splitRede(redeString)
	convOrPoolArray = list(filter(lambda x: x.startswith('layer:conv') or x.startswith('layer:pool-avg') or x.startswith('layer:pool-max'), redeArray))
	return convOrPoolArray;

def getConvOrPoolLayer(convOrPoolString, input_shape):
	layerType = getValueFrom(convOrPoolString, 'layer')

	if(layerType == 'conv'):
		return getConvLayer(convOrPoolString, input_shape)
	elif(layerType == 'pool-avg' or layerType == 'pool-max'):
		return getPoolLayer(convOrPoolString, input_shape)

def getClassificationLayerArray(redeString):
	redeArray = splitRede(redeString)
	convOrPoolLayerQuatity = len(getConvOrPoolLayerArray(redeString))
	classificationQuant = (len(redeArray) - convOrPoolLayerQuatity - 2) # last two elements are softmax and learning

	if(classificationQuant == 2):
		return redeArray[-4:-2]
	else:
		return [redeArray[-3]]

def getClassificationLayer(classificationStr):
	return getFCLayer(classificationStr)

# this method can have an optional parameter called numUnits that will be used when the algorithm runs with different databases than cifar (having diferent number of classes)
def getSoftmaxLayer(redeString, *positional_parameters, **keyword_parameters):
	softmaxStr = getSoftmaxLayerString(redeString)

	if ('numUnits' in keyword_parameters):
		return getFCLayer(softmaxStr, numUnits=keyword_parameters['numUnits'])
	else: 
		return getFCLayer(softmaxStr)

def getLearningOptFromNetwork(redeString):
	learningString = getLearningString(redeString)
	return getLearningOpt(learningString)

####### private ####### 

def splitRede(redeString):
	redeString = redeString.replace('(','')
	redeArray = redeString.split(')')
	redeArray = [x.strip() for x in redeArray]
	return redeArray[:-1]; # removing the last because it is an empty string

def getSoftmaxLayerString(redeString):
	redeArray = splitRede(redeString)
	softmaxString = redeArray[-2]
	return softmaxString

def getLearningString(redeString):
	redeArray = splitRede(redeString)
	learnigString = redeArray[-1]
	return learnigString

def getConvLayer(convString, input_shape):
	numFilters = int(getValueFrom(convString, 'num-filters'))
	filterShapeNum = int(getValueFrom(convString, 'filter-shape'))
	filterShape = (filterShapeNum,filterShapeNum)
	stride = getValueFrom(convString, 'stride')
	padding = getValueFrom(convString, 'padding')
	activation = getValueFrom(convString, 'act')
	bias = True if getValueFrom(convString, 'bias') == 'True' else False
	batchNormalisation = True if getValueFrom(convString, 'batch-normalisation') == 'True' else False
	mergeInput = True if getValueFrom(convString, 'merge-input') == 'True' else False #nao sei como usar
	
	
	return Conv2D(numFilters, filterShape, activation=activation, padding=padding, input_shape=input_shape, use_bias=bias)

def hasBatchNormalization(convOrPoolString):
	layerType = getValueFrom(convOrPoolString, 'layer')

	if(layerType == 'conv'):
		return True if getValueFrom(convOrPoolString, 'batch-normalisation') == 'True' else False
	return False

def getPoolLayer(poolString, input_shape):
	# (pool-type:layer:pool-avg kernel-size:1 stride:1 padding:same) 
	# MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
	# AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
	poolType = getValueFrom(poolString, 'layer')
	kernelSize = int(getValueFrom(poolString, 'kernel-size')) # seria o pool_size?
	stride = int(getValueFrom(poolString, 'stride'))
	padding = getValueFrom(poolString, 'padding')

	if poolType == 'pool-max':
		return 	MaxPooling2D(pool_size=(2, 2), strides=stride, padding=padding, input_shape=input_shape)
	elif poolType == 'pool-avg':
		return AveragePooling2D(pool_size=(2, 2), strides=stride, padding=padding, input_shape=input_shape)

# this method can have an optional parameter called numUnits that will be used when the algorithm runs with different databases than cifar (having diferent number of classes)
def getFCLayer(fcString, *positional_parameters, **keyword_parameters): # classification and softmax
	# (layer:fc act:relu num-units:2048 bias:True) 
	# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
	if ('numUnits' in keyword_parameters):
		numUnits = keyword_parameters['numUnits']
	else:
		numUnits = int(getValueFrom(fcString, 'num-units'))
	activation = getValueFrom(fcString, 'act')
	bias = True if getValueFrom(fcString, 'bias') == 'True' else False

	return Dense(numUnits, activation=activation, use_bias=bias)

def getLearningOpt(learningString):
	# (learning:gradient-descent learning-rate:0.001)
	learningRate = float(getValueFrom(learningString, 'learning-rate'))
	return SGD(lr=learningRate)

def getValueFrom(convString, fieldName):
	regex ='.*?'+fieldName+':(\S+)'
	rg = re.compile(regex,re.IGNORECASE|re.DOTALL)
	m = rg.search(convString)
	return m.group(1)
