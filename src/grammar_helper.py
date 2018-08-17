import re
import os

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Merge, BatchNormalization, Concatenate
from keras.optimizers import SGD
from keras.models import Sequential

# (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:sigmoid bias:False batch-normalisation:False merge-input:False) 
# (pool-type:layer:pool-avg kernel-size:1 stride:1 padding:same) 
# (pool-type:layer:pool-max kernel-size:3 stride:2 padding:valid) 
# (layer:fc act:relu num-units:2048 bias:True) 
# (layer:fc act:softmax num-units:10 bias:True) 
# (learning:gradient-descent learning-rate:0.001)

def getConvOrPoolLayerArray(redeString):
	redeArray = splitRede(redeString)
	convOrPoolArray = list(filter(lambda x: x.startswith('layer:conv') or x.startswith('layer:pool-avg') or x.startswith('layer:pool-max'), redeArray))
	return convOrPoolArray;

def getConvOrPoolLayer(convOrPoolString, input_shape):
	layerType = getValueFrom(convOrPoolString, 'layer')

	if(layerType == 'conv'):
		return getConvLayer(convOrPoolString, input_shape)
	elif(layerType == 'pool-avg' or layerType == 'pool-max'):
		return getPoolLayer(convOrPoolString)

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

def getSoftmaxLayer(redeString):
	softmaxStr = getSoftmaxLayerString(redeString)
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
	stride = getValueFrom(convString, 'stride') #nao coloquei ainda
	padding = getValueFrom(convString, 'padding')
	activation = getValueFrom(convString, 'act')
	bias = True if getValueFrom(convString, 'bias') == 'True' else False
	batchNormalisation = True if getValueFrom(convString, 'batch-normalisation') == 'True' else False
	mergeInput = True if getValueFrom(convString, 'merge-input') == 'True' else False #nao sei como usar
	
	# model_x1 = Sequential()
	# model_x1.add(Conv2D(numFilters, filterShape, activation=activation, padding=padding, input_shape=input_shape, use_bias=bias))
	# if batchNormalisation: 
	# 	model_x1.add(BatchNormalization())

	out_i = Conv2D(numFilters, filterShape, activation=activation, padding=padding, input_shape=input_shape, use_bias=bias)
	if batchNormalisation: 
		out_i = BatchNormalization()(out_i)

	return Concatenate([out_i])

def getPoolLayer(poolString):
	# (pool-type:layer:pool-avg kernel-size:1 stride:1 padding:same) 
	# MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
	# AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
	poolType = getValueFrom(poolString, 'layer')
	kernelSize = int(getValueFrom(poolString, 'kernel-size')) # seria o pool_size?
	stride = int(getValueFrom(poolString, 'stride')) #nao coloquei ainda
	padding = getValueFrom(poolString, 'padding')

	if poolType == 'pool-max':
		return MaxPooling2D(pool_size=(2, 2), padding=padding)
	elif poolType == 'pool-avg':
		return AveragePooling2D(pool_size=(2, 2), padding=padding)

def getFCLayer(fcString): # classification and softmax
	# (layer:fc act:relu num-units:2048 bias:True) 
	# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
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



# ind = '(layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:sigmoid bias:False batch-normalisation:False merge-input:False) (pool-type:layer:pool-avg kernel-size:1 stride:1 padding:same) (pool-type:layer:pool-max kernel-size:3 stride:2 padding:valid) (layer:fc act:relu num-units:2048 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.001)'
# conv = 'layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:sigmoid bias:False batch-normalisation:True merge-input:False'
# pool_avg = 'layer:pool-avg kernel-size:1 stride:1 padding:same'
# pool_max = 'pool-type:layer:pool-max kernel-size:3 stride:2 padding:valid'
# lr = 'learning:gradient-descent learning-rate:0.001'
# input_shape = (3, 150, 150)
# red = splitRede(ind)
# import pdb; pdb.set_trace()
# getConvLayer(conv, input_shape)
# getPoolLayer(pool_max)
# getLearningOpt(lr)
# ind = '(layer:conv num-filters:128 filter-shape:2 stride:3 padding:same act:sigmoid bias:False batch-normalisation:True merge-input:False) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:fc act:sigmoid num-units:0000 bias:False) (layer:fc act:sigmoid num-units:1111 bias:False) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.001)'
# print(getClassificationLayerArray(ind))