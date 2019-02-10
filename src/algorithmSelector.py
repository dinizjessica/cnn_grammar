from neuralNetwork_assuncao_outrasBases import runNeuralNetwork as outrasBases
from neuralNetwork_assuncao import runNeuralNetwork as baseCifar
from neuralNetwork_assuncao_fmri import runNeuralNetwork as baseFMRI

from os import path

#####################################
# pre-defined configuration
#####################################
# select the data base to run the algorithm; it can be either 'cifar' or 'outras'
base = 'outras'

epochs = 100
batch_size = 128  # 128 para as de melanoma e 32 para o cifar
img_width, img_height = 120, 120 # used during tests for melanoma dataset 

# if testing with another database but cifar, inform the dataset diretory
# data_dir = '/mnt/E0A05FEAA05FC5A6/Bases/melanoma/jessica/data/data'
# data_dir = '/content/drive/My Drive/UFRPE/mestrado-melanoma_set/skin_lesions_4_classes'
data_dir = path.abspath(path.join(__file__, "../../../../UFRPE/mestrado-melanoma_set/skin_lesions_4_classes"))

#####################################

def runNeuralNetwork(networkArchitecture):
	if base == 'cifar':
		return baseCifar(networkArchitecture, epochs, batch_size)
	elif base == 'fmri':
		return baseFMRI(networkArchitecture)
	elif base == 'outras':
		return outrasBases(networkArchitecture, 
						   data_dir, 
						   epochs,
						   batch_size, 
						   img_width, 
						   img_height)

############################################
# Testes
############################################
# ind = '(layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
# ind = '(layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:256 filter-shape:4 stride:1 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:64 filter-shape:2 stride:3 padding:same act:linear bias:False batch-normalisation:True merge-input:True) (layer:pool-max kernel-size:1 stride:2 padding:valid) (layer:conv num-filters:32 filter-shape:1 stride:3 padding:valid act:relu bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:5 stride:3 padding:same act:linear bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:128 filter-shape:4 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:False) (layer:conv num-filters:32 filter-shape:2 stride:1 padding:same act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:256 filter-shape:1 stride:2 padding:valid act:relu bias:False batch-normalisation:False merge-input:True) (layer:conv num-filters:128 filter-shape:4 stride:2 padding:same act:sigmoid bias:True batch-normalisation:False merge-input:False) (layer:conv num-filters:256 filter-shape:1 stride:1 padding:same act:sigmoid bias:False batch-normalisation:False merge-input:True) (layer:fc act:sigmoid num-units:128 bias:True) (layer:fc act:softmax num-units:10 bias:True) (learning:gradient-descent learning-rate:0.01)'
# runNeuralNetwork(ind)