import sys

from neuralNetwork_any_dataset import runNeuralNetwork

#############################
# chamar pela linha de comando: python variasExecucoes.py <numero_execucoes> <arquitetura> 
#############################

nExecucoes = 1
arquitetura = '(((conv*3)vazio)*3)fc*2'

if len(sys.argv) > 1:
	for idx, param in enumerate(sys.argv[1:]):
		if idx == 0:
			nExecucoes = param
		if idx == 1:
			arquitetura = param


for i in range(int(nExecucoes)):
	print('Execucao '+str(i))
	runNeuralNetwork(arquitetura)
