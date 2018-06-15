import re
import os

def hasPool(redeString):
    return 'sem_pool' not in redeString;

def hasDropout(redeString):
    return 'sem_dropout' not in redeString;

def getConvQuant(redeString):
	convString = re.findall(r'\([^()]*\)', redeString)
	numberOfConvs = re.findall(r'\d+', convString[0])
	return int(numberOfConvs[0]);

def getLayerQuantity(redeString):
	convToRemoveFromString = re.findall(r'\([^()]*\)', redeString)
	redeString = redeString.replace(convToRemoveFromString[0],'')
	poolToRemoveFromString = re.findall(r'\([^()]*\)', redeString)
	redeString = redeString.replace(poolToRemoveFromString[0],'')

	layerQuantityParenthesis = re.findall(r'\([^()]*\)', redeString)
	numberOfLayer = re.findall(r'\d+', layerQuantityParenthesis[0])
	return int(numberOfLayer[0]);

def getFCquantity(redeString):
	numberOfFC = re.findall(r'\d+', redeString)
	return int(numberOfFC[-1])

def getLearningRate(redeString):
	learningRate = re.findall("\\[(.*?)\\]", redeString)
	return float(learningRate[0])

def getNumberOfClasses(path):
    for _, dirnames, _ in os.walk(path):
        return len(dirnames);

def getQuantityOfFilesInAFolder(path):
	quantFiles = 0
	for _, _, filenames in os.walk(path):
		quantFiles += len([x for x in filenames if '.DS_Store' not in x])
	return quantFiles;

def getDirNamesInAFolder(path):
	for _, dirnames, _ in os.walk(path):
		return dirnames
