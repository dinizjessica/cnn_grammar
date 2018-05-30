import re

def hasPool(redeString):
    return 'pool' in redeString;

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
