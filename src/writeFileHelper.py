import os.path
import datetime


def openOrCreateLogFile():
    directory = 'log'
    fileName = directory+'/log_'+datetime.datetime.now().strftime("%d-%b-%y")+'.txt'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.isfile(fileName):
        arq = open(fileName, 'a')
    else:
        print('Logs created on file: ' + fileName)
        arq = open(fileName, 'w')
    return arq;

def writeLog(logMessage):
    arq = openOrCreateLogFile()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%p")
    msg = '\n'+timestamp+' --- '+logMessage
    print(msg)
    arq.writelines(msg)
    arq.close()

def writeArray(arrayMessage):
    arq = openOrCreateLogFile()
    timestamp = '\n'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%p")+' --- '
    arrayAsStr = ", ".join(map(str,arrayMessage))
    print(str(timestamp)+arrayAsStr)
    arq.writelines(str(timestamp)+arrayAsStr)
    arq.close()

def writeModelSummaryLog(model):
    arq = openOrCreateLogFile()
    print(model.summary())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%p")
    arq.writelines('\n'+timestamp+'\n')
    model.summary(print_fn=lambda x: arq.write(x + '\n'))
    arq.close()
