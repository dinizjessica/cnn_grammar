import os.path
import datetime


def openOrCreateLogFile():
    fileName = 'log_'+datetime.datetime.now().strftime("%d-%b-%y")+'.txt'
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

def writeModelSummaryLog(model):
    arq = openOrCreateLogFile()
    print(model.summary())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%p")
    arq.writelines('\n'+timestamp+'\n')
    model.summary(print_fn=lambda x: arq.write(x + '\n'))
    arq.close()
    

