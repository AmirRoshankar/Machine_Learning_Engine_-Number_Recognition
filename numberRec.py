from mlxtend.data import loadlocal_mnist
import numpy as np
from numpy import random
import sys
import math
import pickle
from random import random
#import mysql.connector as mySql
#from pymongo import


netSave  = open("netSample.txt", "wb")


smallVal = .1
learnRate = .1

trainImgs = []
trainNums = []
network = []

convFrameWPerc = .25 #.1429
convFrameHPerc = .25 #.1429
maxPixVal = 255

imgW = 28
imgH = 28

maskResW=0.03571428571
maskResH=0.03571428571

batchSize = 1
errorSum = 0

layerLen= []
layerLen.append(16)
layerLen.append(10)
#layerLen.append(10)

curLayerLen = round(1/(convFrameWPerc*convFrameHPerc))

iWeights = []
iBias = []
netDepth =2 #includes the output layer but not the raw input

def relu(n):
    #print("n: " + str(n))
    #return n/10
    #return 1/(1+math.exp(-n))
    #return 0 if n <= 0 else n
    #n/=100
    if n>100:
        return 1
    if n <-100:
        return 0
    return (1/(1+math.exp(-n)))
    return (math.atan(n)+math.pi/2)/math.pi


def toConv(input, inW, inH, convWeights, bias):
    global convFrameWPerc, convFrameHPerc
    convLayer = []
    for i in range(0, round(1/convFrameHPerc)):
        for j in range(0, round(1/convFrameWPerc)):
            curFrame = [[0 for a in range (round(convFrameWPerc*inW))] for b in range(round(convFrameHPerc*inH))]
            x = round(j*convFrameWPerc*inW)
            y = round(i*convFrameHPerc*inH)
            #print ("x: " + str(x) + " y: " + str(y))
            s = y * inH + x
            for r in range (0, round(convFrameHPerc*inH)):
                for c in range (0, round(convFrameWPerc*inW)):
                    curFrame[r][c] = input[s+r*inW+c]
            convLayer.append(convFrameTransform(curFrame, convWeights[round(i/convFrameHPerc+j)], bias[round(i/convFrameHPerc+j)]))
        #convLayer.append(convFrameTransform(curFrame))
    #print(str(convLayer))
    return convLayer

def toFlat(prevLayer, flatLen, flatWeights, bias):
    flatLayer = [0 for x in range(flatLen)]
    for i in range(flatLen):
        sum =0
        for j in range(len(prevLayer)):
            sum += prevLayer[j] * flatWeights[i][j]
        flatLayer[i] = relu(sum + bias[i])
    return flatLayer

def convFrameTransform (curFrame, convWeights, bias):
    sum = 0
    for i in range(len(curFrame)):
        for j in range (len(curFrame[0])):
            sum += curFrame[i][j] * convWeights[i][j]
    return relu(sum + bias)

def readTrainData():
    global trainImgs, trainNums
    trainImgs, trainNums = loadlocal_mnist(images_path = 'train-images-idx3-ubyte', labels_path = 'train-labels-idx1-ubyte')

def setRandParams(params, n):
    for i in range(0 , n):
        params[i] = np.random.random_sample()/maxPixVal

def most(list):
    if len(list) <=1:
        return 0
    else:
        max = -1
        maxI = 0
        for i in range(len(list)):
            if list[i]>max:
                max = list[i]
                maxI = i
        return maxI

def mask(img):
    global imgW, imgH, maskResH, maskResW
    convLayer = []
    out = []
    for i in range(0, round(1/maskResH)):
        for j in range(0, round(1/maskResW)):
            curFrame = [[0 for a in range (round(maskResW*imgW))] for b in range(round(maskResH*imgH))]
            x = round(j*maskResW*imgW)
            y = round(i*maskResH*imgH)
            #print ("x: " + str(x) + " y: " + str(y))
            s = y * imgH + x
            sum = 0
            for r in range (0, round(maskResH*imgH)):
                for c in range (0, round(maskResW*imgW)):
                    sum += img[s+r*imgW+c]
            out.append(sum*maskResH*maskResW)
        #convLayer.append(convFrameTransform(curFrame))
    #print(str(convLayer))
    #print("\nmask: " + str(out) + "\n")
    return out

def predicts(imgs):
    global network, layerLen
    out = []
    for i in range(len(imgs)):
        for j in range(len(network)):
            #print("layer: " + str(j))
            network[j]['prevLayer'] = imgs[i] if j ==0 else network[j-1]['layer']
            network[j]['layer'] = toFlat(network[j]['prevLayer'], layerLen[j],
                                    network[j]['weights'], network[j]['bias'])
        out.append(network[len(network)-1]['layer'])
    return out

def forwardProp(img):
    global network, layerLen
    for j in range(len(network)):
        #print("layer: " + str(j))
        network[j]['prevLayer'] = img if j ==0 else network[j-1]['layer']
        network[j]['layer'] = toFlat(network[j]['prevLayer'], layerLen[j],
                                    network[j]['weights'], network[j]['bias'])
    result = most(network[netDepth-1]['layer'])
    print("\nsummary out: " + str(result) +"\n")
    return result

def calcCost(out, real):
    cost = 0
    for i in range(len(real)):
        for j in range(len(real[i])):
            cost += (real[i][j]-out[i][j])**2
    cost/= (2*len(real))
    return cost

#def printNet(network):
    #for layer in range(len(network)):
        #print("layer: " +str(layer))
        #print("\tweights: " + str(network[layer]['weights']))
        #print("\tbiasses: " + str(network[layer]['bias']))


def initNetwork():
    global network, imW, imH, iWeights, netDepth, maskResH, maskResH
    global layerLen, iBias, maxPixVal

    for i in range(netDepth):
        if i ==0:
            iWeights.append([[random()*2-1 for x in range(round(1/(maskResH*maskResW)))] for y in range(layerLen[i])])
            #iWeights.append([[((np.random.random_sample()*2)-1) for x in range(round(1/(maskResH*maskResW)))] for y in range(layerLen[i])])
            #iWeights.append([[0 for x in range(round(1/(maskResH*maskResW)))] for y in range(layerLen[i])])
        else:
            iWeights.append([[random()*2-1 for x in range(layerLen[i-1])] for y in range(layerLen[i])])
            #iWeights.append([[((np.random.random_sample()*2)-1) for x in range(layerLen[i-1])] for y in range(layerLen[i])])
            #iWeights.append([[0 for x in range(layerLen[i-1])] for y in range(layerLen[i])])
        iBias.append([random()*2-1 for x in range(layerLen[i])])
        #iBias.append([(np.random.random_sample()*2-1) for x in range(layerLen[i])])
        #iBias.append([0 for x in range(layerLen[i])])
        #iErrors.append(0 for x in range(layerLen[i])])
        tempLayer = {
            'layer' : None,
            'weights': iWeights[i],
            'bias' : iBias[i],
            'layerType' : 'flat',
            'error': [0 for x in range(layerLen[i])],
            'prevLayer' : None,
            }
        network.append(tempLayer)
'''
def initDB():
    dataBase = mysql.connect(
        host = "localhost",
        user = "root",
        password = "secret_root123"
    )
    #print(dataBase)

    #cursor = dataBase.cursor()

    #try:
    #    cursor.execute("DROP DATABASE neuralNetwork")
    #except Error as errorReport:
        #print(errorReport)

    #cursor.execute("CREATE DATABASE weight (PRIMARY KEY weightFromID INT(), value DECIMAL(13, 5))")
    #cursor.execute("CREATE DATABASE neuron (PRIMARY KEY neuronID INT(), output DECIMAL (13,5), bias DECIMAL (13, 5), error (13, 5), FOREIGN KEY weights INT() REFERENCES weight)")
    #cursor.execute("CREATE DATABASE layer (PRIMARY KEY layerID INT() AUTO_INCREMENT, neuronslayers INT())")
    #cursor.execute("CREATE DATABASE neuralNetwork (PRIMARY KEY netID CHAR(100), netDepth INT(), FOREIGN KEY layers INT() REFERENCES layer)")
'''


def deriv(signal):
    return signal * (1-signal)

def getError(exp, signal):
    return (exp-signal) * deriv(signal)

def backPropErrors(realNum):
    global network
    for layer in reversed(range(len(network))):
        for cellNum in range(len(network[layer]['layer'])):
            if layer==netDepth-1:
                network[layer]['error'][cellNum] += getError(realNum[cellNum], network[layer]['layer'][cellNum])
            else:
                for toCell in range(len(network[layer+1]['layer'])):
                    network[layer]['error'][cellNum] += (network[layer+1]['error'][toCell]*network[layer+1]['weights'][toCell][cellNum])*deriv(network[layer]['layer'][cellNum])
def clearErrors():
    global network, netDepth
    #print(str(network[netDepth-1]['error']) + "\n")
    for layer in range(netDepth):
        for cellNum in range(len(network[layer]['error'])):
            network[layer]['error'][cellNum] = 0

def divErrors(batchSize):
    global network
    for layer in range(len(network)):
        for cellNum in range(len(network[layer]['error'])):
            network[layer]['error'][cellNum] /= batchSize

def backPropAndTrain(batchInImg, batchInNum):
    global network, netDepth
    batchSize = len(batchInNum)

    for sampleCount in range(batchSize):
        print("real: " + str(most(batchInNum[sampleCount])))
        forwardProp(batchInImg[sampleCount])
        backPropErrors(batchInNum[sampleCount])
    divErrors(batchSize)
    train()
    #print("\noutput: " + str((network[netDepth-1]['layer'])) + "\nsummary out: " + str(most(network[netDepth-1]['layer'])) +"\n")
    clearErrors()

def train():
    global network
    for layer in range(len(network)):
        for toCell in range(len(network[layer]['layer'])):
            for fromCell in range(len(network[layer]['prevLayer'])):
                network[layer]['weights'][toCell][fromCell] += learnRate * network[layer]['error'][toCell] * network[layer]['prevLayer'][fromCell]
        network[layer]['bias'][toCell] += learnRate * network[layer]['error'][toCell]


def calcErrorSum():
    global network, errorSum



def main():
    global trainImgs, trainNums, network, imW, imH, iFlatWeights0
    global iFlatWeights1, iFlatWeights2, flatLen1, flatLen2, iFlatBias0
    global iFlatBias1, iFlatBias2, maxPixVal, batchSize, netSave

    initNetwork()
    readTrainData()

    sys.setrecursionlimit(12000)
    print (sys.getrecursionlimit())

    batchInImg = []
    batchInNum = []
    correctCount = 0.0
    afterTrainCount = 0.0
    for i in range(round(len(trainImgs))):
        masked = mask(trainImgs[i])

        batchInImg.append(masked)#trainImgs[i])

        singleInNum=[]
        for x in range(10):
            singleInNum.append(1 if x == trainNums[i] else 0)
        batchInNum.append(singleInNum)
        if len(batchInImg) ==batchSize:

            diffCost = 0
            print("round: " + str(i))
            backPropAndTrain(batchInImg, batchInNum)
            if i > 30000:
                correctCount += forwardProp(batchInImg[batchSize-1]) == most(batchInNum[batchSize-1])
                afterTrainCount += 1
                #print(str(batchInNum[batchSize-1]) + " "  + str(forwardProp(batchInImg[batchSize-1])))
                #print(afterTrainCount)
                print("accuracy: " + str(correctCount/afterTrainCount))
            if i % 30 ==0:
                correctCount = 0.0
                afterTrainCount = 0.0
            '''
            batchOut = (predicts(batchInImg))
            print("\n\nround: " + str(i) + "\npredics: " + str(batchOut[0]) + "\nactual: " + str(batchInNum[0]))

            cost = calcCost(batchOut, batchInNum)
            diffCost = 0
            print("cost1: " + str(cost))
            #while diffCost != 0:
            gradDescent(batchInImg, batchInNum, cost)
            #print("\nnetwork: " + str(network) + "\n")
            #printNet(network)
            newPred = predicts(batchInImg)
            print("predicts: " + str(newPred[0]))
            cost2 = calcCost(newPred, batchInNum)
            print("cost2: " + str(cost2))
            diffCost = cost2-cost
            print("loss in cost: " + str(diffCost))
            #cost = cost2
            '''
            batchInImg = []
            batchInNum = []
            batchOut = []
            if i%5000:
                pickle.dump(network, netSave)
            sys.stdout.flush()

    pickle.dump(network, netSave)
    netSave.close()

        #result = most(network[2]['layer'])
        #print((result))

        #cost = trainNums[i]-result #how much result is short by

        #print("round: " + str(i) + " result: " + str(result) + " cost: " + str(cost))

        #trainNum = [0]*10
        #trainNum[trainNums[i]]+=1

        #print(network[2]['layer'])
        #train(trainNum, network)

        #print()

if __name__ == '__main__':
    main()
