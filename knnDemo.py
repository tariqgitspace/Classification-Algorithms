import numpy as np
import numbers
import math

def normalizeTrainingData(trainingData,normalized_Training_Matrix):
    startvalueForCategoricalData = 0
    trainingDataMeanArray =[]
    trainingDataSDArray = []
    
    dictlist = [dict() for x in range(len(trainingData[0])-1)]
    
    
    for col in range(len(trainingData[0])-1):  ##dont normalize last column
        column =[]
        column = [row[col] for row in trainingData]
        if isinstance(column[0], numbers.Number):
            mean = np.mean(column)
            sd = np.std(column)
            trainingDataMeanArray.append(mean)
            trainingDataSDArray.append(sd)
            normalizedColumn = (column - mean)/(sd)
            normalized_Training_Matrix[:,col] = normalizedColumn               
        else:
            keys = set(column) 
            values = list(range(startvalueForCategoricalData, len(keys)+startvalueForCategoricalData))
            dictionary = dict(zip(keys, values))
            dictlist[col] = dictionary
            for idx in range(len(column)):
                column[idx] = dictionary.get(column[idx])
            mean = np.mean(column)
            sd = np.std(column)
            trainingDataMeanArray.append(mean)
            trainingDataSDArray.append(sd)
            normalizedColumn = (column - mean)/(sd)
            normalized_Training_Matrix[:,col] = normalizedColumn  

    ##Step 3 Insert last Column as it is
    lastColumn = len(trainingData[0])-1
    column = [row[lastColumn] for row in trainingData]
    normalized_Training_Matrix[:,lastColumn] = column
                              
    return normalized_Training_Matrix,trainingDataMeanArray, trainingDataSDArray ,dictlist

                   
def getKNearestNeighbours(trainingData, numOfNeightbours, testLine):
    distanceLineList = []
    closestNeighbors = []
    for idx in range(len(trainingData)):
        distance = 0.0
        for attribute in range(len(testLine)):        ##ignore labal
            distance += math.pow((testLine[attribute] - trainingData[idx][attribute]), 2)
        #distance = math.sqrt(distance)

        distanceLineList.append((distance,trainingData[idx]))
    
    distanceLineList.sort(key=lambda x: x[0])  ##sort according to distance
    
    
    for idx in range(numOfNeightbours):
        closestNeighbors.append(distanceLineList[idx][1])
    return closestNeighbors


if __name__ == '__main__':

    trainingData_file = 'project3_dataset3_train.txt'
    testData_file = 'project3_dataset3_test.txt'
    numOfNeightbours = 
    
    
    ## Step 1 read file
    trainingData = []
    trainingData = np.genfromtxt(trainingData_file, dtype=None)
    trainingData=np.array(trainingData)
    
    testingData =[]
    testingData = np.genfromtxt(testData_file, dtype=None)
    testingData=np.array(testingData)
    
    
    
    numcols = len(trainingData[0])
    normalized_Training_Matrix=np.ndarray(shape=(len(trainingData),numcols))
    
    trainingDataMeanArray =[]
    trainingDataSDArray = []
    normalized_Training_Matrix,trainingDataMeanArray,trainingDataSDArray, dictlist = normalizeTrainingData(trainingData,normalized_Training_Matrix)



    predictedLabels = []
    actualLabels = []
    
    for line in testingData:
        actualLabels.append(int(line[-1]))
        temp =[]
        for i in range(len(trainingDataMeanArray)): ## This has 1 less column
            if isinstance(line[i], numbers.Number):
                temp.append(float((line[i]- trainingDataMeanArray[i])/(trainingDataSDArray[i])))
            else:
                temp.append(((dictlist[i].get(line[i]))- trainingDataMeanArray[i])/(trainingDataSDArray[i]))
        neighbors = getKNearestNeighbours(normalized_Training_Matrix,numOfNeightbours,temp)
        vote = 0
        for neigh in neighbors:
            if neigh[-1] == 1:
                vote += 1
            else:
                vote -= 1
        if(vote > 0):
            predictedLabels.append(1)
        else:
            predictedLabels.append(0)
    
    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    falsePositive = 0
    

    for j in range(len(actualLabels)):
        if(actualLabels[j] == predictedLabels[j]):
            if predictedLabels[j]==0:
                trueNegative +=1
            else:
                truePositive +=1
        else:
            if(predictedLabels[j]==0):
                falseNegative +=1
            else:
                falsePositive +=1
    
    
    print(truePositive)
    print(trueNegative)
    print(falseNegative)
    print(falsePositive)
    
    Accuracy = float(truePositive+trueNegative)/(truePositive+trueNegative+falseNegative+falsePositive)
    Accuracy *= 100
    Precision = float(truePositive/(truePositive+falsePositive))
    Precision *= 100
    Recall = float(truePositive/(truePositive+falseNegative))
    Recall *= 100
    FMeasure = float((2*truePositive)/((2*truePositive)+falseNegative+falsePositive))        
    FMeasure *= 100     
    
    
    print("Accuracy : "+str(Accuracy))
    print("Precision : "+str(Precision))
    print("Recall : "+str(Recall))
    print("FMeasure : "+str(FMeasure))
    