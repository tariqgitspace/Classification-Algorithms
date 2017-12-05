import numpy as np
import numbers

def KFoldCrossValidation(records,crossValidation):
    print("KFold Splitting Start")
    #random.shuffle(records)
    ##get all elements record_number modulo crossValidation
    testingDataIdxs = []
    for i in range(crossValidation):
        idx = i
        testing =[]
        while (idx < len(records)):
            testing.append(idx)
            idx += crossValidation
        testingDataIdxs.append(testing)

    trainingDataIdxs =[]
    
    for i in range(crossValidation):
        training =[]
        for idx in range(len(records)):
            if not(idx in testingDataIdxs[i]):
                training.append(idx)
        trainingDataIdxs.append(training)
       
    print("KFold Complete")
    return trainingDataIdxs,testingDataIdxs
        
    
def normalizeTrainingData(trainingData,normalized_Training_Matrix):
    startvalueForCategoricalData = 5
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
    numberOfValidColumns = len(trainingData[0])-1
    for line in trainingData:  ##Find distance from every Training Data
        distance = np.linalg.norm(np.array(testLine)-np.array(line[0:numberOfValidColumns]))
        distanceLineList.append((distance,line))
    
    distanceLineList.sort(key=lambda x: x[0])  ##sort according to distance
    
    closestNeighbors = []
    for idx in range(numOfNeightbours):
        closestNeighbors.append(distanceLineList[idx][1])
    return closestNeighbors


if __name__ == '__main__':

    text_file = 'project3_dataset2.txt'
    crossValidation = 10
    numOfNeightbours = 11
    
    
    ## Step 1 read file
    fileData = np.genfromtxt(text_file, dtype=None)
    fileData=np.array(fileData)
    
    numrows = len(fileData)
    numcols = len(fileData[0])
    
    #numOfNeightbours = int((numrows**0.5)/2)
    print(numOfNeightbours)    
    #print("In", text_file, "==== row count-->",numrows,"columns count-->",numcols)
    
    
    
   
    ##Step 4 Get K- fold
    trainingDataIdxs,testingDataIdxs = KFoldCrossValidation(fileData,crossValidation)
    
    
    totalAccuracy = []
    totalPrecision = []
    totalRecall = []
    totalFMeasure = []
    
    
    
    for fold in range(crossValidation):
        
        trainingData = []
        for idx in trainingDataIdxs[fold]:
            trainingData.append(fileData[idx])
            
        normalized_Training_Matrix=np.ndarray(shape=(len(trainingData),numcols))
        
        testingData =[]
        for idx in testingDataIdxs[fold]:
            testingData.append(fileData[idx])  
        

        normalized_Training_Matrix,trainingDataMeanArray,trainingDataSDArray, dictlist = normalizeTrainingData(trainingData,normalized_Training_Matrix)

    
        #print(normalized_matrix[:,lastColumn])
       
    
        predictedLabels = []
        actualLabels = []
        
        for line in testingData:
            actualLabels.append(line[-1])
            temp =[]
            for i in range(len(trainingDataMeanArray)):
                if isinstance(line[i], numbers.Number):
                    temp.append((line[i]- trainingDataMeanArray[i])/(trainingDataSDArray[i]))
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
        
        
        Accuracy = float(truePositive+trueNegative)/(truePositive+trueNegative+falseNegative+falsePositive)
        Accuracy *= 100
        Precision = float(truePositive/(truePositive+falsePositive))
        Precision *= 100
        Recall = float(truePositive/(truePositive+falseNegative))
        Recall *= 100
        FMeasure = float((2*truePositive)/((2*truePositive)+falseNegative+falsePositive))        
        FMeasure *= 100
        
        print(str(fold+1)+" Iteration Complete !!")
        
        
        print("Accuracy : "+str(Accuracy))
        print("Precision : "+str(Precision))
        print("Recall : "+str(Recall))
        print("FMeasure : "+str(FMeasure))
        
        totalAccuracy.append(Accuracy)
        totalPrecision.append(Precision)
        totalRecall.append(Recall)
        totalFMeasure.append(FMeasure)

    
    print("averageAccuracy  : "+str(np.sum(totalAccuracy)/crossValidation))
    print("averagePrecision  : "+str(np.sum(totalPrecision)/crossValidation))
    print("averageRecall  : "+str(np.sum(totalRecall)/crossValidation))
    print("averageFMeasure  : "+str(np.sum(totalFMeasure)/crossValidation))    
    