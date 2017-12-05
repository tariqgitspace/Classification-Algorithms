import numpy as np
import math
import numbers
import pandas
import random 


def nominalDataPartition(records,attributeColumn,attributeValue):
    #print("Categorical partion")
    class1 =[]
    class0 = []
    for record in records:
        if record[attributeColumn] == attributeValue:        
            class1.append(record)
        else:
            class0.append(record)
    return class0,class1


def continuousDataPartition(records,attributeColumn,attributeValue):
    ##print("Values partion "+ str(attributeColumn) + " "+str(attributeValue))
    class1 =[]
    class0 = []
    attributeColumn = int(attributeColumn)
    attributeValue = float(attributeValue)
    for record in records:
        if record[attributeColumn] >= attributeValue:        
            class1.append(record)
        else:
            class0.append(record)
    return class0,class1



def GINI(records):
    zeroCount = 0
    oneCount = 0
    for record in records:
        label = record[-1]
        if int(label)==0:
            zeroCount +=1
        else:
            oneCount +=1

    Gini = 1.0
    zeroProbability = 0.0
    oneProbability = 0.0
    zeroProbability = float(zeroCount/(len(records)))
    oneProbability = float(oneCount/(len(records)))
    Gini = Gini - math.pow(zeroProbability,2) - math.pow(oneProbability,2)
    
    return Gini



def MaxGainSplitting(records):
    max_gain = 0
    max_attributeColumn = 0
    max_attribute_Value = 0
    
    if len(records) == 0:
        print("Zero Recoords")
        return max_gain, max_attributeColumn,max_attribute_Value
    parentGINI = GINI(records)
    allColumnsExceptLast = len(records[0]) - 1

    for attributeColumn in range(allColumnsExceptLast):
        diffValuesOfAttribute = set([row[attributeColumn] for row in records]) 
        
        for attributeval in diffValuesOfAttribute:
            if isinstance(attributeval, numbers.Number):
                class0, class1 = continuousDataPartition(records, attributeColumn, attributeval)
            else:
                class0, class1 = nominalDataPartition(records, attributeColumn, attributeval)

            ##cannot split based on this attribute
            if len(class1) == 0 or len(class0) == 0:
                continue

            probabilityClass1 = float(len(class1) / (len(class1) + len(class0)))
            probabilityClass0 = float(len(class0) / (len(class1) + len(class0)))
            gain = parentGINI - (probabilityClass1 * GINI(class1)) - (probabilityClass0 * GINI(class0))

            if gain >= max_gain:
                max_gain = gain
                max_attributeColumn = attributeColumn
                max_attribute_Value = attributeval

    return max_gain, max_attributeColumn,max_attribute_Value





class Tree(object):
    def __init__(self,attribute,attribute_Value, left, right,class1Count,class0Count):
        self.attributeColumn = attribute
        self.attribute_Value = attribute_Value
        self.left = left
        self.right = right
        self.class1Count=class1Count
        self.class0Count=class0Count


def saveResultInLeaf(records,attributeColumn, attribute_Value):
    class1Count=0
    class0Count=0
    for record in records:
        if int(record[-1])==1:
            class1Count +=1
        else:
            class0Count +=1
    return Tree(attributeColumn, attribute_Value, None, None,class1Count,class0Count)
    
def recursiveBST(records,level,levelThreshold,minNodeSize):

    """
    if(level > levelThreshold):
        return saveResultInLeaf(records,None,None)
    
    if(len(records) < minNodeSize):
        return saveResultInLeaf(records,None,None)
    """
    
    Gain, attributeColumn, attribute_Value = MaxGainSplitting(records)

    if Gain == 0:
        return saveResultInLeaf(records,None,None)
        
    if isinstance(attribute_Value, numbers.Number):
        class0Count, class1Count  = continuousDataPartition(records, attributeColumn,attribute_Value)
    else:
        class0Count, class1Count = nominalDataPartition(records, attributeColumn,attribute_Value)

    node = Tree(attributeColumn,attribute_Value, None, None,-1,-1)
    node.left = recursiveBST(class0Count,level+1,levelThreshold,minNodeSize)
    node.right = recursiveBST(class1Count,level+1,levelThreshold,minNodeSize)

    return node


def TestingDataClassification(root,record):

    #print("TestingDataClassification " + str(root.attributeColumn) + " "+str(root.attribute_Value))
   
   
    if root.left == None and root.right==None:
        if (root.class1Count ==-1) and (root.class0Count == -1):
            print("something Wrong in class1Count")
        if(root.class1Count >= root.class0Count):
            return 1
        else:
            return 0
    """
    if root.left == None and root.right==None:
        if isinstance(root.attribute_Value, numbers.Number):
            if(record[root.attributeColumn] >= root.attribute_Value):
                #print("Prediction True")
                return 1
            else:
                #print("Prediction False")
                return  0
        else:
            if(record[root.attributeColumn] == root.attribute_Value):
                #print("Prediction True")
                return 1
            else:
                #print("Prediction False")
                return 0
        return
    """
    if isinstance(root.attribute_Value, numbers.Number):
        if(record[root.attributeColumn] >= root.attribute_Value):
            return TestingDataClassification(root.right,record)
        else:
            return TestingDataClassification(root.left,record)        
    else: ## is string
        if record[root.attributeColumn] == root.attribute_Value:
            return TestingDataClassification(root.right,record)
        else:
            return TestingDataClassification(root.left,record)





def KFoldCrossValidation(records,crossValidation):
    print("KFold Splitting Start")
    #random.shuffle(records)
    ##get all elements record_number modulo 10
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
       
   
    print("KFold Splitting Complete")
    
    return trainingDataIdxs,testingDataIdxs
      
def bootstrapSample(weight,Data,bagSize):
    
    #print("Weight Probability Sum: "+ str(np.sum(weight)))
    sampleData =[]
    sampleData = np.random.choice(Data, bagSize, p=weight,replace=True)
    #randlist = pandas.DataFrame(index=np.random.randint(numrows, size=bagSize))
        
    """
    for i in range(len(sampleData)): #generate pairs
        for j in range(i+1,len(sampleData)): 
            if np.array_equal(sampleData[i],sampleData[j]):
                print (i, j)
    """
    
    return sampleData
    


     
def testAccuracy(Data,predictedLabels):
    length = len(Data)

    same =0
    for idx in range(length):
        if (predictedLabels[idx] == Data[idx][-1]):
            same +=1
            
    print("testAccuracy: " + str(float(same/length)))
  
    
    
def testDataBoostAccuracy(testData,allLearners,allAlpha):
    
    predicitonForDifferentLearners =[]
    lengthOfTestData = len(testData)
    numLearners = len(allLearners)
    
    #print(allAlpha)
    
    for learner in allLearners:
        testingprediction = []
        for line in testData:
            label = TestingDataClassification(learner,line)
            testingprediction.append(int(label))
        #testAccuracy(testData,testingprediction)
        predicitonForDifferentLearners.append(testingprediction)

           
   
    
    finalPredictedLabels = []    
    alphaSum = np.sum(allAlpha)
    
    for col in range(lengthOfTestData):
        prediction = 0.0
        for learner in range(numLearners):
            prediction += allAlpha[learner]*predicitonForDifferentLearners[learner][col]
        
        prediction = float(prediction/alphaSum)    ### ???
        #print(prediction)
        if(prediction >= 0.5):
            finalPredictedLabels.append(1)
        else:
            finalPredictedLabels.append(0)

    """
    for col in range(lengthOfTestData):
        for learner in range(numLearners):
            if predicitonForDifferentLearners[learner][col] == 1 :
                finalPredictedLabels[col] += 1
            else:
                finalPredictedLabels[col] -= 1


    for idx in range(lengthOfTestData):
        if finalPredictedLabels[idx] >= 0:
            finalPredictedLabels[idx] = 1
        else:
            finalPredictedLabels[idx]=0     

    """

    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    falsePositive = 0
    
    for row in range(lengthOfTestData):
        #print(str(finalPredictedLabels[j]) + "  : " + str(testData[j][-1]))
        if(finalPredictedLabels[row] == testData[row][-1]):
            if finalPredictedLabels[row]==0:
                trueNegative +=1
            else:
                
                truePositive +=1
        else:
            #print(str(finalPredictedLabels[j]) + "  : " + str(testData[j][-1]))
            if(finalPredictedLabels[row] == 0):
                falseNegative +=1
            else:
                falsePositive +=1
              
    
    Accuracy = float(truePositive+trueNegative)/(truePositive+trueNegative+falseNegative+falsePositive)
    Precision = float(truePositive/(truePositive+falsePositive))
    Recall = float(truePositive/(truePositive+falseNegative))
    FMeasure = float((2*truePositive)/((2*truePositive)+falseNegative+falsePositive))        
    
    
    
    return Accuracy,Precision,Recall,FMeasure
        
    
if __name__ == '__main__':

    text_file = 'project3_dataset1.txt'
    levelThreshold = 3
    minNodeSize =10
    crossValidation = 10
    bagSizeRatio = 1
    numLearners = 5

    fileData = np.genfromtxt(text_file, dtype=None)
    fileData=np.array(fileData)
    
    #random.shuffle(fileData)
    
    numrows = len(fileData)
    numcols = len(fileData[0])
    print("In", text_file, "==== row count-->",numrows,"columns count-->",numcols)
    

    trainingDataIdxs,testingDataIdxs = KFoldCrossValidation(fileData,crossValidation)
    
    totalAccuracy = []
    totalPrecision = []
    totalRecall = []
    totalFMeasure = []

    
    for fold in range(crossValidation):
        trainingData = []
        
        for idx in trainingDataIdxs[fold]:
            trainingData.append(fileData[idx])
        
        lengthTraining = len(trainingData)
        bagSize = np.ceil(lengthTraining * bagSizeRatio)
        bagSize = int(bagSize)
        initalWeight = float(1/lengthTraining)
        weights = []
        weights = initalWeight * np.ones(lengthTraining)
        
        
        allAlpha =[]
        allLearners = []
        root = None
        validLearners = 0
        while(validLearners < numLearners):
            ##choose sample and call learner
            sampleData =[]
            sampleData = bootstrapSample(weights,trainingData,bagSize)
            root = recursiveBST(sampleData,1,levelThreshold,minNodeSize)
            allLearners.append(root)
            
            
            ##calculate error
            error = 0.0
            denominatorOfError =0.0
            predictions =[]
            for rowId in range(lengthTraining):
                denominatorOfError += weights[rowId]
                label = TestingDataClassification(root,trainingData[rowId])
                predictions.append(label)
                if label != trainingData[rowId][-1]:
                    error += weights[rowId]
                    
            #testAccuracy(trainingData,predictions)
            error = float(error/denominatorOfError)
            if(error > 0.5):
                print("Errror:  "+str(error))
                continue
            else:
                validLearners += 1
            
            
            ##calculate normalizationFactor
            alpha = 0.0
            alpha = float((.5)*(np.log((1-error)/error)))
            #print("Alpha: "+str(alpha))
            allAlpha.append(alpha)
            
            
            #calculate  normalizationFactor
            normalizationFactor = 0.0 
            for rowId in range(lengthTraining):
                #label = TestingDataClassification(root,trainingData[rowId]) 
                if predictions[rowId] == trainingData[rowId][-1]:
                    normalizationFactor += (weights[rowId]*np.exp(-1*alpha))
                else:
                    normalizationFactor += (weights[rowId]*np.exp(1*alpha))
                  
            
                
            ##update weights
            for rowId in range(lengthTraining):
                #label = TestingDataClassification(root,trainingData[rowId]) 
                if predictions[rowId] == trainingData[rowId][-1]:
                    weights[rowId] = weights[rowId] *np.exp(-1*alpha)
                else:
                    weights[rowId] = weights[rowId] *np.exp(1*alpha)
            
            ##Normalize Weights
            weights /= normalizationFactor
            
        
        ####### TRAINING COMPLETE FOR A FOLD###################################
        

        ## TESTING DATA
        testingData =[]
        
        for idx in testingDataIdxs[fold]:
            testingData.append(fileData[idx])  
            
        Accuracy,Precision,Recall,FMeasure = testDataBoostAccuracy(testingData,allLearners,allAlpha)
        print(str(fold+1)+" Iteration Complete !!")
        
        Accuracy *= 100
        Precision *= 100
        Recall *= 100
        FMeasure *= 100
        
        
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
