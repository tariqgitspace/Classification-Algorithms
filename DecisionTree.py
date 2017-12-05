import numpy as np
import math
import numbers


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
        
    ##If column to use for splitting is number
    if isinstance(attribute_Value, numbers.Number):
        class0Count, class1Count  = continuousDataPartition(records, attributeColumn,attribute_Value)
    else:
        class0Count, class1Count = nominalDataPartition(records, attributeColumn,attribute_Value)

    node = Tree(attributeColumn,attribute_Value, None, None,-1,-1)
    node.left = recursiveBST(class0Count,level+1,levelThreshold,minNodeSize)
    node.right = recursiveBST(class1Count,level+1,levelThreshold,minNodeSize)

    return node


def TestingDataClassification(root,record):

    if root.left == None and root.right==None:
        if (root.class1Count ==-1) and (root.class0Count == -1):
            print("something Wrong in class1Count")
        if(root.class1Count >= root.class0Count):
            return 1
        else:
            return 0

    ##For Numbers class1 belongs on right
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

    """
    print(len(testingDataIdxs[0]))
    print(len(testingDataIdxs[1]))
    print(len(testingDataIdxs[2]))
    print(len(testingDataIdxs[3]))
    print(len(testingDataIdxs[4]))
    print(len(testingDataIdxs[5]))
    print(len(testingDataIdxs[6]))
    print(len(testingDataIdxs[7]))
    print(len(testingDataIdxs[8]))
    print(len(testingDataIdxs[9]))
    """
    trainingDataIdxs =[]
    
    
    for i in range(crossValidation):
        training =[]
        for idx in range(len(records)):
            if not(idx in testingDataIdxs[i]):
                training.append(idx)
        trainingDataIdxs.append(training)
       
   
    """
    print(len(trainingDataIdxs[0]))
    print(len(trainingDataIdxs[1]))
    print(len(trainingDataIdxs[2]))
    print(len(trainingDataIdxs[3]))
    print(len(trainingDataIdxs[4]))
    print(len(trainingDataIdxs[5]))
    print(len(trainingDataIdxs[6]))
    print(len(trainingDataIdxs[7]))
    print(len(trainingDataIdxs[8]))
    print(len(trainingDataIdxs[9]))
    """
    print("KFold Splitting Complete")
    
    return trainingDataIdxs,testingDataIdxs
        
        

if __name__ == '__main__':

    text_file = 'project3_dataset1.txt'
    levelThreshold = 3
    minNodeSize =10
    crossValidation = 10
    
    

    fileData = np.genfromtxt(text_file, dtype=None)
    fileData=np.array(fileData)
    
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
            
        testingData =[]
        for idx in testingDataIdxs[fold]:
            testingData.append(fileData[idx])  

        root = None
        root = recursiveBST(trainingData,1,levelThreshold,minNodeSize)

        actualLabels = []        
        for line in testingData:
            actualLabels.append(int(line[-1]))

        predictedLabels = []
        for testRecord in testingData:
            label = TestingDataClassification(root,testRecord)   
            predictedLabels.append(int(label))
        
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
        print(str(fold+1)+" Iteration Complete !!")
        
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
        
        totalAccuracy.append(Accuracy)
        totalPrecision.append(Precision)
        totalRecall.append(Recall)
        totalFMeasure.append(FMeasure)

    
    print("averageAccuracy  : "+str(np.sum(totalAccuracy)/crossValidation))
    print("averagePrecision  : "+str(np.sum(totalPrecision)/crossValidation))
    print("averageRecall  : "+str(np.sum(totalRecall)/crossValidation))
    print("averageFMeasure  : "+str(np.sum(totalFMeasure)/crossValidation))  