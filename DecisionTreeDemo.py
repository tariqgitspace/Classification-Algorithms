import numpy as np
import math
import numbers
from colorama import Fore,Back,Style

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

            if gain > max_gain:
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
    
def recursiveBST(records):

    Gain, attributeColumn, attribute_Value = MaxGainSplitting(records)

    if Gain == 0:
        return saveResultInLeaf(records,None,None)
        
    ##If column to use for splitting is number
    if isinstance(attribute_Value, numbers.Number):
        class0Count, class1Count  = continuousDataPartition(records, attributeColumn,attribute_Value)
    else:
        class0Count, class1Count = nominalDataPartition(records, attributeColumn,attribute_Value)

    node = Tree(attributeColumn,attribute_Value, None, None,-1,-1)
    node.left = recursiveBST(class0Count)
    node.right = recursiveBST(class1Count)

    return node


def getSpace(counter):
    space = ""
    for i in range(counter):
        space += "     "
    return space



def print_tree(root, tabCounter):

    space =getSpace (tabCounter)
    if (root.left==None) and (root.right==None) :
        if root.class1Count >= root.class0Count: 
            print (Fore.BLACK+space + "   ↳Class 1"+Fore.RESET)
        else:
            print (Fore.BLACK +space + "   ↳Class 0"+Fore.RESET)
        return

    print (space +Back.CYAN + "SPLIT"+Back.RESET+": "+str(root.attribute_Value))

    
    print (Style.BRIGHT+Fore.GREEN +space + '   ↳Left(≠):'+Fore.RESET+Style.RESET_ALL)
    print_tree(root.left, tabCounter+1)
    
    
    print (Style.BRIGHT+Fore.BLUE +space + '   ↳Right(==):'+Fore.RESET+Style.RESET_ALL)
    print_tree(root.right, tabCounter+1)


if __name__ == '__main__':

    text_file = 'project3_dataset4.txt'
    
    
    trainingData = []
    trainingData = np.genfromtxt(text_file, dtype=None)
    trainingData=np.array(trainingData)
    

    root = None
    root = recursiveBST(trainingData)
    print_tree(root,0)

