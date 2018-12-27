import math
def readExamples40(file):
    with open(file,'r') as file:
        i=0
        processingTime = []
        weigth = []
        dueDate = []
        for line in file:
            i+=1
            if i==1:
                processingTime.append(line.split())
            elif i==2:
                processingTime[len(processingTime)-1]+= line.split()
            elif i==3:
                weigth.append(line.split())
            elif i==4:
                weigth[len(weigth)-1]+= line.split()
            elif i==5:
                dueDate.append(line.split())
            elif i==6:
                dueDate[len(dueDate)-1]+= line.split()
                i=0
        file.closed
    return processingTime[:-1],weigth,dueDate

def readExamplesGeneric(file,size):
    with open(file,'r') as file:
        numberLinesByProperty = math.ceil(size/20)
        newPropertyIndicator = 0
        parameter = 0
        processingTime = []
        weigth = []
        dueDate = []
        problem = [processingTime,weigth,dueDate]
        for line in file:
            newPropertyIndicator+=1
            if newPropertyIndicator == 1:
                problem[parameter].append(line.split())
            elif newPropertyIndicator < numberLinesByProperty:
                problem[parameter][len(problem[parameter])-1]+= line.split()
            elif newPropertyIndicator == numberLinesByProperty:
                problem[parameter][len(problem[parameter])-1]+= line.split()
                newPropertyIndicator = 0
                parameter +=1
            if parameter >=3:
                parameter=0
        file.closed
        problem[0] = problem[0][:-1]
    return problem

def verifyLen(array,size):
    error = False
    for component in array:
        if len(component)!=size:
            error = True
            break
    if error:
        print("ERROR")
    else:
        print("EVERYTHING IS OK")