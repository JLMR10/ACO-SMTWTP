import math,operator,copy,random
import networkx as nx

class Ant(object):
    
    def __init__(self,n):
        self.cost = 0.0
        self.benefit = 0.0
        self.actualNode = n
        self.solution = [self.actualNode]
        self.unifo = 0.0
        self.iterationList = []
        self.processedTime = 0
        self.probabilityMatrix = []

    def getSolution(self):
        while len(self.solution)!=len(self.probabilityMatrix[0]):
            self.actualNode = self.nextStep()
            self.solution.append(self.actualNode)
        
    
    def nextStep(self):
        antProbability = random.uniform(0,1)
        probabilityNodes = copy.copy(self.probabilityMatrix[self.actualNode])

        for node in self.solution:
            probabilityNodes[node] = 0

        probabilityNodes = [probabilityNodes[i]/sum(probabilityNodes) for i in range(len(probabilityNodes))]
        
        acc = 0
        node = 0
        for i,probability in enumerate(probabilityNodes):
            acc+=probability
            if acc>antProbability:
                node = i
                break

        return node


    def __str__(self):
        return ("esta hormiga tiene un coste de %s y un beneficio de %s con la solución: %s",(self.cost,self.benefit,self.solution))


class Job(object):

    def __init__(self,numero,pT,w,dD):
        self.processingTime = pT
        self.weight = w
        self.dueDate = dD
        self.node = numero
    
    def __str__(self):
        return ("Job")

class ACO(object):

    def __init__(self,alphaP,betaP,jobListP,antListP):
        self.alpha = alphaP
        self.beta = betaP
        self.jobList = jobListP
        self.antList = antListP
        self.size = len(self.jobList)
        self.graph = nx.complete_graph(self.size)
        self.pheromonesMatrix = initializePheromones(self.jobList)
        self.heuristicMatrix = initializeHeuristic(self.jobList)
        self.probabilityMatrix = initializeTrasitionProbability(self.alpha,self.beta,self.jobList,self.heuristicMatrix,self.pheromonesMatrix)


    def __str__(self):
        return "ACO"

def earliestDueDate(jobList):
    return sorted(jobList, key=operator.attrgetter('dueDate'))

# def totalTardiness(jobList):
#     totalFlowTime = 0
#     flowTimeOut = []
#     earliness = [0]*len(jobList)
#     tardiness = [0]*len(jobList)
#     for i,job in enumerate(jobList):
#         totalFlowTime += job.processingTime
#         flowTimeOut.append(totalFlowTime)
#         lateness = totalFlowTime-job.dueDate
#         if lateness >= 0:
#             tardiness[i] = lateness
#         else:
#             earliness[i] = lateness
#     return sum(tardiness)

def totalWeightedTardiness(jobList):
    totalFlowTime = 0
    flowTimeOut = []
    earliness = [0]*len(jobList)
    tardiness = [0]*len(jobList)
    for i,job in enumerate(jobList):
        totalFlowTime += job.processingTime
        flowTimeOut.append(totalFlowTime)
        lateness = totalFlowTime-job.dueDate
        if lateness >= 0:
            tardiness[i] = lateness*job.weight
        else:
            earliness[i] = lateness*job.weight
    return sum(tardiness)

def mddOp(processed, job):
    return max(processed+job.processingTime,job.dueDate)*job.weight

def mddSort(jobList):
    unsortedJobList = copy.copy(jobList)
    sortedJobList = []
    processed = 0
    while unsortedJobList:
        bestJob = unsortedJobList[0]
        bestMdd = mddOp(processed,bestJob)
        for job in unsortedJobList:
            mdd = mddOp(processed,job)
            if mdd < bestMdd:
                bestMdd = mdd
                bestJob = job
        sortedJobList.append(bestJob)
        unsortedJobList.remove(bestJob)
        processed+=bestJob.processingTime
    return sortedJobList

def initializePheromones(unsortedJobList):
    sortedJobList = earliestDueDate(unsortedJobList)
    tedd = totalWeightedTardiness(sortedJobList)
    size = len(sortedJobList)
    t0 = 1/(len(unsortedJobList)*tedd)
    matrix = []
    for i in range(size):
        matrixJ = []
        for j in range(size):
            if i==j:
                matrixJ.append(0)
            else:
                matrixJ.append(t0)
        matrix.append(matrixJ)
    return matrix

def initializeHeuristic(unsortedJobList):
    size = len(unsortedJobList)
    # return [[mddOp(0,job) for job in unsortedJobList] for i in range(size)]
    heuristicMatrix = [0]*size
    for i in range(size):
        heuristicMatrixJ = [0]*size
        for j in range(size):
            if i==j:
                heuristicMatrixJ[j] = 0
            else:
                heuristicMatrixJ[j] = 1/mddOp(unsortedJobList[i].processingTime,unsortedJobList[j])
        heuristicMatrix[i] = heuristicMatrixJ
    return heuristicMatrix


def initializeTrasitionProbability(alpha,beta,unsortedJobList,heuristicMatrix,pheromonesMatrix):
    size = len(unsortedJobList)
    probabilityMatrix = []
    for i in range(size):
        probabilityMatrixJ = []
        for j in range(size):
            numerator = (pheromonesMatrix[i][j]**alpha)*(heuristicMatrix[i][j]**beta)
            denominator = sum((pheromonesMatrix[i][h]**alpha)*(heuristicMatrix[i][h]**beta) for h in range(size))
            probabilityMatrixJ.append(numerator/denominator)
        probabilityMatrix.append(probabilityMatrixJ)
    return probabilityMatrix

problema = [Job(1,26,1,179),Job(2,24,10,183),Job(3,79,9,196),Job(4,46,4,202),Job(5,32,3,192)]
vuelos = [Job(0,8,2,12),Job(1,9,5,18),Job(2,5,2,10),Job(3,6,8,14)]
#numero,processingTime,weight,dueDate

# def probando(heuristicMatrix,pheromonesMatrix):
