import math,operator,copy,random

class Ant(object):
    
    def __init__(self,n):
        self.valueSolution = 0.0
        self.initialNode = n
        self.actualNode = self.initialNode
        self.solution = [self.initialNode]
        self.probabilityMatrix = []

    def getSolution(self,q0,pheromonesMatrix,heuristicMatrix,alpha,beta):
        self.solution = [self.initialNode]
        self.actualNode = self.initialNode
        while len(self.solution)!=len(self.probabilityMatrix[0]):
            self.actualNode = self.nextStep(q0,pheromonesMatrix,heuristicMatrix,alpha,beta)
            self.solution.append(self.actualNode)
        # print(self.solution)
    
    def nextStep(self,q0,pheromonesMatrix,heuristicMatrix,alpha,beta):
        q = random.uniform(0,1)
        node = float("-inf")
        probabilityNodes = copy.copy(self.probabilityMatrix[self.actualNode])
        if(q<=q0):
            antValue = float("-inf")
            h = 0 
            for j in range(len(probabilityNodes)):
                newAntValue = (pheromonesMatrix[self.actualNode][j]**alpha)*(heuristicMatrix[self.actualNode][j]**beta)
                if(j not in self.solution):
                    if(newAntValue>antValue):
                        antValue=newAntValue
                        h = j
            node = h
        else:

            for node in self.solution:
                probabilityNodes[node] = 0
            probabilityNodes = [probabilityNodes[i]/sum(probabilityNodes) for i in range(len(probabilityNodes))]
            acc = 0
            for i,probability in enumerate(probabilityNodes):
                acc+=probability
                # print("prob",probability,"Acc",acc,"----",antProbability)
                if acc>q:
                    node = i
                    # print(node)
                    break
            # print("========")
        return node


    def __str__(self):
        return ("esta hormiga tiene un coste de %s y un beneficio de %s con la solución: %s",(self.solution))


class Job(object):

    def __init__(self,numero,pT,w,dD):
        self.processingTime = pT
        self.weight = w
        self.dueDate = dD
        self.node = numero
    
    def __str__(self):
        return ("Job")

class ACO(object):

    def __init__(self,alphaP,betaP,pP,q0,jobListP,nAnts,n):
        self.alpha = alphaP
        self.beta = betaP
        self.p = pP
        self.q0 = q0
        self.jobList = jobListP
        randomNodes = random.sample(range(len(jobListP)),nAnts)
        self.antList = [Ant(x) for x in randomNodes]
        self.generations = n
        # self.i = 0
        self.size = len(self.jobList)
        self.pheromonesMatrix = initializePheromones(self.jobList)
        self.heuristicMatrix = initializeHeuristic(self.jobList)
        self.probabilityMatrix = calculateTrasitionProbability(self.alpha,self.beta,self.jobList,self.heuristicMatrix,self.pheromonesMatrix)

    def execute(self):
        i = 0
        j=0
        h = 0
        solution = []
        
        while (j<self.generations and i<20):
            # print(j)
            thisSolution = self.iterate()
            if thisSolution == solution:
                i+=1
            else:
                i=0
                h+=1
            j+=1
            solution = thisSolution
        #     print(solution)
        # print(h,i,j)
        # print(solution)
        return solution
    
    def iterate(self):
        valueBestSolution = float("inf")
        bestSolution = []
        for ant in self.antList:
            ant.probabilityMatrix = self.probabilityMatrix
            ant.getSolution(self.q0,self.pheromonesMatrix,self.heuristicMatrix,self.alpha,self.beta)
            antJobList = []
            for i in ant.solution:
                antJobList.append(self.jobList[i])
            ant.valueSolution = totalWeightedTardiness(antJobList)
            # print("Hormiga ",ant.initialNode,"==== Solucion ",ant.solution,"Valor",ant.valueSolution)
            if ant.valueSolution<valueBestSolution:
                valueBestSolution = ant.valueSolution
                bestSolution = ant.solution
                # print("Mejor solucion %s y valor %s",bestSolution,valueBestSolution)
        self.updatePheromones(bestSolution,valueBestSolution)
        # print("Fin una iteración")
        self.probabilityMatrix = calculateTrasitionProbability(self.alpha,self.beta,self.jobList,self.heuristicMatrix,self.pheromonesMatrix)
        # if self.i<3:
        #     print(self.probabilityMatrix)

        # self.i+=1
        opt2Solution,opt2Value = two_opt(bestSolution,self.jobList)
        if opt2Value < valueBestSolution:
            bestSolution = opt2Solution
            valueBestSolution = opt2Value
        return (bestSolution,valueBestSolution)

    def updatePheromones(self,solution,valueSolution):
        affectedArcs = []
        size = len(solution)
        for i in range(size-1):
            affectedArcs.append((solution[i],solution[i+1]))
        affectedArcs.append((solution[size-1],solution[0]))
        sizePheromoneMatrix = len(self.pheromonesMatrix)
        for i in range(sizePheromoneMatrix):
            for j in range(sizePheromoneMatrix):
                self.pheromonesMatrix[i][j] = (1-self.p)*self.pheromonesMatrix[i][j]

        for i,j in affectedArcs:
            self.pheromonesMatrix[i][j] = self.pheromonesMatrix[i][j] + self.p*(1/valueSolution)

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
def two_opt(route,jobList):
    best = route
    bestValue = totalWeightedTardiness([jobList[i] for i in best])
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)):
                if j-i == 1: continue # changes nothing, skip then
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                actualValue = totalWeightedTardiness([jobList[i] for i in new_route])
                if actualValue < bestValue:
                    best = new_route
                    bestValue = actualValue
                    improved = True
        route = best
    return (best,bestValue)


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


def calculateTrasitionProbability(alpha,beta,unsortedJobList,heuristicMatrix,pheromonesMatrix):
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


def readExamplesGeneric(file,size):
    with open(file,'r') as file:
        numberLinesByProperty = math.ceil(size/20)
        newPropertyIndicator = 0
        parameter = 0
        processingTime = []
        weigth = []
        dueDate = []
        problem = [processingTime,weigth,dueDate]
        total = []
        for line in file:
            newPropertyIndicator+=1
            if newPropertyIndicator == 1:
                problem[parameter] = (line.split())
            elif newPropertyIndicator < numberLinesByProperty:
                problem[parameter]+= line.split()
            elif newPropertyIndicator == numberLinesByProperty:
                problem[parameter]+= line.split()
                newPropertyIndicator = 0
                parameter +=1
            if parameter >=3:
                total.append(problem)
                processingTime = []
                weigth = []
                dueDate = []
                problem = [[],[],[]]
                parameter=0
        file.closed
        problem[0] = problem[0][:-1]
    return total

#numero,processingTime,weight,dueDate

# def probando(heuristicMatrix,pheromonesMatrix):

# prueba = ACO(1,1,0.1,0,vuelos,2,100)
# ho = Ant(3)
# ho.probabilityMatrix = prueba.probabilityMatrix
# ho.getSolution()
# prueba.execute()

problema = readExamplesGeneric("wt40.txt",40)
def creaJobs(problema):
    sol = []
    i = 0
    for p,w,d in zip(problema[0],problema[1],problema[2]):
        sol.append(Job(i,float(p),float(w),float(d)))
        i+=1
    return sol
datos = creaJobs(problema[0])

# prueba = ACO(1,1,0.1,0,datos,20,500)
# prueba.execute()
def probando():
    solution = float("inf")
    sch = []
    i=0
    while i<10:
        prueba = ACO(1,1,0.1,0.9,datos,20,100)
        x,y = prueba.execute()
        if y<solution:
            solution = y
            print(solution)
            sch = x
        i+=1
    return (sch,solution)
# pruebasol = probando()

