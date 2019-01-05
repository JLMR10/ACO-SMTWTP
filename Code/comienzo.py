import math,operator,copy,random

class Ant(object):
    
    def __init__(self,n):
        self.valueSolution = 0.0
        self.initialNode = n
        self.actualNode = self.initialNode
        self.solution = [self.initialNode]
        self.probabilityMatrix = []

    def getSolution(self,q0,pheromonesMatrix,heuristicList,jobList,alpha,beta,activatedWeight,aco_h,aco_s,aco_d):
        self.solution = [self.initialNode]
        self.actualNode = self.initialNode
        if aco_d:
            sortedJobList = sorted(jobList, key=operator.attrgetter('dueDate'))
            sortedJobListNoSolution = [job for job in sortedJobList if job not in [jobList[i] for i in self.solution]]

        while len(self.solution)!=len(self.probabilityMatrix[0]):
            if aco_d:
                processed = sum(jobList[i].processingTime for i in self.solution)
                for i,job in enumerate(sortedJobListNoSolution):
                    if i+1 != len(sortedJobListNoSolution):
                        if processed > job.dueDate and processed < sortedJobListNoSolution[i+1].dueDate:
                            break #codition for determinist aco
            self.actualNode = self.nextStep(q0,jobList,pheromonesMatrix,heuristicList,alpha,beta,aco_s)
            self.solution.append(self.actualNode)
            heuristicList = self.updateHeuristic(jobList,self.solution,activatedWeight,aco_h)

        if aco_d:
            for sortedJob in sortedJobListNoSolution:
                for i,job in enumerate(jobList):
                    if sortedJob == job:
                        self.solution.append(i)
                        break
        # print(self.solution)
    
    def nextStep(self,q0,jobList,pheromonesMatrix,heuristicList,alpha,beta,aco_s):
        q = random.uniform(0,1)
        node = float("-inf")
        size = len(heuristicList)
        
        if(q<=q0):
            antValue = float("-inf")
            h = 0 
            for j in range(size):
                if aco_s:
                    newAntValue = (sum(pheromonesMatrix[k][j] for k in range(self.actualNode))**alpha)*(heuristicList[j]**beta)
                else:
                    newAntValue = (pheromonesMatrix[self.actualNode][j]**alpha)*(heuristicList[j]**beta)
                if(j not in self.solution):
                    if(newAntValue>antValue):
                        antValue=newAntValue
                        h = j
            node = h
        else:
            probabilityNodes = []
            for j in range(size):
                if aco_s:
                    numerator = (sum(pheromonesMatrix[k][j] for k in range(self.actualNode))**alpha)*(heuristicList[j]**beta)
                    denominator = sum((sum(pheromonesMatrix[k][h] for k in range(self.actualNode))**alpha)*(heuristicList[h]**beta) for h in range(size))
                else:
                    numerator = (pheromonesMatrix[self.actualNode][j]**alpha)*(heuristicList[j]**beta)
                    denominator = sum((pheromonesMatrix[self.actualNode][h]**alpha)*(heuristicList[h]**beta) for h in range(size))

                probabilityNodes.append(numerator/denominator)
            # print(sum(probabilityNodes))
            acc = 0
            for i,probability in enumerate(probabilityNodes):
                acc+=probability
                if acc>q:
                    # print("???????????")
                    node = i
                    break
        return node

    def updateHeuristic(self,jobList,solution,activatedWeight,aco_h):
        size = len(jobList)
        heuristicList = [0]*size
        # print(solution)
        processed = sum(jobList[i].processingTime for i in solution)
        for i in range(size):
            if i in solution:
                heuristicList[i] = 0
            else:
                if aco_h:
                    if activatedWeight:
                        heuristicList[i] = 1/(mddOp(processed,jobList[i],activatedWeight)-processed*jobList[i].weight)
                    else:
                        heuristicList[i] = 1/(mddOp(processed,jobList[i],activatedWeight)-processed)
                else:
                    heuristicList[i] = 1/mddOp(processed,jobList[i],activatedWeight)
        return heuristicList

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

class ACO_ACS(object):

    def __init__(self,alphaP,betaP,pP,q0,jobListP,nAnts,n,activatedWeightP,activated2OptmP,h,s,d):
        self.aco_h = h
        self.aco_s = s
        self.aco_d = d
        self.alpha = alphaP
        self.beta = betaP
        self.p = pP
        self.q0 = q0
        self.jobList = jobListP
        randomNodes = random.sample(range(len(jobListP)),nAnts)
        self.antList = [Ant(x) for x in randomNodes]
        self.generations = n
        self.activatedWeight = activatedWeightP
        self.activated2Opt = activated2OptmP
        self.size = len(self.jobList)
        self.pheromonesMatrix = initializePheromones(self.jobList,self.activatedWeight)
        self.heuristicMatrix = initializeHeuristic(self.jobList,self.activatedWeight,self.aco_h)
        self.probabilityMatrix = calculateTrasitionProbability(self.alpha,self.beta,self.jobList,self.heuristicMatrix,self.pheromonesMatrix,self.aco_s)
        
        
    def execute(self):
        i=0
        while (i<self.generations):
            solution = self.iterate()
            print("OKA")
            i+=1
        return solution
    
    def iterate(self):
        valueBestSolution = float("inf")
        bestSolution = []
        for ant in self.antList:
            ant.probabilityMatrix = self.probabilityMatrix
            ant.getSolution(self.q0,self.pheromonesMatrix,self.heuristicMatrix[ant.initialNode],self.jobList,self.alpha,self.beta,self.activatedWeight,self.aco_h,self.aco_s,self.aco_d)
            antJobList = []
            for i in ant.solution:
                antJobList.append(self.jobList[i])
            ant.valueSolution = totalTardiness(antJobList,self.activatedWeight)
            if ant.valueSolution<valueBestSolution:
                valueBestSolution = ant.valueSolution
                bestSolution = ant.solution
        
        if self.activated2Opt:
            opt2Solution,opt2Value = two_opt(bestSolution,self.jobList,self.activatedWeight)
            if opt2Value < valueBestSolution:
                bestSolution = opt2Solution
                valueBestSolution = opt2Value

        self.updatePheromones(bestSolution,valueBestSolution)
        self.probabilityMatrix = calculateTrasitionProbability(self.alpha,self.beta,self.jobList,self.heuristicMatrix,self.pheromonesMatrix,self.aco_s)

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
        return "ACO_ACS"


def earliestDueDate(jobList):
    return sorted(jobList, key=operator.attrgetter('dueDate'))


def two_opt(route,jobList,activatedWeight):
    best = route
    bestValue = totalTardiness([jobList[i] for i in best],activatedWeight)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)):
                if j-i == 1: continue # changes nothing, skip then
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                actualValue = totalTardiness([jobList[i] for i in new_route],activatedWeight)
                if actualValue < bestValue:
                    best = new_route
                    bestValue = actualValue
                    improved = True
        route = best
    return (best,bestValue)


def totalTardiness(jobList,activatedWeight):
    totalFlowTime = 0
    flowTimeOut = []
    tardiness = [0]*len(jobList)
    for i,job in enumerate(jobList):
        totalFlowTime += job.processingTime
        flowTimeOut.append(totalFlowTime)
        lateness = totalFlowTime-job.dueDate
        if lateness >= 0:
            if activatedWeight:
                tardiness[i] = lateness*job.weight
            else:
                tardiness[i] = lateness
    return sum(tardiness)

def mddOp(processed, job,activatedWeight):
    value = 0
    if activatedWeight:
        value = max(processed+job.processingTime,job.dueDate)*job.weight
    else:
        value = max(processed+job.processingTime,job.dueDate)
    return value


def initializePheromones(unsortedJobList,activatedWeight):
    sortedJobList = earliestDueDate(unsortedJobList)
    tedd = totalTardiness(sortedJobList,activatedWeight)
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

def initializeHeuristic(unsortedJobList,activatedWeight,aco_h):
    size = len(unsortedJobList)
    heuristicMatrix = [0]*size
    for i in range(size):
        heuristicMatrixJ = [0]*size
        for j in range(size):
            if i==j:
                heuristicMatrixJ[j] = 0
            else:
                if aco_h:
                    heuristicMatrixJ[j] = 1/(mddOp(unsortedJobList[i].processingTime,unsortedJobList[j],activatedWeight)-unsortedJobList[i].processingTime)
                else:
                    heuristicMatrixJ[j] = 1/mddOp(unsortedJobList[i].processingTime,unsortedJobList[j],activatedWeight)
        heuristicMatrix[i] = heuristicMatrixJ
    return heuristicMatrix



def calculateTrasitionProbability(alpha,beta,unsortedJobList,heuristicMatrix,pheromonesMatrix,aco_s):
    size = len(unsortedJobList)
    probabilityMatrix = []
    for i in range(size):
        probabilityMatrixJ = []
        for j in range(size):
            if aco_s:
                numerator = (sum(pheromonesMatrix[k][j] for k in range(i))**alpha)*(heuristicMatrix[i][j]**beta)
                denominator = sum((sum(pheromonesMatrix[k][h] for k in range(i))**alpha)*(heuristicMatrix[i][h]**beta) for h in range(size))
            else:
                numerator = (pheromonesMatrix[i][j]**alpha)*(heuristicMatrix[i][j]**beta)
                denominator = sum((pheromonesMatrix[i][h]**alpha)*(heuristicMatrix[i][h]**beta) for h in range(size))
            if i!=j:
                print(numerator)
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

# prueba = ACO_ACS(1,1,0.1,0,vuelos,2,100)
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
#(alphaP,betaP,pP,q0,jobListP,nAnts,n,activatedWeightP,activated2OptmP,h,s,d)
prueba = ACO_ACS(1,1,0.1,0,datos,20,100,True,True,False,True,False)
prueba.execute()
# def probando():
#     solution = float("inf")
#     sch = []
#     i=0
#     while i<10:
#         prueba = ACO_ACS(1,1,0.1,0.9,datos,20,100)
#         x,y = prueba.execute()
#         if y<solution:
#             solution = y
#             print(solution)
#             sch = x
#         i+=1
#     return (sch,solution)
# pruebasol = probando()

