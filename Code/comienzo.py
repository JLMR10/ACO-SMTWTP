import math,operator

class Ant(object):
    
    def __init__(self,numero):
        self.cost = 0.0
        self.benefit = 0.0
        self.solution = []
        self.unifo = 0.0
        self.listaIteracion = []
        self.actualEdge = numero

    def __str__(self):
        return ("esta hormiga tiene un coste de %s y un beneficio de %s con la solución: %s",(self.cost,self.benefit,self.solution))


class Job(object):

    def __init__(self,numero,pT,w,dD):
        self.processingTime = pT
        self.weigth = w
        self.dueDate = dD
        self.nodo = numero
    
def earliestDueDate(jobList):
    return sorted(jobList, key=operator.attrgetter('dueDate'))

def totalTardiness(jobList):
    totalFlowTime = 0
    flowTimeOut = []
    earliness = [0]*len(jobList)
    tardiness = [0]*len(jobList)
    for i,job in enumerate(jobList):
        totalFlowTime += job.processingTime
        flowTimeOut.append(totalFlowTime)
        lateness = totalFlowTime-job.dueDate
        if lateness >= 0:
            tardiness[i] = lateness
        else:
            earliness[i] = lateness
    return sum(tardiness)

problema = [Job(1,26,1,179),Job(2,24,10,183),Job(3,79,9,196),Job(4,46,4,202),Job(5,32,3,192)]



