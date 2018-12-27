class Ant(object):
    
    def __init__(self,numero):
        self.cost = 0.0
        self.benefit = 0.0
        self.solution = []
        self.unifo = 0.0
        self.listaIteracion = []
        self.actualEdge = numero

    def getActualEdge(self):
        return self.actualEdge

    def getSolution(self):
        return self.solution

    def getCost(self):
        return self.cost
    
    def getBenefit(self):
        return self.benefit
    
    def getUnifo(self):
        return self.unifo
    
    def getListaIteracion(self):
        return self.listaIteracion

    def __str__(self):
        return ("esta hormiga tiene un coste de %s y un beneficion de %s con la solucion: %s",(self.cost,self.benefit,self.solution))
        