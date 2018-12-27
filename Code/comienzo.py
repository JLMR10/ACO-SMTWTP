class Ant(object):
    
    def __init__(self,numero):
        self.cost = 0.0
        self.benefit = 0.0
        self.solution = []
        self.unifo = 0.0
        self.listaIteracion = []
        self.actualEdge = numero

    def actualEdge(self):
        return self.actualEdge

    def solution(self):
        return self.solucion

    def cost(self):
        return self.cost
    
    def benefit(self):
        return self.benefit
    
    def unifo(self):
        return self.unifo
    
    def listaIteracion(self):
        return self.listaIteracion

    def __str__(self):
        return "esta hormiga tiene un coste de %s y un beneficion de %s con la solucion: %s" (self.cost,self.benefit,self.solution)
        