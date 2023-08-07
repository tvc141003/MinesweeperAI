from pysat.card import *
from pysat.solvers import Solver
from itertools import combinations

    
class Matrix():
    def __init__(self, matrix):
        self.__matrix = matrix
        self.__rows = len(matrix)
        self.__columns = len(matrix[0])
        self.__dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.__dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    def rows(self):
        return self.__rows

    def columns(self):
        return self.__columns

    def reShape(self):
        result = []
        for i in range(self.__rows):
            for j in range(self.__columns):
                result.append(self.__matrix[i][j])
        return result

    def mapping(self, row, column):
        return self.__columns * row + column + 1
    
    def remapping(self, index):
        index = index - 1
        return [int(index / self.__columns), index % self.__columns]
    
    def isInside(self, row, column):
        if (row < 0 or column < 0 or row >= self.__rows or column >= self.__columns):
            return False
        return True

    def getNeighbors(self, row, column):
        result = []
        for i in range(8):
            dx = row + self.__dx[i]
            dy = column + self.__dy[i]
            if (self.isInside(dx, dy) == True and self.__matrix[dx][dy] == 0):
                result.append(self.mapping(dx, dy))
        return result

        
class CNFs():
    def __init__(self, matrix):
        self.__matrix = matrix
        self.__clauses = []

    def getPositiveClauses(self, value, neighbors):
        result = []
        for comb in combinations(neighbors, len(neighbors) - value + 1):
            clause = []
            for i in range(len(comb)):
                clause.append(comb[i])
            result.append(clause)
        return result

    def getNegativeClauses(self, value, neighbors):
        result = []

        for comb in combinations(neighbors, value + 1):
            clause = []
            for i in range(len(comb)):
                clause.append(-comb[i])

            result.append(clause)
        return result
    
    def getClauses(self):
        for i in range(len(self.__matrix)):
            for j in range(len(self.__matrix[i])):
                if (self.__matrix[i][j] > 0):
                    value = self.__matrix[i][j]
                    neighbors = Matrix(self.__matrix).getNeighbors(i, j)

                    positiveClauses = self.getPositiveClauses(value, neighbors)
                    negativeClauses = self.getNegativeClauses(value, neighbors)

                    self.__clauses += positiveClauses + negativeClauses


        return self.__clauses

    def run(self):
        cnf = CNF(from_clauses=self.__clauses)

        with Solver(bootstrap_with=cnf) as solver:
            if (solver.solve(assumptions=[])):
                print("SAT")
            else:
                print("UNSAT")
               
            return solver.get_model()

if (__name__ == "__main__"):
    input = [[3, 0, 2, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    
    cnf = CNFs(input)
    clauses = cnf.getClauses()
    result = cnf.run()
    print(result)
