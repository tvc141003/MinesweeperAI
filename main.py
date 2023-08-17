from pysat.card import *
from pysat.solvers import Solver
from itertools import combinations
import sys
import copy
import time
    
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
    
    def convertToMatrix(self, arr):
        result = copy.deepcopy(self.__matrix)
        for i in range(len(arr)):
            x, y = self.remapping(abs(arr[i]))
            if (arr[i] > 0):
                result[x][y] = -1
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
            solver.solve(assumptions=[])
            return solver.get_model()

class BackTracking():
    def __init__(self, matrix):
        self.__matrix = matrix
        self.__dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.__dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    def isInside(self, row, column):
        if (row < 0 or column < 0 or row >= len(self.__matrix) or column >= len( self.__matrix[0])):
            return False
        return True

    def getMine(self, x, y):
        count = 0
        for i in range(len(self.__dx)):
            dx = x + self.__dx[i]
            dy = y + self.__dy[i]
            if (self.isInside(dx, dy) and self.__matrix[dx][dy] == -1): count += 1

        return count


    def isDone(self):
        for i in range(len(self.__matrix)):
            for j in range(len(self.__matrix[i])):
                if (self.__matrix[i][j] > 0):
                    count = self.getMine(i, j)
                    if (count != self.__matrix[i][j]): return False
        return True

    def run(self, x, y):
        if (self.isDone() == True):
            return self.__matrix
        
        if (x >= len(self.__matrix)): return "Cutoff"

        for i in range(len(self.__matrix)):
            for j in range(len(self.__matrix[i])):
                if (i < x or (i == x and j <= y) or self.__matrix[i][j] > 0): continue
                self.__matrix[i][j] = -1
                result = self.run(i, j)
                if (result != "Cutoff"): return result

                self.__matrix[i][j] = 0

        return "Cutoff"

class BruteForce():
    def __init__(self, matrix):
        self.__matrix = matrix
        self.__dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.__dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        
    def isInside(self, row, column):
        if (row < 0 or column < 0 or row >= len(self.__matrix) or column >= len( self.__matrix[0])):
            return False
        return True

    def getMine(self, x, y):
        count = 0
        for i in range(len(self.__dx)):
            dx = x + self.__dx[i]
            dy = y + self.__dy[i]
            if (self.isInside(dx, dy) and self.__matrix[dx][dy] == -1): count += 1

        return count


    def isDone(self):
        for i in range(len(self.__matrix)):
            for j in range(len(self.__matrix[i])):
                if (self.__matrix[i][j] > 0):
                    count = self.getMine(i, j)
                    if (count != self.__matrix[i][j]): return False
        return True

    def run(self, x, y):
        if (self.isDone() == True):
            return self.__matrix
        
        if (x >= len(self.__matrix)): return "Cutoff"

        for i in range(len(self.__matrix)):
            for j in range(len(self.__matrix[i])):
                if (self.__matrix[i][j] != 0): continue
                self.__matrix[i][j] = -1
                result = self.run(i, j)
                if (result != "Cutoff"): return result

                self.__matrix[i][j] = 0

        return "Cutoff"

class newAlgorithms():
    def __init__(self, dataset):
        self.__dataset = dataset
        self.__dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.__dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    def isInside(self, row, column):
        if (row < 0 or column < 0 or row >= len(self.__dataset) or column >= len( self.__dataset[0])):
            return False
        return True

    def getMine(self, x, y):
        count = 0
        for i in range(len(self.__dx)):
            dx = x + self.__dx[i]
            dy = y + self.__dy[i]
            if (self.isInside(dx, dy) and self.__dataset[dx][dy] == -1): count += 1
        return count


    def isDone(self):
        for i in range(len(self.__dataset)):
            for j in range(len(self.__dataset[i])):
                if (self.__dataset[i][j] > 0):
                    count = self.getMine(i, j)
                    if (count != self.__dataset[i][j]): return False
        return True

    def isPositive(self):
        for i in range(len(self.__dataset)):
            for j in range(len(self.__dataset[i])):
                if (self.__dataset[i][j] > 0):
                    return True
        return False

    def run(self):
        result = copy.deepcopy(self.__dataset)
        while (self.isPositive() == True):
            target = []
            for i in range(len(self.__dataset)):
                target.append([0 for _ in range(len(self.__dataset[i]))])

            for i in range(len(self.__dataset)):
                for j in range(len(self.__dataset[i])):
                    if (self.__dataset[i][j] > 0):
                        for k in range(len(self.__dx)):
                            dx = i + self.__dx[k]
                            dy = j + self.__dy[k]
                            if (self.isInside(dx, dy) and self.__dataset[dx][dy] == 0):
                                target[dx][dy] += 1
            priority = []
            for i in range(len(target)):
                for j in range(len(target[i])):
                    if (target[i][j] != 0):
                        priority.append((8 - target[i][j], i, j))
            if (priority == []):
                print("UNSAT")
                return
            priority.sort()
            x = priority[0][1]
            y = priority[0][2]
            self.__dataset[x][y] = -1
            for k in range(len(self.__dx)):
                dx = self.__dx[k] + x
                dy = self.__dy[k] + y
                if (self.isInside(dx, dy) and self.__dataset[dx][dy] > 0):
                    self.__dataset[dx][dy] -= 1

        for i in range(len(self.__dataset)):
            for j in range(len(self.__dataset[i])):
                if (result[i][j] > 0):
                    self.__dataset[i][j] = result[i][j]
        
        if (self.isDone() == True):
            return self.__dataset
        

def readFile(fileName):
    matrix = []
    with open(fileName, 'r') as file:
        line = file.readline()
        while (line != ''):
            items = line.strip().split(', ')
            matrix.append([int(item) for item in items])
            line = file.readline()

        file.close()
    return matrix

def writeFile(fileName, result):
    with open(fileName, 'w') as file:
        for i in range(len(result)):
            line = ", ".join(str(item) if item >= 0 else 'X' for item in result[i])
            if (i < len(result) - 1):
                line = line + "\n"
            file.write(line)

if (__name__ == "__main__"):
    sys.setrecursionlimit(1000000000)
    fileName = input("Input file name: ")
    matrix = readFile(fileName)

    cnfBegin = time.time()
    cnf = CNFs(copy.deepcopy(matrix))
    clauses = cnf.getClauses()
    result = cnf.run()
    cnfEnd = time.time()
    result = Matrix(matrix).convertToMatrix(result)
    fileName = fileName.replace('input', 'output')

    writeFile(fileName, result)

    print("Generate CNFs automatically: ", clauses)
    print("CNFs Algorithms: ",result)
    print("CNFs time running: ", round(cnfEnd - cnfBegin, 10), 's\n')

    newAgl = newAlgorithms(copy.deepcopy(matrix))
    newAglBegin = time.time()
    newResult = newAgl.run()
    newAglEnd = time.time()
    print("4. Algorithms solve minesweeper")
    print("result: ", newResult)
    print("Algorithms time running: ", newAglEnd - newAglBegin, 's\n')

    backTracking = BackTracking(copy.deepcopy(matrix))
    backTrackingBegin = time.time()
    resultBackTracking = backTracking.run(0, -1)
    backTrackingEnd = time.time()
    print("BackTracking")
    print("result: ", resultBackTracking)
    print("Back Tracking time running: ", round(backTrackingEnd - backTrackingBegin, 10), 's\n')


    bruteForce = BruteForce(copy.deepcopy(matrix))
    bruteForceBegin = time.time()
    resultBruteForce = bruteForce.run(0, 0)
    bruteForceEnd = time.time()
    print("Brute-Force")
    print("result: ", resultBackTracking)
    print("rute-Force time running: ", round(bruteForceEnd - bruteForceBegin, 10), 'ms\n')
    print(resultBruteForce)