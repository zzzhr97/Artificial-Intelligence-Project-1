from time import time
from Astar_2 import *
from Astar_3 import *
from GA_2 import *
from GA_3 import *

COST_MATCH = 0
COST_MISMATCH = 5
COST_GAP = 3

QUERY_RANGE_2 = (1, 6)  # location of query string for 2-matching
QUERY_RANGE_3 = (7, 9)  # location of query string for 3-matching

class Test(object):
    def __init__(self):
        self.matchNum = '0'
        self.algorithm = '0'
        self.hFunction = '0'
        self.strs = ""
        self.targetStrs = ""

    def reader(self):
        # input algorithm type
        self.algorithm = input(f"Please input algorithm type (A --> Astar, G --> GA): ")
        while self.algorithm != 'A' and self.algorithm != 'G':
            self.algorithm = input(f"Error input!\nInput algorithm type again (A --> Astar, G --> GA): ")

        # input match number
        self.matchNum = input(f"Please input match number (2 or 3): ")
        while self.matchNum != '2' and self.matchNum != '3':
            self.matchNum = input(f"Error input!\nInput match number again (2 or 3): ")

        # input heuristic function type (only Astar)
        if self.algorithm == 'A':
            self.hFunction = input(f"Please input heuristic function type (1 or 2): ")
            while self.hFunction != '1' and self.hFunction != '2':
                self.hFunction = input(f"Error input!\nInput heuristic function type (1 or 2): ")
        
        # read database
        with open("MSA_database.txt", "r") as file:
            self.strs = [line.strip() for line in file]

        # read query string
        with open("MSA_query.txt", "r") as file:
            targetStrs = [line.strip() for line in file]
            if self.matchNum == '2':
                self.targetStrs = targetStrs[QUERY_RANGE_2[0]:QUERY_RANGE_2[1]]
            else:
                self.targetStrs = targetStrs[QUERY_RANGE_3[0]:QUERY_RANGE_3[1]]

    def solver(self):
        if self.algorithm == 'A':
            if self.matchNum == '2':
                if self.hFunction == '1':
                    AstarSolver = Astar_2_test(self.strs, self.targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP, '1')
                    AstarSolver.selectMinCost()
                else:
                    AstarSolver = Astar_2_test(self.strs, self.targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP, '2')
                    AstarSolver.selectMinCost()
            else:
                if self.hFunction == '1':
                    AstarSolver = Astar_3_test(self.strs, self.targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP, '1')
                    AstarSolver.selectMinCost()
                else:
                    AstarSolver = Astar_3_test(self.strs, self.targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP, '2')
                    AstarSolver.selectMinCost()
        else:
            if self.matchNum == '2':
                GASolver = GA_2_test(self.strs, self.targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP)
                GASolver.selectMinCost()
            else:
                GASolver = GA_3_test(self.strs, self.targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP)
                GASolver.selectMinCost()



if __name__ == "__main__":
    startTime = time()

    test = Test()
    test.reader()
    test.solver()

    print("========================================Output========================================")
    print(f"Total time: {time()-startTime}")