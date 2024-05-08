import numpy as np
import random
import sys
from time import time

# The number of iterations
ITERATIONS = 200       

# the size of population
SIZE_POPULATION = 50    

# probability
PR_CROSSOVER = 0.6
PR_MUTATION = 0.25
PR_MUTATION_ADD = 0.64   # the pr of adding two '-'s in mutation
PR_MUTATION_DEL = 1 - PR_MUTATION_ADD

# after this times with best cost unchanged, the iteration will be stopped
TIME_CHANGELESS = 50

index = 0

class GA_2(object):
    def __init__(self, X, Y, MATCH, MISMATCH, GAP):
        self.MATCH = MATCH
        self.MISMATCH = MISMATCH
        self.GAP = GAP
        self.X = X
        self.Y = Y
        self.lenX = len(X)
        self.lenY = len(Y)
        self.intArray = np.arange(self.lenX + self.lenY + 5000)
        self.bestIndiv = GA_individual(self, MATCH, MISMATCH, GAP)

    # select SIZE_POPULATION individuals by fitness
    def chooseIndiv(self, costArray):
        # calculate prob by fitness
        minCost = np.min(costArray)
        fitnessArray = 1000 / (costArray - minCost + 50)
        totalFitness = np.sum(fitnessArray)
        cumulateFitnessArray = np.cumsum(fitnessArray / totalFitness)

        # select next offspring by fitness
        nextOffspring = np.searchsorted(cumulateFitnessArray, np.random.rand(SIZE_POPULATION))

        return nextOffspring
    
    # dynamic crossover prob
    def calRealProb(self, time):
        k = 0.36
        return PR_CROSSOVER * (1 - (1 - k) / ITERATIONS * time)
    
    # debug
    def printA(self, population):
        if 0:
            for i, indiv in enumerate(population):
                print(f">>>>>>   The {i}-th cost: {population[i].cost}\tindividual:")
                print(f"\tgeneX: {indiv.geneX}")
                print(f"\tgeneY: {indiv.geneY}")
            print(f"Total cost: {np.sum([_.cost for _ in population])}")
        # sys.exit(0)

    def iterate(self):
        population = np.array([GA_individual(self, self.MATCH, self.MISMATCH, self.GAP) for _ in range(SIZE_POPULATION)])

        # iteration
        changelessCnt = 0
        for time in range(ITERATIONS):

            if changelessCnt >= TIME_CHANGELESS:
                break

            ## selection
            costArray = np.array([population[i].cost for i in range(SIZE_POPULATION)])
            nextOffspring = self.chooseIndiv(costArray)
            population = population[nextOffspring]

            ## crossover
            offspring = []
            for _ in range(SIZE_POPULATION):

                # randomly crossover
                parent1, parent2 = np.random.choice(population, size=2, replace=False)
                REAL_PR_CROSSOVER = self.calRealProb(time)
                if np.random.rand() < REAL_PR_CROSSOVER:
                    offspring += parent1.crossover(parent2)
            
            # merge parents and childs to offspring
            offspring = np.append(population, offspring)

            # select population in the offspring
            costArray = np.array([offspring[i].cost for i in range(len(offspring))])
            nextOffspring = self.chooseIndiv(costArray)
            population = offspring[nextOffspring]

            ## mutation
            for i in range(SIZE_POPULATION):
                if np.random.rand() < PR_MUTATION:
                    population[i] = population[i].mutation()
                population[i] = population[i].delExtraGap()

            ## check population
            fitnessArray = np.array([population[i].fitness for i in range(SIZE_POPULATION)])
            bestIndiv = population[np.argmax(fitnessArray)]
            if self.bestIndiv.fitness > bestIndiv.fitness:
                population[np.random.randint(0, SIZE_POPULATION)] = self.bestIndiv
                changelessCnt += 1
            else:
                self.bestIndiv = bestIndiv
                changelessCnt = 0

            # debug
            if 0:
                print(f">>>>>>   The {time}-th iteration minimum cost: {self.bestIndiv.cost}")

            # debug
            self.printA(population)

        global index
        index += 1
        #print(f"The {index}-th best individual cost: {self.bestIndiv.cost}")
        return self.bestIndiv

class GA_2_test(object):
    def __init__(self, strs, targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP):
        self.COST_MATCH = COST_MATCH
        self.COST_MISMATCH = COST_MISMATCH
        self.COST_GAP = COST_GAP
        self.strs = strs
        self.targetStrs = targetStrs
    
    def selectMinCost(self):
        for idx, targetStr in enumerate(self.targetStrs):
            minCost = sys.maxsize
            bestIndiv = None

            startTime = time()

            for testStr in self.strs:
                solver = GA_2(targetStr, testStr, self.COST_MATCH, self.COST_MISMATCH, self.COST_GAP)
                indiv = solver.iterate()
                if indiv.cost < minCost:
                    minCost = indiv.cost
                    bestIndiv = indiv

            endTime = time()
            
            print("========================================Output========================================")
            print(f"\tThe {idx+1}-th aligned cost: {minCost}\ttime: {endTime-startTime}")
            bestIndiv.getString()

class GA_individual(object):
    def __init__(self, module, MATCH, MISMATCH, GAP, geneX = np.empty(0), geneY = np.empty(0), label = 1):
        self.module = module
        self.MATCH = MATCH
        self.MISMATCH = MISMATCH
        self.GAP = GAP

        self.lenX = module.lenX + len(geneX)
        self.lenY = module.lenY + len(geneY)
        self.lenTotal = self.lenX if self.lenX > self.lenY else self.lenY

        self.geneX = geneX
        self.geneY = geneY

        self.cost = -1
        self.fitness = -1
        self.init(label)    # label == 0: not randomly init ; else randomly init

    def init(self, label):
        if label == 1:
            delta_len = abs(self.lenX - self.lenY)
            if self.lenX < self.lenY:
                self.geneX = np.sort(np.random.choice(np.arange(0, self.lenTotal, dtype=int), delta_len, replace=False))
            elif self.lenX > self.lenY:
                self.geneY = np.sort(np.random.choice(np.arange(0, self.lenTotal, dtype=int), delta_len, replace=False))
        
        self.getCost()
        self.getFitness()

    # given an aligned index (A-index), find the original index
    # if gene[A-index] is char, find O-index of it
    # if gene[A-index] is '-', find O-index of the first char before it
    def find_O_Index(self, index, label):
        if label == 'X':
            return index - np.where(self.geneX <= index)[0].shape[0]
        else:
            return index - np.where(self.geneY <= index)[0].shape[0]

    # given an original index (O-index), find the aligned index
    #   in aligned string
    def find_A_Index(self, index, label):
        if label == 'X':
            return np.setdiff1d(self.module.intArray, self.geneX)[index]
        else:
            return np.setdiff1d(self.module.intArray, self.geneY)[index]

    def getCostChar(self, x, y):
        if x == y:
            return self.MATCH
        elif x == '-' or y == '-':
            return self.GAP
        else:
            return self.MISMATCH

    def getCost(self):
        cost = 0
        x, y, pre_x, pre_y = -1, -1, -1, -1 # must be -1
        for index in range(self.lenTotal):
            x = self.find_O_Index(index, 'X')
            y = self.find_O_Index(index, 'Y')
            x_ch = self.module.X[x] if x > pre_x else '-'
            y_ch = self.module.Y[y] if y > pre_y else '-'
            pre_x = x
            pre_y = y
            cost += self.getCostChar(x_ch, y_ch)

        self.cost = cost

    # the method of calculating fitness
    def getFitness(self):
        self.fitness = 1000 / (self.cost + 1)

    # print aligned strings
    def getString(self):
        aligned_X = ""
        aligned_Y = ""

        x, y, pre_x, pre_y = -1, -1, -1, -1 # must be -1
        for index in range(self.lenTotal):
            x = self.find_O_Index(index, 'X')
            y = self.find_O_Index(index, 'Y')
            x_ch = self.module.X[x] if x > pre_x else '-'
            y_ch = self.module.Y[y] if y > pre_y else '-'
            pre_x = x
            pre_y = y
            aligned_X += x_ch
            aligned_Y += y_ch
        
        print(aligned_X)
        print(aligned_Y)

    def delExtraGap(self):

        i, j, k = 0, 0, 0
        geneX = self.geneX.tolist()
        geneY = self.geneY.tolist()

        while i < len(geneX) and j < len(geneY):
            if geneX[i] == geneY[j]:
                del geneX[i]
                del geneY[j]
                k += 1
            elif geneX[i] < geneY[j]:
                geneX[i] -= k
                i += 1
            else:
                geneY[j] -= k
                j += 1

        geneX = np.array(geneX)
        geneY = np.array(geneY)

        if i < len(geneX):
            geneX[i:] -= k
        if j < len(geneY):
            geneY[j:] -= k
            
        return GA_individual(self.module, self.MATCH, self.MISMATCH, self.GAP, geneX, geneY, 0)

    # crossover
    # the output is the second offspring
    # the first offspring is self
    def crossover(self, other):
        cutPos_X = random.randint(self.find_A_Index(0, 'X'), self.lenTotal - 2)
        cutPos_Y = random.randint(self.find_A_Index(0, 'Y'), self.lenTotal - 2)
        O_index_X = self.find_O_Index(cutPos_X, 'X')
        O_index_Y = self.find_O_Index(cutPos_Y, 'Y')

        # calculate cut position in aligned string
        parent1_X_cutPos = cutPos_X
        parent1_Y_cutPos = cutPos_Y
        parent2_X_cutPos = other.find_A_Index(O_index_X, 'X')
        parent2_Y_cutPos = other.find_A_Index(O_index_Y, 'Y')

        # cut gene
        parent1_X_left = self.geneX[self.geneX <= parent1_X_cutPos]
        parent1_Y_left = self.geneY[self.geneY <= parent1_Y_cutPos]
        parent2_X_left = other.geneX[other.geneX <= parent2_X_cutPos]
        parent2_Y_left = other.geneY[other.geneY <= parent2_Y_cutPos]

        parent1_X_right = self.geneX[self.geneX > parent1_X_cutPos]
        parent1_Y_right = self.geneY[self.geneY > parent1_Y_cutPos]
        parent2_X_right = other.geneX[other.geneX > parent2_X_cutPos]
        parent2_Y_right = other.geneY[other.geneY > parent2_Y_cutPos]

        # complement in each fragment
        parent1_left_totalPos = max(parent1_X_cutPos, parent1_Y_cutPos)
        parent1_right_totalPos = min(parent1_X_cutPos, parent1_Y_cutPos)
        parent2_left_totalPos = max(parent2_X_cutPos, parent2_Y_cutPos)
        parent2_right_totalPos = min(parent2_X_cutPos, parent2_Y_cutPos)

        if parent1_left_totalPos != parent1_right_totalPos:
            if parent1_X_cutPos < parent1_left_totalPos:
                parent1_X_left = np.concatenate((parent1_X_left, np.arange(parent1_X_cutPos + 1, parent1_left_totalPos + 1, dtype=int)))
            if parent1_X_cutPos > parent1_right_totalPos:
                parent1_X_right = np.concatenate((np.arange(parent1_right_totalPos + 1, parent1_X_cutPos + 1, dtype=int), parent1_X_right))
            
            if parent1_Y_cutPos < parent1_left_totalPos:
                parent1_Y_left = np.concatenate((parent1_Y_left, np.arange(parent1_Y_cutPos + 1, parent1_left_totalPos + 1, dtype=int)))
            if parent1_Y_cutPos > parent1_right_totalPos:
                parent1_Y_right = np.concatenate((np.arange(parent1_right_totalPos + 1, parent1_Y_cutPos + 1, dtype=int), parent1_Y_right))

        if parent2_left_totalPos != parent2_right_totalPos:
            if parent2_X_cutPos < parent2_left_totalPos:
                parent2_X_left = np.concatenate((parent2_X_left, np.arange(parent2_X_cutPos + 1, parent2_left_totalPos + 1, dtype=int)))
            if parent2_X_cutPos > parent2_right_totalPos:
                parent2_X_right = np.concatenate((np.arange(parent2_right_totalPos + 1, parent2_X_cutPos + 1, dtype=int), parent2_X_right))
            
            if parent2_Y_cutPos < parent2_left_totalPos:
                parent2_Y_left = np.concatenate((parent2_Y_left, np.arange(parent2_Y_cutPos + 1, parent2_left_totalPos + 1, dtype=int)))
            if parent2_Y_cutPos > parent2_right_totalPos:
                parent2_Y_right = np.concatenate((np.arange(parent2_right_totalPos + 1, parent2_Y_cutPos + 1, dtype=int), parent2_Y_right))

        # Reassemble
        child_1_X = np.concatenate((parent1_X_left, parent2_X_right - parent2_right_totalPos + parent1_left_totalPos))
        child_1_Y = np.concatenate((parent1_Y_left, parent2_Y_right - parent2_right_totalPos + parent1_left_totalPos))
        child_2_X = np.concatenate((parent2_X_left, parent1_X_right - parent1_right_totalPos + parent2_left_totalPos))
        child_2_Y = np.concatenate((parent2_Y_left, parent1_Y_right - parent1_right_totalPos + parent2_left_totalPos))

        child_1 = GA_individual(self.module, self.MATCH, self.MISMATCH, self.GAP, child_1_X, child_1_Y, 0)
        child_2 = GA_individual(self.module, self.MATCH, self.MISMATCH, self.GAP, child_2_X, child_2_Y, 0)
        return [child_1, child_2]
 
    # add or delete a pair of '-'
    def mutation(self):

        if np.random.rand() < PR_MUTATION_ADD:
            # add two '-'s
            all_values = np.arange(self.lenTotal + 1)

            missing_values_X = np.setdiff1d(all_values, self.geneX)
            X_add = np.random.choice(missing_values_X)
            geneX_new = np.sort(np.append(self.geneX, X_add))
            geneX_new[geneX_new > X_add] += 1

            missing_values_Y = np.setdiff1d(all_values, self.geneY)
            Y_add = np.random.choice(missing_values_Y)
            geneY_new = np.sort(np.append(self.geneY, Y_add))
            geneY_new[geneY_new > Y_add] += 1

            return GA_individual(self.module, self.MATCH, self.MISMATCH, self.GAP, geneX_new, geneY_new, 0)
        else:
            # delete two '-'s
            if len(self.geneX) > 0 and len(self.geneY) > 0:
                X_del = np.random.randint(0, len(self.geneX))
                Y_del = np.random.randint(0, len(self.geneY))
                geneX_new = np.delete(self.geneX, X_del)
                geneY_new = np.delete(self.geneY, Y_del)
                geneX_new[X_del:] -= 1
                geneY_new[Y_del:] -= 1

                return GA_individual(self.module, self.MATCH, self.MISMATCH, self.GAP, geneX_new, geneY_new, 0)
            else:
                return self

if __name__ == "__main__":
    s1 = "KJXXJAJKPXKJJXJKPXKJXXJAJKPXKJJXJKPXKJXXJAJKPXKJXXJAJKHXKJXXJAJKPXKJXXJAJKHXKJXX"
    s2 = "ILOTGJJLABWTSTGGONXJMUTUXSJHKWJHCTOQHWGAGIWLZHWPKZULJTZWAKBWHXMIKLZJGLXBPAHOHVOLZWOSJJLPO"
    s3 = "ABCDEOSIDJLKHS"
    s4 = "BCOAIYDKZJHS"
    test = GA_2_test([s1], [s2], 0, 5, 3)
    test.selectMinCost()
