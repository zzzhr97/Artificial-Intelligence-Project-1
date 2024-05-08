from queue import PriorityQueue
from time import time
import sys
import numpy as np

class AstarNode(object):
    def __init__(self, tree, x=-1, y=-1, gValue=-1, parent=None):
        self.tree = tree        # Astar_2 
        self.x = x
        self.y = y
        self.gValue = gValue
        self.hValue = self.getHValue()
        self.fValue = self.gValue + self.hValue
        self.parent = parent    # used for tracing back
    
    # <
    def __lt__(self, other):
        return self.fValue < other.fValue
    
    # detect whether leaf or not
    def isLeaf(self):
        return self.x == len(self.tree.X) and self.y == len(self.tree.Y)
    
    # calculate g(n)
    def getGValue(self, delta_x, delta_y):
        if delta_x == 1 and delta_y == 1:
            return self.gValue + self.tree.getCost(self.x, self.y)
        else:
            return self.gValue + self.tree.COST_GAP * abs(delta_x - delta_y)
    
    # calculate h(n)
    def getHValue(self):
        if self.tree.hFunction == '1':
            return abs((len(self.tree.X) - self.x) - (len(self.tree.Y) - self.y)) * self.tree.COST_GAP
        else:
            delta_x = len(self.tree.X) - self.x
            delta_y = len(self.tree.Y) - self.y

            hValue = 0

            if delta_x > 0 and delta_y > 0:
                if delta_x == delta_y:
                    if self.tree.X[self.x] != self.tree.Y[self.y]:
                        hValue = self.tree.COST_MISMATCH 
                else:
                    if self.tree.X[self.x] != self.tree.Y[self.y]:
                        hValue = self.tree.COST_GAP

            return hValue
    
    # extend this node to 3 new nodes
    def extend(self):
        if self.tree.visited[self.x][self.y] == True:
            return []
        else:
            self.tree.visited[self.x][self.y] = True
            extendNodes = []
            if len(self.tree.X) == self.x:
                delta_y = len(self.tree.Y) - self.y
                extendNodes.append(AstarNode(self.tree, self.x, len(self.tree.Y), self.getGValue(0, delta_y), self))
            elif len(self.tree.Y) == self.y:
                delta_x = len(self.tree.X) - self.x
                extendNodes.append(AstarNode(self.tree, len(self.tree.X), self.y, self.getGValue(delta_x, 0), self))
            else:
                extendNodes.append(AstarNode(self.tree, self.x+1, self.y+1, self.getGValue(1, 1), self))
                extendNodes.append(AstarNode(self.tree, self.x+1, self.y, self.getGValue(1, 0), self))
                extendNodes.append(AstarNode(self.tree, self.x, self.y+1, self.getGValue(0, 1), self))
            return extendNodes
    
    # print aligned strings
    def getString(self):
        X = self.tree.X
        Y = self.tree.Y
        aligned_X = ""
        aligned_Y = ""
        x = self.x
        y = self.y
        parent = self.parent
        while x != 0 or y != 0:
            delta_x = x - parent.x
            delta_y = y - parent.y
            if delta_x == 0:
                for idx in range(delta_y):
                    y -= 1
                    aligned_Y += Y[y]
                    aligned_X += '-'
            elif delta_y == 0:
                for idx in range(delta_x):
                    x -= 1
                    aligned_X += X[x]
                    aligned_Y += '-'
            else:
                x -= 1
                y -= 1
                aligned_X += X[x]
                aligned_Y += Y[y]
            parent = parent.parent
        
        aligned_X = aligned_X[::-1]
        aligned_Y = aligned_Y[::-1]
        print(aligned_X)
        print(aligned_Y)

class Astar_2_test(object):
    def __init__(self, strs, targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP, hFunction):
        self.strs = strs
        self.targetStrs = targetStrs
        self.COST_MATCH = COST_MATCH
        self.COST_MISMATCH = COST_MISMATCH
        self.COST_GAP = COST_GAP
        self.hFunction = hFunction

    def selectMinCost(self):
        for idx, targetStr in enumerate(self.targetStrs):
            minCost = sys.maxsize
            minCostNode = None

            startTime = time()

            for testStr in self.strs:
                solver = Astar_2(targetStr, testStr, self.COST_MATCH, self.COST_MISMATCH, self.COST_GAP, self.hFunction)
                node = solver.process()
                if node.fValue < minCost:
                    minCost = node.fValue
                    minCostNode = node

            endTime = time()
            
            print("========================================Output========================================")
            print(f"\tThe {idx+1}-th aligned cost: {minCost}\ttime: {endTime-startTime}")
            minCostNode.getString()
                

class Astar_2(object):
    def __init__(self, X, Y, COST_MATCH, COST_MISMATCH, COST_GAP, hFunction):
        self.X = X
        self.Y = Y
        self.COST_MATCH = COST_MATCH
        self.COST_MISMATCH = COST_MISMATCH
        self.COST_GAP = COST_GAP
        self.hFunction = hFunction
        self.pq = PriorityQueue()
        self.pq.put(AstarNode(self, 0, 0, 0, None))
        self.visited = np.full((len(X)+1, len(Y)+1), False)

    # calculate basic cost
    def getCost(self, x, y):
        if self.X[x] == self.Y[y]:
            return self.COST_MATCH
        else:
            return self.COST_MISMATCH
        
    # run A* algorithm
    def process(self):
        node = self.pq.get()
        while not node.isLeaf():
            extendNodes = node.extend()
            for extendNode in extendNodes:
                self.pq.put(extendNode)
            node = self.pq.get()
        return node

if __name__ == "__main__":
    pass
    """strs = ["IPZJILPLTHOUKOSTMPJOGLKJFBLPMJKJTLBWWKOMOYJBJPLJSKLFLOSZHGVPGJSLWGXBHOHLVWUKWXXTWJ"]
    targetStrs = ["KJXXJAJKPXKJJXJKPXKJXXJAJKPXKJJXJKPXKJXXJAJKPXKJXXJAJKHXKJXXJAJKPXKJXXJAJKHXKJXX"]
    x = Astar_2_test(strs, targetStrs, 0, 5, 3)
    x.selectMinCost()"""

