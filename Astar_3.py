from queue import PriorityQueue
from time import time
import sys
import numpy as np
from Astar_2 import *

class AstarNode(object):
    def __init__(self, tree, x=-1, y=-1, z=-1, gValue=-1, parent=None):
        self.tree = tree        # Astar_2 
        self.x = x
        self.y = y
        self.z = z
        self.gValue = gValue
        self.hValue = self.getHValue()
        self.fValue = self.gValue + self.hValue
        self.parent = parent    # used for tracing back
    
    # <
    def __lt__(self, other):
        return self.fValue < other.fValue
    
    # detect whether leaf or not
    def isLeaf(self):
        return self.x == len(self.tree.X) and self.y == len(self.tree.Y) and self.z == len(self.tree.Z)
        
    # calculate g(n)
    def getGValue(self, delta_x, delta_y, delta_z):
        if delta_x == 1 and delta_y == 1:
            delta_GValue_xy = self.tree.getCost(self.x, self.y, "xy")
        else:
            delta_GValue_xy = self.tree.COST_GAP * abs(delta_x - delta_y)

        if delta_x == 1 and delta_z == 1:
            delta_GValue_xz = self.tree.getCost(self.x, self.z, "xz")
        else:
            delta_GValue_xz = self.tree.COST_GAP * abs(delta_x - delta_z)

        if delta_y == 1 and delta_z == 1:
            delta_GValue_yz = self.tree.getCost(self.y, self.z, "yz")
        else:
            delta_GValue_yz = self.tree.COST_GAP * abs(delta_y - delta_z)
        
        return self.gValue + delta_GValue_xy + delta_GValue_xz + delta_GValue_yz
    
    # calculate h(n)
    def getHValue(self):
        if self.tree.hFunction == '1':
            return (abs((len(self.tree.X) - self.x) - (len(self.tree.Y) - self.y)) \
                + abs((len(self.tree.X) - self.x) - (len(self.tree.Z) - self.z)) \
                + abs((len(self.tree.Y) - self.y) - (len(self.tree.Z) - self.z))) \
                * self.tree.COST_GAP
        else:
            delta_x = len(self.tree.X) - self.x
            delta_y = len(self.tree.Y) - self.y
            delta_z = len(self.tree.Z) - self.z

            hValue = 0

            if delta_x > 0 and delta_y > 0:
                if delta_x == delta_y:
                    if self.tree.X[self.x] != self.tree.Y[self.y]:
                        hValue += self.tree.COST_MISMATCH 
                else:
                    if self.tree.X[self.x] != self.tree.Y[self.y]:
                        hValue += self.tree.COST_GAP

            if delta_x > 0 and delta_z > 0:
                if delta_x == delta_z:
                    if self.tree.X[self.x] != self.tree.Z[self.z]:
                        hValue += self.tree.COST_MISMATCH 
                else:
                    if self.tree.X[self.x] != self.tree.Z[self.z]:
                        hValue += self.tree.COST_GAP

            if delta_y > 0 and delta_z > 0:
                if delta_y == delta_z:
                    if self.tree.Y[self.y] != self.tree.Z[self.z]:
                        hValue += self.tree.COST_MISMATCH 
                else:
                    if self.tree.Y[self.y] != self.tree.Z[self.z]:
                        hValue += self.tree.COST_GAP

            return hValue
    
    # extend this node to 3 new nodes
    def extend(self):
        if self.tree.visited[self.x][self.y][self.z] == True:
            return []
        else:
            self.tree.visited[self.x][self.y][self.z] = True
            extendNodes = []
            delta_x = len(self.tree.X) - self.x
            delta_y = len(self.tree.Y) - self.y
            delta_z = len(self.tree.Z) - self.z
            if delta_x == 0:
                if delta_y == 0:
                    # (0,0,x)
                    extendNodes.append(AstarNode(self.tree, self.x, self.y, len(self.tree.Z), self.getGValue(0, 0, delta_z), self))
                else:
                    if delta_z == 0:
                        # (0,x,0)
                        extendNodes.append(AstarNode(self.tree, self.x, len(self.tree.Y), self.z, self.getGValue(0, delta_y, 0), self))
                    else:
                        # (0,x,x)
                        extendNodes.append(AstarNode(self.tree, self.x, self.y, self.z+1, self.getGValue(0, 0, 1), self))
                        extendNodes.append(AstarNode(self.tree, self.x, self.y+1, self.z, self.getGValue(0, 1, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x, self.y+1, self.z+1, self.getGValue(0, 1, 1), self))
            else:
                if delta_y == 0:
                    if delta_z == 0:
                        extendNodes.append(AstarNode(self.tree, len(self.tree.X), self.y, self.z, self.getGValue(delta_x, 0, 0), self))
                    else:
                        # (x,0,x)
                        extendNodes.append(AstarNode(self.tree, self.x, self.y, self.z+1, self.getGValue(0, 0, 1), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y, self.z, self.getGValue(1, 0, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y, self.z+1, self.getGValue(1, 0, 1), self))
                else:
                    if delta_z == 0:
                        # (x,x,0)
                        extendNodes.append(AstarNode(self.tree, self.x, self.y+1, self.z, self.getGValue(0, 1, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y, self.z, self.getGValue(1, 0, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y+1, self.z, self.getGValue(1, 1, 0), self))
                    else:
                        # (x,x,x)
                        extendNodes.append(AstarNode(self.tree, self.x, self.y, self.z+1, self.getGValue(0, 0, 1), self))
                        extendNodes.append(AstarNode(self.tree, self.x, self.y+1, self.z, self.getGValue(0, 1, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y, self.z, self.getGValue(1, 0, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x, self.y+1, self.z+1, self.getGValue(0, 1, 1), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y, self.z+1, self.getGValue(1, 0, 1), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y+1, self.z, self.getGValue(1, 1, 0), self))
                        extendNodes.append(AstarNode(self.tree, self.x+1, self.y+1, self.z+1, self.getGValue(1, 1, 1), self))
            return extendNodes
    
    # print aligned strings
    def getString(self):
        X = self.tree.X
        Y = self.tree.Y
        Z = self.tree.Z
        aligned_X = ""
        aligned_Y = ""
        aligned_Z = ""
        x = self.x
        y = self.y
        z = self.z
        parent = self.parent
        while x != 0 or y != 0 or z != 0:
            delta_x = x - parent.x
            delta_y = y - parent.y
            delta_z = z - parent.z
            if delta_x == 0:
                if delta_y == 0:
                    # (0,0,x)
                    for idx in range(delta_z):
                        z -= 1
                        aligned_X += '-'
                        aligned_Y += '-'
                        aligned_Z += Z[z]
                else:
                    if delta_z == 0:
                        # (0,x,0)
                        for idx in range(delta_y):
                            y -= 1
                            aligned_X += '-'
                            aligned_Y += Y[y]
                            aligned_Z += '-'
                    else:
                        # (0,1,1)
                        y -= 1
                        z -= 1
                        aligned_X += '-'
                        aligned_Y += Y[y]
                        aligned_Z += Z[z]
            else:
                if delta_y == 0:
                    if delta_z == 0:
                        # (x,0,0)
                        for idx in range(delta_x):
                            x -= 1
                            aligned_X += X[x]
                            aligned_Y += '-'
                            aligned_Z += '-'
                    else:
                        # (1,0,1)
                        x -= 1
                        z -= 1
                        aligned_X += X[x]
                        aligned_Y += '-'
                        aligned_Z += Z[z]
                else:
                    if delta_z == 0:
                        # (1,1,0)
                        x -= 1
                        y -= 1
                        aligned_X += X[x]
                        aligned_Y += Y[y]
                        aligned_Z += '-'
                    else:
                        # (1,1,1)
                        x -= 1
                        y -= 1
                        z -= 1
                        aligned_X += X[x]
                        aligned_Y += Y[y]
                        aligned_Z += Z[z]
            parent = parent.parent
        
        aligned_X = aligned_X[::-1]
        aligned_Y = aligned_Y[::-1]
        aligned_Z = aligned_Z[::-1]
        print(aligned_X)
        print(aligned_Y)
        print(aligned_Z)

class Astar_3_test(object):
    def __init__(self, strs, targetStrs, COST_MATCH, COST_MISMATCH, COST_GAP, hFunction):
        self.strs = strs
        self.targetStrs = targetStrs
        self.COST_MATCH = COST_MATCH
        self.COST_MISMATCH = COST_MISMATCH
        self.COST_GAP = COST_GAP
        self.pairCost = {}
        self.hFunction = hFunction

    # calculate pairwise cost
    def calPairCost(self):
        strs = self.strs
        targetStrs = self.targetStrs

        for i in range(len(strs)):
            for j in range(i+1, len(strs)):
                solver = Astar_2(strs[i], strs[j], self.COST_MATCH, self.COST_MISMATCH, self.COST_GAP, self.hFunction)
                node = solver.process()
                self.pairCost[(i, j)] = node.fValue

        for i in range(len(targetStrs)):
            for j in range(len(strs)):
                solver = Astar_2(targetStrs[i], strs[j], self.COST_MATCH, self.COST_MISMATCH, self.COST_GAP, self.hFunction)
                node = solver.process()
                self.pairCost[(i+len(strs), j)] = node.fValue

    def selectMinCost(self):

        # calculate pairwise cost 
        # to eliminate unnecessary matching sequences
        startTime = time()
        self.calPairCost()
        print("========================================Output========================================")
        print(f"Time for calculating pairwise cost: {time() - startTime}")

        for idx, targetStr in enumerate(self.targetStrs):
            minCost = sys.maxsize
            minCostNode = None
            matchNum = 0

            startTime = time()

            for i in range(len(self.strs)):
                for j in range(i+1, len(self.strs)):

                    # eliminate unnecessary matching sequences
                    pairCost_3 = self.pairCost[(i, j)] + \
                        self.pairCost[(idx+len(self.strs), i)] + self.pairCost[(idx+len(self.strs), j)]
                    if pairCost_3 >= minCost:
                        continue
                    matchNum += 1

                    solver = Astar_3(targetStr, self.strs[i], self.strs[j], \
                                      self.COST_MATCH, self.COST_MISMATCH, self.COST_GAP, self.hFunction)
                    node = solver.process()
                    if node.fValue < minCost:
                        minCost = node.fValue
                        minCostNode = node

            endTime = time()
            
            print("========================================Output========================================")
            print(f"\tThe ({idx})-th aligned cost: {minCost}\ttime: {endTime-startTime}        match number: {matchNum}")
            print("aligned pairs:")
            minCostNode.getString()
                

class Astar_3(object):
    def __init__(self, X, Y, Z, COST_MATCH, COST_MISMATCH, COST_GAP, hFunction):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.COST_MATCH = COST_MATCH
        self.COST_MISMATCH = COST_MISMATCH
        self.COST_GAP = COST_GAP
        self.hFunction = hFunction
        self.pq = PriorityQueue()
        self.pq.put(AstarNode(self, 0, 0, 0, 0, None))
        self.visited = np.full((len(X)+1, len(Y)+1, len(Z)+1), False)

    # calculate basic cost
    def getCost(self, u, v, label):
        if label == "xy":
            if self.X[u] == self.Y[v]:
                return self.COST_MATCH
            else:
                return self.COST_MISMATCH
        if label == "xz":
            if self.X[u] == self.Z[v]:
                return self.COST_MATCH
            else:
                return self.COST_MISMATCH
        if label == "yz":
            if self.Y[u] == self.Z[v]:
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
    strs = ["IPJTUMAOULBGAIJHUGBSOWBWLKKBGKPGTGWCIOBGXAJLGTWCBTGLWTKKKYGWPOJL",
            "IWTJBGTJGJTWGBJTPKHAXHAGJJXJJKPJTPJHJHJHJHJHJHJHJHJHKUTJJUWXHGHHGALKLPJTPJPGVXPLBJHH"]
    targetStrs = ["IPZJJLMLTKJULOSTKTJOGLKJOBLTXGKTPLUWWKOMOYJBGALJUKLGLOSVHWBPGWSLUKOBSOPLOOKUKSARPPJ"]
    solver = Astar_3(targetStrs[0], strs[0], strs[1], 0, 5, 3, '1')
    node = solver.process()
    print(node.fValue)
    node.getString()