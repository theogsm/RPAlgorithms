import itertools
import random
import numpy as np
import copy
import pickle
import csv

class Graph:

    def __init__(self, A_G):
        self.A_G = A_G
        self.M_G = None
        self.B_G = None
        self.sigmaPartition = None
        self.rightResolvers = None
        self.period = None
        self.isBunchy = None
        self.isAlmostBunchy = None
        self.O_B_G = None
        self.B_GCongClasses = None

    def constructM_G(self):
        partition_0 = [np.ones(len(self.A_G), dtype=int)] # partitions are vectors of ones and zeros indexed by states of G. starts with partition_0=V(G)
        while True:
            vtxToPar0 = np.matmul(self.A_G, np.transpose(partition_0)) #vtxToPar0[i][j] = num of edges from state i into part j of partition_0
            parVecs = np.unique(vtxToPar0, axis=0) # set of unique row vectors in vtxToPar0
            lexOrder = np.lexsort(np.transpose(parVecs)) # lexographic order of parVecs stored as a list that represents a permutation
            partition_1 = np.zeros((len(parVecs), len(self.A_G)), dtype=int)
            k=0
            for part in partition_0: # runs over parts in partition_0 in order
                for i in range(0,len(parVecs)): # adds states from part in partition_0 according to place that that their corresponding vector appears in the lex order of parVecs
                    nextPar_1 = False
                    for j in range(0, len(part)):
                        if part[j] == 1 and np.array_equal(parVecs[lexOrder[i]],vtxToPar0[j]):
                            partition_1[k][j] = 1
                            nextPar_1 = True
                    if nextPar_1:
                        k = k+1
            if np.array_equal(partition_0,partition_1): # if partition_0 didn't change, make the adjMtx of M_G and set M_G
                adjMtx = []
                for i in range(0,len(partition_0)):
                    vtxRep = np.where(partition_0[i] == 1)[0][0] # any vtx in the ith part of partition_0
                    adjMtx.append(list(vtxToPar0[vtxRep]))
                self.M_G = Graph(adjMtx)
                self.sigmaPartition = partition_0
                return
            partition_0 = partition_1


    def permuteMatrix(self): # permutes adjMtx. for testing purposes
        if self.M_G is None:
            prmtn = np.arange(len(self.A_G))
            random.shuffle(prmtn)
            self.A_G = [[self.A_G[prmtn[i]][prmtn[j]] for j in range(0,len(self.A_G))] for i in range(0,len(self.A_G))]

    def isMinimalRR(self):
        if self.M_G is None:
            self.constructM_G()
        return len(self.A_G) == len(self.M_G.A_G)

    def getPeriod(self):
        if self.period is None:
            cycleLengths = []
            prodMtx = self.A_G
            for i in range(1,len(self.A_G)): #checks for cycles of length i
                for vtx in range(0,len(self.A_G)):
                    if prodMtx[vtx][vtx] > 0:
                        cycleLengths.append(i)
                prodMtx = np.matmul(prodMtx,self.A_G)
            for vtx in range(0, len(self.A_G)):
                if prodMtx[vtx][vtx] > 0:
                    cycleLengths.append(len(self.A_G))
            self.period = np.gcd.reduce(cycleLengths)
        return self.period

    def getIsBunchy(self):
        if self.isBunchy is None:
            self.setBunchiness()
        return self.isBunchy

    def getIsAlmostBunchy(self):
        if self.isAlmostBunchy is None:
            self.setBunchiness()
        return self.isAlmostBunchy

    def setBunchiness(self):
        if self.M_G is None:
            self.constructM_G()
        self.isBunchy = True
        self.isAlmostBunchy = True
        for fiber1 in self.sigmaPartition:
            for fiber2 in self.sigmaPartition:# for each pair of fibers
                offenders = 0
                for vtx in range(0,len(self.A_G)): # count offending vxts in fiber 1
                    if fiber1[vtx] == 1:
                        followersInFiber2 = 0
                        for follower in range(0,len(self.A_G)):
                            if self.A_G[vtx][follower] > 0 and fiber2[follower] == 1:
                                followersInFiber2 += 1
                        if followersInFiber2 > 1: # vtx is an offender if it has more than 1 follower in fiber 2
                            offenders += 1
                            self.isBunchy = False
                            if offenders > 1:
                                self.isAlmostBunchy = False
                                return
        return

    def constructB_G(self):
        if self.M_G is None:
            self.constructM_G()
        sigmaT = np.transpose(self.sigmaPartition)
        auxAdjMtx = np.zeros((len(self.A_G)**2,len(self.A_G)**2), dtype=int) # adj matrix on pairs of vtxs. Add an edge (a,b)->(c,d) iff there are edges a->b and c->d
        for init1 in range(0,len(self.A_G)):
            for init2 in range(0, len(self.A_G)):
                for term1 in range(0, len(self.A_G)):
                    for term2 in range(0, len(self.A_G)):
                        if np.array_equal(sigmaT[init1],sigmaT[init2]) and np.array_equal(sigmaT[term1],sigmaT[term2]):
                            if self.A_G[init1][term1] > 0 and self.A_G[init2][term2] > 0:
                                auxAdjMtx[self.mtxIndex([init1,init2])][self.mtxIndex([term1,term2])] = 1
        auxGraph = Graph(auxAdjMtx)
        diagonal = np.zeros(len(auxGraph.A_G), dtype=int)
        for i in range(0,len(self.A_G)):
            diagonal[self.mtxIndex([i,i])] = 1
        pathFromDiagonal = vtxsWithPathFromSet(auxGraph,diagonal) # pairs of vtxs with paths of the same length from the same vtx respecting the fibers into M(G)
        congClasses = list(np.zeros((len(self.A_G), len(self.A_G)), dtype=int))
        for i in range(0,len(pathFromDiagonal)): # initialize congruence classes with pairs from pathFromDiagonal
            if pathFromDiagonal[i] == 1:
                congClasses[self.pairIndex(i)[0]][self.pairIndex(i)[1]] = 1
        for vtx in range(0, len(self.A_G)): # consolidate congClasses to form the transitive closure. Note that pathFromDiagonal is symmetric by construction
            vtxIndex = None
            part = 0
            while part < len(congClasses):
                if congClasses[part][vtx] == 1:
                    if vtxIndex is None:
                        vtxIndex = part
                        part += 1
                    else:
                        congClasses[vtxIndex] = [congClasses[vtxIndex][i] or congClasses[part][i] for i in range(0,len(self.A_G))]
                        del congClasses[part]
                else:
                    part += 1

        #print("conj classes:")
        #print(np.array(conjClasses))
        self.B_GCongClasses = congClasses
        self.B_G = generateQuotientGraph(self,congClasses) # Sets B(G) to be the quotient of G over congClasses

    def constructO_B_G(self): # constructs O(B(G)). Assuming the bunchy factor conjecture, O(B(G))=O(G)
        if self.B_G is None:
            self.constructB_G()
        rightRes = generateRightResolver(self.B_G,self.M_G)
        rightRes.constructStabClasses()
        self.O_B_G = generateQuotientGraph(self.B_G, rightRes.stabClasses) # Stability classes from B(G) to M(G) are unique, hence O(B(G)) is the quotient of B(G) over this congruence


    def pairIndex(self, mtxIndex):
        return [int(mtxIndex/len(self.A_G)),mtxIndex % len(self.A_G)]

    def mtxIndex(self, pairIndex): # matrix index for graphs with V=V(G)xV(G) ordered lex.
        return pairIndex[0]*len(self.A_G)+pairIndex[1]

def generateGraphs(M, minN, maxN):
    n = len(M.A_G)
    graphs = []
    for N in range(max(n,minN), maxN+1):
        partitions = generatePartitions(N,n, False) # possible num of vtxs in each fiber, indexed by states in M_G
        #print("partitions:", partitions)
        for partn in partitions:
            #print("partition:", partn)
            possibleParttoPart = [[generatePartitions(M.A_G[i][j], partn[j], True) for j in range(0,n)] for i in range(0,n)] # possible out edges from a vtx in one fiber to the vtxs in another fiber
            adjMtxs = [[]]
            for i in range(0,n):
                possiblePartOut = [[]]
                for j in range(0,n): # builds the possible rows of the adj matrix for a vtx in the ith fiber based on the possible segments into each fiber
                    possiblePartOut = list(itertools.product(possiblePartOut,possibleParttoPart[i][j]))
                    for k in range(0,len(possiblePartOut)):
                        possiblePartOut[k] = list(itertools.chain.from_iterable(possiblePartOut[k]))
                #print(possiblePartOut)
                for j in range(0,partn[i]): # builds the possible ith fiber portion of the possible adj matrices. no mtxs are complete until i runs through every fiber
                    adjMtxs = list(itertools.product(adjMtxs,possiblePartOut))
                    for k in range(0,len(adjMtxs)):
                        adjMtxs[k] = [*adjMtxs[k][0], adjMtxs[k][1]]
            graphs = [*graphs, *[Graph(adjMtxs[i]) for i in range(0, len(adjMtxs))]] # adds the graphs for the "partn" partition to the list of graphs
    return graphs

def generateRandomGraphByOutDeg(maxOutDegree, n):
    for m in range(0,10000):
        adjMtx = []
        for i in range(0,n):
            adjMtx.append(randomPartition(np.random.randint(0,maxOutDegree+1),n, True)) # each row is a random list that sums to a random degree <= maxOutDegree
        if (isIrreducibile(adjMtx)):
            return Graph(adjMtx)
    print("Too few irreducible graphs. maxOutDegree =", maxOutDegree, ", n =", n)
    return None

def generateRandomGraph(M_G,n): # returns a ramdom graph G with given M(G) and order n
    for m in range(0,100000):
        adjMtx = []
        vtxPartition = randomPartition(n,len(M_G.A_G),False) # random partition into fibers
        for i in range(0,len(vtxPartition)):
            for j in range(0,vtxPartition[i]):
                outEdges = []
                for k in range(0, len(vtxPartition)):
                    outEdges = [*outEdges, *randomPartition(M_G.A_G[i][k],vtxPartition[k], True)] # builds random out edges based on possible edges into each part
                adjMtx.append(outEdges)
        if isIrreducibile(adjMtx):
            return Graph(adjMtx)
    print("Too few irreducible graphs. M_G =", M_G.A_G, ", n =", n)
    return None

def generateBiResExtension(G, vtxsInFiber):
    for tryNum in range(0, 10000):
        adjMtx = np.zeros((len(G.A_G)*vtxsInFiber, len(G.A_G)*vtxsInFiber), dtype=int)
        for initFiber in range(0,len(G.A_G)):
            for termFiber in range(0,len(G.A_G)):
                for edge in range(0,G.A_G[initFiber][termFiber]):
                    biject = np.arange(vtxsInFiber) # an extension is bi-resolving iff lifting an edge gives a bijection between fibers.
                    random.shuffle(biject)
                    for i in range(0,vtxsInFiber):
                        adjMtx[initFiber*vtxsInFiber+i][termFiber*vtxsInFiber+biject[i]] += 1
        if isIrreducibile(adjMtx):
            return Graph(adjMtx)
    print("no irreducible bi-resolving extension found")

def existsBiResToM_G(graph):
    if graph.M_G is None:
        graph.constructM_G()
    vtxToFiber = np.matmul(graph.A_G,np.transpose(graph.sigmaPartition))
    fiberToVtx = np.matmul(graph.sigmaPartition, graph.A_G)
    for fiber1 in range(0,len(graph.M_G.A_G)):
        for fiber2 in range(0,len(graph.M_G.A_G)):
            for vtx in range(0,len(graph.sigmaPartition[fiber1])): # Checks that, for each vtx in fiber1, the number of edges between that vtx and fiber2 matches the number edges between corresponding vertices in M(G).
                if graph.sigmaPartition[fiber1][vtx] == 1:
                    if vtxToFiber[vtx][fiber2] != graph.M_G.A_G[fiber1][fiber2]:
                        return False
                    elif fiberToVtx[fiber2][vtx] != graph.M_G.A_G[fiber2][fiber1]:
                        return False
    return True

def generatePartitions(size, parts, trivial): # generates every partition with a given number of parts of a set of a given size. "trivial" allows trivial partitions. Recursive, not used in the main testing procedure
    if parts == 1:
        return [[size]]
    else:
        partitions = []
        if not trivial:
            for i in range(1, size - parts + 2):
                for subPartition in generatePartitions(size-i,parts-1, False):
                    partitions.append([i,*subPartition])
        else:
            for i in range(0, size+1):
                for subPartition in generatePartitions(size-i,parts-1, True):
                    partitions.append([i,*subPartition])
        return partitions

def randomPartition(size, parts, trivial):
    if trivial:
        partitionNum = np.random.randint(1,combinationsWithReplacement(size+1,parts-1)+1) # chooses a random partition. indexed lexicographically
        partition = []
        numEarlierPartitions = 0
        for i in range(1,parts): # constructs the partitionNum partition in the lexicographic order
            k=0
            numShortPartitions = combinationsWithReplacement(size+1-sum(partition)-k,parts-1-i) # num partitions that start with the constructed partition and k
            while numEarlierPartitions + numShortPartitions < partitionNum: # adds partitions earlier in the lexographic order while incrementing k
                numEarlierPartitions = numEarlierPartitions + numShortPartitions
                k=k+1
                numShortPartitions = combinationsWithReplacement(size + 1 - sum(partition) - k, parts - 1 - i)
            partition.append(k) # add k to the constructed partition
        partition.append(size - sum(partition)) # remainder belongs to the final partition element
        return partition
    else:
        return randomPartition(size-parts,parts, True) + np.ones(parts, dtype=int) # partitions not permitting zeros

def combinationsWithReplacement(n,k): # returns (n+k-1)!/((n-1)!k!)
    result = 1
    if n-1>k:
        for i in range(1, k+1):
            result = float(result*(n-1+i))/i
        return int(result)
    else:
        for i in range(1,n):
            result = float(result*(k+i))/i
        return int(result)

def isIrreducibile(adjMtx):
    vtxList = [0] # Lists vtxs in order encountered by constructing a spanning tree
    lowestConn = np.arange(0,len(adjMtx)) # entry at index i = vtx connected to i that occurs earliest on vtxList
    connections(0,adjMtx,vtxList,lowestConn)
    for i in range(1, len(adjMtx)):
        try:
            if vtxList.index(lowestConn[i]) >= vtxList.index(i): # if there is a non-root vtx (ie not vtx 0) not connected to something lower of vtxList, return False
                return False
        except ValueError: # vtxList.index(i) throws ValueError if there is a vtx i that does not occur on vtxList. ie. there is a vtx not reached by the spanning tree
            return False
    return True

def connections(vtx, adjMtx, vtxList, lowestConn):
    for i in range(0,len(adjMtx[vtx])):
        if adjMtx[vtx][i] > 0: # consider followers of vtx
            try:
                if vtxList.index(lowestConn[vtx])>vtxList.index(i): # if i is on vtxList and appears lower than current lowest, make current lowest i
                    lowestConn[vtx]=i
            except ValueError: # throws if i is not on vtxList
                vtxList.append(i)
                connections(i,adjMtx,vtxList,lowestConn) # iterate on i
                if vtxList.index(lowestConn[vtx])>vtxList.index(lowestConn[i]): # if the lowest connection of i is lower than current lowest of vtx, make current lowest of vtx the lowest connection of i
                    lowestConn[vtx]=lowestConn[i]

def testIrreducibile(adjMtx): # sum powers of adjMtx upto n-1. return whether the result is positive. used for testing isIrreducibile
    sumMtx = np.identity(len(adjMtx), dtype=int)
    A_n = np.identity(len(adjMtx), dtype=int)
    for i in range(1,len(adjMtx)):
        A_n = np.matmul(A_n,adjMtx)
        sumMtx = sumMtx + A_n
    for i in range(0,len(adjMtx)):
        for j in range(0, len(adjMtx)):
            if sumMtx[i][j] == 0:
                return False
    return True

def generateQuotientGraph(graph,congClasses):
    adjMtx = []
    congT = np.transpose(congClasses)
    for part in congClasses:
        vtxRep = 0
        while part[vtxRep] == 0: # finds a vertex representative in the congruence class "part"
            vtxRep += 1
        adjMtx.append(np.matmul(graph.A_G[vtxRep],congT)) # sets the outgoing edges of the corresponding vertex in the quotient according to the edges from vtxRep into each congruence class
    return Graph(adjMtx)


class RightResolver:
    def __init__(self, domainGraph, rangeGraph, vtxMap, edgeMap):
        self.domainGraph = domainGraph
        self.rangeGraph = rangeGraph
        self.vtxMap = vtxMap
        self.edgeMap = edgeMap
        self.stabRelation = None
        self.stabClasses = None
        self.isSync = None
        self.isAsync = None
        self.selfFiberProd = None
        self.isBiResolving = None

    def constructSelfFiberProd(self):
        self.selfFiberProd = FiberProduct(self, self)
        self.selfFiberProd.constructFiberProduct()

    def constructStabRelation(self):
        if self.selfFiberProd is None:
            self.constructSelfFiberProd()
        diagonal = list(np.zeros(len(self.selfFiberProd.productGraph.A_G), dtype=int))
        for i in range(0, len(self.domainGraph.A_G)):
            diagonal[self.selfFiberProd.mtxIndex([i, i])] = 1
        toDiagonal = vtxsWithPathToSet(self.selfFiberProd.productGraph,
                                       diagonal)  # pairs of vertices that can be syncronized
        noneToDiagonal = [(toDiagonal[i] + 1) % 2 for i in
                          range(0, len(toDiagonal))]  # pairs of vertices that cannot be syncronized
        toNoneToDiagonal = vtxsWithPathToSet(self.selfFiberProd.productGraph,
                                             noneToDiagonal)  # pairs of vertices that can be taken to an unsyncroniable pair by lifting a path. ie. pairs that are not stable
        self.stabRelation = [self.selfFiberProd.pairIndex(i) for i in range(0, len(toNoneToDiagonal)) if
                             toNoneToDiagonal[i] == 0]

    def constructStabClasses(self):  # constructs the equivalence classes of the stability congruence "stabRelation"
        if self.stabRelation is None:
            self.constructStabRelation()
        stabClasses = np.zeros((len(self.domainGraph.A_G), len(self.domainGraph.A_G)), dtype=int)
        for pair in self.stabRelation:
            stabClasses[pair[0]][pair[1]] = 1
        stabClasses = np.unique(stabClasses, axis=0)
        self.stabClasses = stabClasses

    def getIsSync(self):
        if self.isSync is None:
            if self.stabClasses is None:
                self.constructStabClasses()
            self.isSync = len(self.vtxMap) == len(
                self.stabClasses)  # a right resolver is syncronizing iff it has as many stability classes as fibers
        return self.isSync

    def getIsAsync(self):  # returns true iff the stability congruence is trivial
        if self.isAsync is None:
            if self.stabClasses is None:
                self.constructStabClasses()
            self.isAsync = len(self.domainGraph.A_G) == len(self.stabClasses)
        return self.isAsync

    def getIsBiResolving(self):
        if self.isBiResolving is None:
            for termH in range(0, len(self.edgeMap)):
                for termG in range(0, len(self.domainGraph.A_G)):
                    if self.vtxMap[termH][termG] == 1:
                        for initH in range(0, len(self.edgeMap[termH])):
                            for edge in range(0, len(self.edgeMap[initH][termH])):
                                sum = 0
                                for i in range(0,
                                               len(self.domainGraph.A_G)):  # sums the number of edges in the preimage of the initH->termH edge "edge" that terminate at termG.
                                    sum += self.edgeMap[initH][termH][edge][i][termG]
                                if sum != 1:  # each edge must have a unique preimage
                                    self.isBiResolving = False
                                    return False
            self.isBiResolving = True
            return True
        return self.isBiResolving

    def display(self):
        for initH in range(0, len(self.rangeGraph.A_G)):
            print("fiber of vtx ", initH, ":")
            print(self.vtxMap[initH])
            print("fibers of out edges of vtx ", initH, ":")
            for termH in range(0, len(self.rangeGraph.A_G)):
                if self.rangeGraph.A_G[initH][termH] > 0:
                    print("to vtx", termH, ":")
                    for edge in range(0, self.rangeGraph.A_G[initH][termH]):
                        print(np.array(self.edgeMap[initH][termH][edge]))

    def equals(self, rightRes):  # returns true iff right resolvers are equal. Sensitve to permutations of parallel edges in the range graph
        if not np.array_equal(self.rangeGraph.A_G, rightRes.rangeGraph.A_G):
            return False
        if not np.array_equal(self.domainGraph.A_G, rightRes.domainGraph.A_G):
            return False
        for fiber in range(0, len(self.vtxMap)):
            for vtx in range(0, len(self.vtxMap[fiber])):
                if self.vtxMap[fiber][vtx] != rightRes.vtxMap[fiber][vtx]:
                    return False
        for initH in range(0, len(self.edgeMap)):
            for termH in range(0, len(self.edgeMap)):
                for edge in range(0, len(self.edgeMap[initH][termH])):
                    if not np.array_equal(self.edgeMap[initH][termH][edge], rightRes.edgeMap[initH][termH][edge]):
                        return False
        return True

def generateRightResolver(G,H): # Generates a random right resolver from G to H
    if G.M_G is None:
        G.constructM_G()
    if H.M_G is None:
        H.constructM_G()
    if not np.array_equal(G.M_G.A_G,H.M_G.A_G):
        print("M(G) is not M(H)")
        return None
    for tryNum in range(0,1000000):
        vtxMap = np.zeros((len(H.A_G),len(G.A_G)), dtype=int)
        for vtx in range(0,len(G.M_G.A_G)):
            if sum(G.sigmaPartition[vtx])<sum(H.sigmaPartition[vtx]):
                print("H fiber bigger than G fiber")
                return None
            fiberSurj = randomSurjection(sum(G.sigmaPartition[vtx]),sum(H.sigmaPartition[vtx])) # a random surjection from the G fiber of vtx to the H fiber of vtx
            i=0
            for iFib in range(0,len(fiberSurj)):
                while H.sigmaPartition[vtx][i] == 0:
                    i = i+1
                j=0
                for jFib in range(0,len(fiberSurj[iFib])):
                    while G.sigmaPartition[vtx][j] == 0:
                        j = j+1
                    vtxMap[i][j] = fiberSurj[iFib][jFib] # map the "jFib"th vertex in the G fiber of "vtx" to the "iFib"th vertex in the H fiber of "vtx" iff "jFib" maps to "iFib" in "fiberSurj"
                    j = j+1
                i = i+1
        edgeMap = [[[] for j in range(0,len(H.A_G))] for k in range(0,len(H.A_G))] # The 2 dimensional array "edgeMap[initH][termH][edge]" is the preimage of the "edge"th edge from "initH" to "termH" in H
        for i in range(0,len(H.A_G)):
            for j in range(0,len(H.A_G)):
                edgeMap[i][j] = [np.zeros((len(G.A_G), len(G.A_G)), dtype=int) for k in range(0, H.A_G[i][j])]

        breakOut = False
        for fiber in range(0,len(vtxMap)):
            intoFiber = np.matmul(G.A_G,np.transpose(vtxMap[fiber]))
            for vtxG in range(0, len(G.A_G)):
                vtxH = np.where(np.transpose(vtxMap)[vtxG] == 1)[0][0] # vtxG maps to vtxH
                if intoFiber[vtxG] != H.A_G[vtxH][fiber]: # the number of edges from "vtxG" into the "fiber" fiber must be the same as the number of edges from "vtxH" to the "fiber" vertex in H
                    breakOut = True
                    break

                edgeMapToFiber = list(np.identity(H.A_G[vtxH][fiber], dtype=int))
                random.shuffle(edgeMapToFiber)

                for edge in range(0, H.A_G[vtxH][fiber]):
                    intoFiberIndx = 0
                    for i in range(0,len(vtxMap[fiber])):
                        if vtxMap[fiber][i] == 1: # for vertices "i" that map into "fiber"
                            for j in range(0,G.A_G[vtxG][i]): # for edges "j" from "vtxG" to "i"
                                if edgeMapToFiber[edge][intoFiberIndx] == 1: # if the "edge"th edge from "vtxH" to "fiber" should map to the "intoFiberIndx"th edge from "vtxG" into the "fiber" fiber
                                    edgeMap[vtxH][fiber][edge][vtxG][i] = 1 # set an edge vtxG->i in G to map to the edge in H. Since the restricted edge map is injective, this value should only ever be 1
                                intoFiberIndx = intoFiberIndx + 1
            if breakOut:
                break
        if not breakOut:
            return RightResolver(G,H,vtxMap, edgeMap)
    print("no RR found")
    print("A_G:")
    print(np.array(G.A_G))
    print("A_H:")
    print(np.array(H.A_G))

def generateRightResToB_G(G): # generates a random right resolver from G to B(G) using the constructed congruence relation
    if G.B_G is None:
        G.constructB_G()

    vtxMap = G.B_GCongClasses

    edgeMap = [[[] for j in range(0, len(G.B_G.A_G))] for k in range(0, len(G.B_G.A_G))]
    for i in range(0, len(G.B_G.A_G)):
        for j in range(0, len(G.B_G.A_G)):
            edgeMap[i][j] = [np.zeros((len(G.A_G), len(G.A_G)), dtype=int) for k in range(0, G.B_G.A_G[i][j])]

    for fiber in range(0, len(vtxMap)):
        intoFiber = np.matmul(G.A_G, np.transpose(vtxMap[fiber]))
        for vtxG in range(0, len(G.A_G)):
            vtxH = np.where(np.transpose(vtxMap)[vtxG] == 1)[0][0]
            if intoFiber[vtxG] != G.B_G.A_G[vtxH][fiber]: # never satisfied
                print("no right res for vtx map by cong")
                print(np.array(G.A_G))
                print(np.array(G.B_G.A_G))
                print(np.array(vtxMap))
            edgeMapToFiber = list(np.identity(G.B_G.A_G[vtxH][fiber], dtype=int))
            random.shuffle(edgeMapToFiber)
            for edge in range(0, G.B_G.A_G[vtxH][fiber]):
                intoFiberIndx = 0
                for i in range(0, len(vtxMap[fiber])):
                    if vtxMap[fiber][i] == 1:
                        for j in range(0, G.A_G[vtxG][i]):
                            if edgeMapToFiber[edge][intoFiberIndx] == 1:
                                edgeMap[vtxH][fiber][edge][vtxG][i] = 1
                            intoFiberIndx = intoFiberIndx + 1
    return RightResolver(G, G.B_G, vtxMap, edgeMap)

def randomSurjection(domainSize, rangeSize): # random surjection of domainSize elements onto rangeSize elements in the form of a rangeSize by domainSize binary array
    partition = randomPartition(domainSize,rangeSize, False)
    surjectionMtx = np.zeros((rangeSize,domainSize), dtype=int)
    k=0
    for i in range(0,len(partition)):
        for j in range(0,partition[i]):
            surjectionMtx[i][k] = 1
            k = k+1
    surjT = list(np.transpose(surjectionMtx))
    random.shuffle(surjT)
    return np.transpose(surjT)

def isRightResolver(rightRes): # checks if a right resolver is a homomorphism and bijective on outgoing edges (for testing)
    if not isHomomorphism(rightRes):
        return False

    for initH in range(0, len(rightRes.edgeMap)):
        for initG in range(0, len(rightRes.domainGraph.A_G)):
            if rightRes.vtxMap[initH][initG] == 1:
                for termH in range(0, len(rightRes.edgeMap[initH])):
                    for edge in range(0, len(rightRes.edgeMap[initH][termH])):
                        if sum(rightRes.edgeMap[initH][termH][edge][initG]) != 1:
                            print("edge", edge, "from", initH, "to", termH, "has", sum(rightRes.edgeMap[initH][termH][edge][initG]), "preimages in E_", initG, "(G)")
                            return False
    return True

def isHomomorphism(homorph): # checks whether a right resolver is a homomorphism (for testing)
    vtxMapT = np.transpose(homorph.vtxMap)
    for vtx in range(0,len(vtxMapT)):
        if sum(vtxMapT[vtx]) != 1:
            print(vtx, "in G has", sum(vtxMapT[vtx]), "images in H" )
            return False

    sumMtx = np.zeros((len(homorph.domainGraph.A_G), len(homorph.domainGraph.A_G)), dtype=int)
    for initH in range(0,len(homorph.edgeMap)):
        for termH in range(0,len(homorph.edgeMap[initH])):
            for edge in range(0,len(homorph.edgeMap[initH][termH])):
                sumMtx = sumMtx + np.array(homorph.edgeMap[initH][termH][edge], dtype=int)
                for initG in range(0,len(homorph.edgeMap[initH][termH][edge])):
                    for termG in range(0,len(homorph.edgeMap[initH][termH][edge][initG])):
                        if homorph.edgeMap[initH][termH][edge][initG][termG] > 0:
                            if not(homorph.vtxMap[initH][initG] == 1 and homorph.vtxMap[termH][termG] == 1):
                                print("a", initG, "->", termG, "edge maps to a ", initH, "->", termH, "edge, but (", initG, termG,") does not map to (", initH, termH,")")
                                return False
    if not np.array_equal(homorph.domainGraph.A_G, sumMtx):
        print("some edge does not have a unique image")
        return False
    return True

def vtxsWithPathToSet(graph, vtxSet): # returns the set of vtxs with a path to vtxSet as a binary vector
    withPathToSet = vtxSet.copy()
    for i in range(0,len(vtxSet)):
        if vtxSet[i] == 1:
            withPathToVtx(i,graph.A_G,withPathToSet)
    return withPathToSet

def withPathToVtx(vtx, A_G, withPathToSet): # recursive. Adds vertices with outgoing edges incedent to vtx to the vector withPathToSet
    for i in range(len(A_G)):
        if A_G[i][vtx] > 0:
            if withPathToSet[i] == 0:
                withPathToSet[i] = 1
                withPathToVtx(i, A_G, withPathToSet)

def vtxsWithPathFromSet(graph, vtxSet):
    withPathFromSet = vtxSet.copy()
    for i in range(0,len(vtxSet)):
        if vtxSet[i] == 1:
            withPathFromVtx(i,graph.A_G,withPathFromSet)
    return withPathFromSet

def withPathFromVtx(vtx, A_G, withPathFromSet):
    for i in range(len(A_G)):
        if A_G[vtx][i] > 0:
            if withPathFromSet[i] == 0:
                withPathFromSet[i] = 1
                withPathFromVtx(i, A_G, withPathFromSet)


class FiberProduct:
    def __init__(self,rightRes1,rightRes2):
        self.rightRes1 = rightRes1
        self.rightRes2 = rightRes2
        self.productGraph = None
        self.projMap1 = None
        self.projMap2 = None

    def pairIndex(self, mtxIndex):
        return [int(mtxIndex/len(self.rightRes2.domainGraph.A_G)),mtxIndex % len(self.rightRes2.domainGraph.A_G)]
    def mtxIndex(self, pairIndex):
        return pairIndex[0]*len(self.rightRes2.domainGraph.A_G)+pairIndex[1]

    def constructFiberProduct(self):
        if not np.array_equal(self.rightRes1.rangeGraph.A_G, self.rightRes2.rangeGraph.A_G):
            print("range adj matrices not equal") # matrices will be literally equal when taking selfFiberProd for a single right resolver
            return
        AG_1 = self.rightRes1.domainGraph.A_G
        AG_2 = self.rightRes2.domainGraph.A_G
        A_H = self.rightRes1.rangeGraph.A_G
        adjMtx = np.zeros((len(AG_1)*len(AG_2),len(AG_1)*len(AG_2)), dtype=int)
        for initH in range(0,len(A_H)): # populates the adjacency matrix of the product graph with an edge (initG_1,initG_2)->(termG_1,termG_2) for each pair of edges initG_1->termG_1, initG_1->termG_2 with the same image under the respective edge maps.
            for termH in range(0,len(A_H)):
                for edge in range(0,A_H[initH][termH]):
                    for initG_1 in range(0,len(AG_1)):
                        for termG_1 in range(0,len(AG_1)):
                            for i in range(0,self.rightRes1.edgeMap[initH][termH][edge][initG_1][termG_1]):
                                for initG_2 in range(0,len(AG_2)):
                                    for termG_2 in range(0,len(AG_2)):
                                        for j in range(0,self.rightRes2.edgeMap[initH][termH][edge][initG_2][termG_2]):
                                            adjMtx[self.mtxIndex([initG_1,initG_2])][self.mtxIndex([termG_1,termG_2])] += 1
        self.productGraph = Graph(adjMtx)

def indexOfGraph(graph): # returns a unique index for any adjacency matrix based on a lexicographic ordering
    graphNum = 0
    edgeSum = 0
    for i in range(0, len(graph.A_G)):
        for j in range(0, len(graph.A_G)):
            if i+j>0:
                for k in range(1, graph.A_G[i][j]+1):
                    graphNum += combinationsWithReplacement(edgeSum+k+1, i*len(graph.A_G)+j-1)
            edgeSum += graph.A_G[i][j]
    return [len(graph.A_G), edgeSum, graphNum]

def probRRIsSync(G,H,trials, maxTries = 1000): # estimates probability that a right resolver from G to H synchronizing using a sum of geometric random variables
    sample = np.zeros(trials, dtype=int)
    for n in range(0,trials):
        i = 1
        while not generateRightResolver(G,H).getIsSync() and i<maxTries:
            i+=1
        if i == maxTries:
            #print("no sync found")
            return -1
        sample[n] = i
    return float(trials/sum(sample))

def probRRToB_GIsSync(G,trials, byQuotient, maxTries = 1000): # probability that a right resolver from G to B(G) is synchronizing. Uses the constructed bunchy congruence for the vertex map if "byQuotient" is true
    if G.B_G is None:
        G.constructB_G()
    if byQuotient:
        sample = np.zeros(trials, dtype=int)
        for n in range(0, trials):
            i = 1
            while not generateRightResToB_G(G).getIsSync() and i < maxTries:
                i += 1
            if i == maxTries:
                # print("no sync found")
                return -1
            sample[n] = i
        return float(trials / sum(sample))
    else:
        return probRRIsSync(G, G.B_G, trials, maxTries)
def testGraphsByM_G(M_G,n,numGraphs,trials,byQuotient = False, maxTries = 1000):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'

    try:
        file = open(fileName, 'rb')
        hists = pickle.load(file)
        file.close()
    except OSError:
        hists = []

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists):
        hists.append([M_G])
    if len(hists[M_GIndex]) <= n:
        hists[M_GIndex].extend([[]]*(n - len(hists[M_GIndex]) + 1))

    hist = copy.deepcopy(hists[M_GIndex][n])

    graphNum = len(hist)
    hist.extend([[]]*numGraphs)
    while graphNum < len(hist):
        print(graphNum)
        G = generateRandomGraph(M_G, n)
        G.constructB_G()
        prob = probRRToB_GIsSync(G, trials, byQuotient, maxTries)
        i = graphNum
        while i > 1 and hist[i-1][1] < prob:
            hist[i] = hist[i-1]
            i += -1
        hist[i] = [G,prob]
        graphNum += 1

    hists[M_GIndex][n] = hist

    file = open(fileName, 'wb')
    pickle.dump(hists, file)
    file.close()

def printProbHists(M_G, n, numBins,byQuotient):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'
    file = open(fileName, 'rb')
    hists = pickle.load(file)
    file.close()

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists) or len(hists[M_GIndex])<=n or len(hists[M_GIndex][n]) == 0:
        print("no hist for M(G) and n")
    else:
        hist = np.zeros(numBins, dtype= int)
        print("Graphs with prob != 1:")
        for pair in hists[M_GIndex][n]:
            if pair[1] < 1:
                print(np.array(pair[0].A_G))
                print("prob:", pair[1])
                print()
            index = int(pair[1]*numBins)
            if index == numBins:
                index += -1
            hist[index] += 1
        print("Hist:")
        print(hist)

def generateBOverOScatterCSV(M_G,n,byQuotient = False):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'
    file = open(fileName, 'rb')
    hists = pickle.load(file)
    file.close()

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists) or len(hists[M_GIndex])<=n or len(hists[M_GIndex][n]) == 0:
        print("no hist for M(G) and n")
    else:
        if byQuotient:
            writeFileName = "qBvOScatter_"
        else:
            writeFileName = "BvOScatter_"
        index = indexOfGraph(M_G)
        f = open(writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv", 'w', newline='')
        writer = csv.writer(f, delimiter = ',')

        rows = []
        for pair in hists[M_GIndex][n]:
            if pair[0].O_B_G is None:
                pair[0].constructO_B_G()
            rows.append([float(len(pair[0].B_G.A_G)/len(pair[0].O_B_G.A_G)), pair[1]])

        writer.writerows(rows)
        f.close()
        print("scatter written to " + writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv")

def generateBOverGAvgProbScatterCSV(M_G, n, byQuotient = False):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'
    file = open(fileName, 'rb')
    hists = pickle.load(file)
    file.close()

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists) or len(hists[M_GIndex])<=n or len(hists[M_GIndex][n]) == 0:
        print("no hist for M(G) and n")
    else:
        if byQuotient:
            writeFileName = "qBvOAvgProbScatter_"
        else:
            writeFileName = "BvOAvgProbScatter_"
        index = indexOfGraph(M_G)
        f = open(writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv", 'w', newline='')
        writer = csv.writer(f, delimiter = ',')

        rows = []
        for pair in hists[M_GIndex][n]:
            if pair[0].O_B_G is None:
                pair[0].constructO_B_G()
            ratioIndex = 0
            while ratioIndex < len(rows) and rows[ratioIndex][0][1]*len(pair[0].B_G.A_G) != rows[ratioIndex][0][0]*len(pair[0].O_B_G.A_G):
                ratioIndex += 1
            if ratioIndex == len(rows):
                rows.append([[len(pair[0].B_G.A_G), len(pair[0].O_B_G.A_G)], 0, 0])
                while 0 < ratioIndex and float(rows[ratioIndex-1][0][0]/rows[ratioIndex-1][0][1]) > float(len(pair[0].B_G.A_G)/len(pair[0].O_B_G.A_G)):
                    rows[ratioIndex] = rows[ratioIndex-1]
                    ratioIndex += -1
                rows[ratioIndex] = [[len(pair[0].B_G.A_G), len(pair[0].O_B_G.A_G)], 0, 0]
            rows[ratioIndex][1] += pair[1]
            rows[ratioIndex][2] += 1

        writer.writerows([[float(row[0][0]/row[0][1]), float(row[1]/row[2]), row[2]] for row in rows])
        f.close()
        print("hist written to " + writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv")

def generateBOverOMultiHistCSV(M_G,n, numBins, byQuotient = False):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'
    file = open(fileName, 'rb')
    hists = pickle.load(file)
    file.close()

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists) or len(hists[M_GIndex])<=n or len(hists[M_GIndex][n]) == 0:
        print("no hist for M(G) and n")
    else:
        if byQuotient:
            writeFileName = "qBvOMultiHist_"
        else:
            writeFileName = "BvOMultiHist_"
        index = indexOfGraph(M_G)
        f = open(writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv", 'w', newline='')
        writer = csv.writer(f, delimiter = ',')

        col = [["-", *[float(i/numBins) for i in range(0,numBins)]]]
        for pair in hists[M_GIndex][n]:
            if pair[0].O_B_G is None:
                pair[0].constructO_B_G()
            probIndex = int(pair[1]*numBins)+1
            if probIndex == numBins+1:
                probIndex += -1
            ratioIndex = 1
            while ratioIndex < len(col) and col[ratioIndex][0][1]*len(pair[0].B_G.A_G) != col[ratioIndex][0][0]*len(pair[0].O_B_G.A_G):
                ratioIndex += 1
            if ratioIndex == len(col):
                col.append([[len(pair[0].B_G.A_G), len(pair[0].O_B_G.A_G)], *list(np.zeros(numBins, dtype= int))])
                while 1 < ratioIndex and float(col[ratioIndex-1][0][0]/col[ratioIndex-1][0][1]) > float(len(pair[0].B_G.A_G)/len(pair[0].O_B_G.A_G)):
                    col[ratioIndex] = col[ratioIndex-1]
                    ratioIndex += -1
                col[ratioIndex] = [[len(pair[0].B_G.A_G), len(pair[0].O_B_G.A_G)], *list(np.zeros(numBins, dtype= int))]
            col[ratioIndex][probIndex] += 1
        col = [col[0], *[[float(cl[0][0]/cl[0][1]), *cl[1:]] for cl in col[1:]]]
        writer.writerows(list(np.transpose(col)))
        f.close()
        print("hist written to " + writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv")

def generateProbHistCSV(M_G,n,numBins,byQuotient = False):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'
    file = open(fileName, 'rb')
    hists = pickle.load(file)
    file.close()

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists) or len(hists[M_GIndex])<=n or len(hists[M_GIndex][n]) == 0:
        print("no hist for M(G) and n")
    else:
        if byQuotient:
            writeFileName = "qHist_"
        else:
            writeFileName = "hist_"
        index = indexOfGraph(M_G)
        f = open(writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + "_" + str(numBins) + ".csv", 'w', newline='')
        writer = csv.writer(f, delimiter = ',')

        rows = [[float(i/numBins),0] for i in range(0, numBins)]
        for pair in hists[M_GIndex][n]:
            bin = int(pair[1]*numBins)
            if bin == numBins:
                bin += -1
            rows[bin][1] += 1

        writer.writerows(rows)
        f.close()
        print("histogram written to " + writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + "_" + str(numBins) + ".csv")

def generateB_GHistCSV(M_G,n,byQuotient = False):
    if byQuotient:
        fileName = 'qHistGraphs.pkl'
    else:
        fileName = 'histGraphs.pkl'
    file = open(fileName, 'rb')
    hists = pickle.load(file)
    file.close()

    M_GIndex = 0
    while M_GIndex < len(hists) and not np.array_equal(hists[M_GIndex][0].A_G, M_G.A_G):
        M_GIndex += 1
    if M_GIndex == len(hists) or len(hists[M_GIndex])<=n or len(hists[M_GIndex][n]) == 0:
        print("no hist for M(G) and n")
    else:
        if byQuotient:
            writeFileName = "qB_GHist_"
        else:
            writeFileName = "B_GHist_"
        index = indexOfGraph(M_G)
        f = open(writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv", 'w', newline='')
        writer = csv.writer(f, delimiter = ',')

        rows = [[i,0] for i in range(len(M_G.A_G), n+1)]
        for pair in hists[M_GIndex][n]:
            rows[len(pair[0].B_G.A_G)-len(M_G.A_G)][1] += 1

        writer.writerows(rows)
        f.close()
        print("histogram written to " + writeFileName + str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + "_" + str(n) + ".csv")

