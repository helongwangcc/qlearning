from functools import partial
import networkx as nx
from shapely.geometry import Point, MultiPoint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from fieldosophy.GRF import FEM
from fieldosophy.GRF import GRF
from fieldosophy import mesh as mesher
from scipy import stats



class RectangleGraph:

    def __init__(self, numOfRow: int, numOfColumn: int) -> None:
        self.__graphDimension = (numOfRow, numOfColumn)
        self.__vertices = self.__createVertices(numOfRow, numOfColumn)
        self.__edges = self.__createEdges()
        self.__graph = self.__createGraph()
        self.SPDMatrix,self.matrixShape = self.weightSPD(3000)

    def addObstacle(self, graph, lowerLeft, upperRight):
        assert lowerLeft[0] < upperRight[0], "Lower left row x > upper right row x."
        assert lowerLeft[1] < upperRight[1], "Lower left col y > upper right col y."
        assert (lowerLeft in self.__vertices) & (upperRight in self.__vertices)
        insideVertices = [
            vertex for vertex in self.__vertices 
            if (lowerLeft[0] <= vertex[0] <= upperRight[0]) & (lowerLeft[1] <= vertex[1] <= upperRight[1])
        ]
        numOfVertex = np.random.randint(3, 6)
        index = np.random.choice(range(len(insideVertices) - 1), numOfVertex, replace=False)     
        polygon = MultiPoint([insideVertices[i] for i in index] + [insideVertices[index[0]]]).convex_hull
        obstacle = [
            vertex for vertex in insideVertices if polygon.contains(Point(vertex))
        ] + [
            insideVertices[i] for i in index
        ]
        for vertex in obstacle:
            for u in graph.neighbors(vertex):
                graph[vertex][u]['weight'] = np.inf
        newPoly = MultiPoint(obstacle).convex_hull
        return newPoly.exterior.coords.xy
    
    def weightSPD(self, numOfSample):
        numOfRow, numOfColumn = self.__graphDimension
        limitOfCoordinate = np.asarray(
            [
                [0, numOfRow],
                [0, numOfColumn]
            ]
        )
        minCorelation = (numOfColumn + numOfRow) * 0.2
        extension = minCorelation * 1.5 
        meshPlane = mesher.regularMesh.meshInPlaneRegular( 
            limitOfCoordinate + \
            extension * np.array([-1,1]).reshape((1,2)), 
            minCorelation /5/np.sqrt(2) 
        )
        r = minCorelation * 1.2
        nu = 1.3
        sigma = 1
        sigmaEps = 2e-2
        BCRobin = np.ones( (meshPlane.getBoundary()["edges"].shape[0], 2) )
        BCRobin[:, 0] = 0  # Association with constant
        BCRobin[:, 1] = - 1 # Association with function
        fem = FEM.MaternFEM(
            mesh = meshPlane,
            childParams = {'r':r},
            nu = nu,
            sigma = sigma, 
            BCDirichlet = None, 
            BCRobin = BCRobin 
        )
        Z = fem.generateRandom(numOfSample)
        graphRow = np.arange(numOfRow)
        graphCol = np.arange(numOfColumn)
        obsPoints = np.meshgrid(graphCol, graphRow)
        obsMat = fem.mesh.getObsMat(
            np.hstack( 
                (obsPoints[0].reshape(-1,1),
                 obsPoints[1].reshape(-1,1)) 
                )
        )
        ZObs = obsMat.tocsr() * Z + \
            stats.norm.rvs( loc = 0, scale = sigmaEps, size = numOfSample * obsMat.shape[0] ).reshape((obsMat.shape[0], numOfSample))
        return ZObs, obsPoints[0].shape
    
    def createWeightFunction(self, map, u, v, e):
        # this is used for creating weight function for Dijkstra's algorithm
        if e['weight'] == np.inf :
            return np.inf
        else:
            return map[u] ** 2 + map[v] ** 2

    
    def findShortestRoute(self, graph, source, target, map):
        # weight is a function of node, node and edge dict
        weight = partial(self.createWeightFunction, map)
        return nx.shortest_path(graph, source, target, weight)       

    
    def drawGraph(self, graph, obstacles = None):
        ax = plt.gca()
        nodelist = list(graph)
        xy = np.asarray(nodelist)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=5,
            c='k',
            marker='o',
        )
        edgelist = list(graph.edges())
        for u, v in edgelist:
            ax.plot(
                [u[0], v[0]], 
                [u[1], v[1]], 
                color = 'k', lw = 1)
        for obstacle in obstacles:
            ax.plot(
                    obstacle[0], 
                    obstacle[1], 
                    color = 'r', lw = 2)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.show()


    def isWithinBoundary(self, x, y):
        return (x, y) in self.__vertices
        
    def __createGraph(self):
        g = nx.Graph()
        for edge in self.__edges:
            g.add_edge(edge[0], edge[1], weight=1.)
        return g

    def __createVertices(self, numOfRow, numOfColumn):
        return [(i, j) for i in range(numOfRow) for j in range(numOfColumn)]
    
    def __createEdges(self):
        iterVertice = deepcopy(self.__vertices)
        direction = [(1,0),(-1,0),(0,1),(0,-1)]
        edge = []
        while iterVertice:
            v = iterVertice.pop()
            for dir in direction:
                if self.isWithinBoundary(v[0] + dir[0], v[1] + dir[1]):
                    edge.append([v, (v[0] + dir[0], v[1] + dir[1])])   
        return edge

    @property
    def vertices(self):
        return self.__vertices
    
    @property
    def edges(self):
        self.__edges

    @property
    def graph(self):
        return self.__graph
    
    @property
    def graphDimension(self):
        return self.__graphDimension
    

if __name__ == "__main__":

    from copy import deepcopy

    rgInstance = RectangleGraph(40, 40)
    
    # # copy graph in the instance
    # graph = deepcopy(rgInstance.graph)

    # # add obstacles
    # obstacles = []
    # obstacles.append(rgInstance.addObstacle(graph, (5,5), (10, 10)))    
    # obstacles.append(rgInstance.addObstacle(graph, (10,10), (15, 15)))
    # obstacles.append(rgInstance.addObstacle(graph, (15,15), (20, 20)))
    # obstacles.append(rgInstance.addObstacle(graph, (20,20), (35, 35)))
    
    # # draw graph and obstacle
    # rgInstance.drawGraph(graph, obstacles)
    
    # # print shortest route
    # print (rgInstance.findShortestRoute(graph, (1,0), (39,39)))