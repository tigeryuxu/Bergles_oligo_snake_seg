
import struct
import numpy as np
import os
import math
import scipy as sp
import scipy.spatial
import scipy.signal
import matplotlib.pyplot as plt

class vertex:
    def __init__(self, x, y, z, e_out, e_in):
        self.p = np.array([x, y, z])
        self.Eout = e_out
        self.Ein = e_in

class edge:
    def __init__(self, v0, v1, p):
        self.v = (v0, v1)
        self.p = p

class linesegment:
    def __init__(self, p0, p1):
        self.p = (p0, p1)

    #return a set of points sampling the line segment at the specified spacing
    def pointcloud(self, spacing):
        v = self.p[1] - self.p[0]
        l = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        n = int(np.ceil(l / spacing))
        if n == 0:
            return [self.p[0], self.p[1]]
        
        pc = []
        for i in range(n+1):
            pc.append(self.p[0] + v * (i/n))
        return pc

class NWT:
    def __init__(self, filename):        
        [_, fext] = os.path.splitext(filename)                          #get the file extension so that we know the file type
        if fext == ".nwt":                                              #if the file extension is NWT
            self.load_nwt(filename)                                     #load a NWT file
        elif fext == ".obj":                                            #if the file extension is OBJ
            self.load_obj(filename)                                     #load an OBJ file
        else:                                                           #otherwise raise an exception
            raise ValueError("file type is unsupported as a network")

    def load_nwt(self, filename):        
        fid = open(filename, "rb")                                      #open a binary file for reading
        self.header = fid.read(14).decode("utf-8")                      #load the header
        self.desc = fid.read(58).decode("utf-8")                        #load the description
        nv = struct.unpack("I", fid.read(4))[0]                         #load the number of vertices and edges
        ne = struct.unpack("I", fid.read(4))[0]

        self.v = []                                                     #create an empty list to store the vertices        
        for _ in range(nv):                                             #iterate across all vertices
            p = np.fromfile(fid, np.float32, 3)                         #read the vertex position
            E = np.fromfile(fid, np.uint32, 2)                          #read the number of edges
            e_out = np.fromfile(fid, np.uint32, E[0])                   #read the indices of the outgoing edges
            e_in = np.fromfile(fid, np.uint32, E[1])                    #read the indices of the incoming edges
            v = vertex(p[0], p[1], p[2], e_out, e_in)                   #create a vertex
            self.v.append(v)

        self.e = []                                                     #create an empty array to store the edges        
        for _ in range(ne):                                             #iterate over all edges            
            v = np.fromfile(fid, np.uint32, 2)                          #load the vertex indices that this edge connects            
            npts = struct.unpack("I", fid.read(4))[0]                   #read the number of points defining this edge            
            pv = np.fromfile(fid, np.float32, 4*npts)                   #read the array of points            
            p = [(pv[i],pv[i+1]) for i in range(0,npts,2)]              #conver the point values to an array of 4-element tuples            
            self.e.append(edge(v[0], v[1], p))                          #create and append the edge to the edge list

    def load_obj(self, filename):        
        fid = open(filename, "r")                                       #open the file for reading        
        vertices = []                                                   #create an array of vertices
        lines = []                                                      #create an array of lines
        for line in fid:                                                #for each line in the file
            elements = line.split(" ")                                  #split it into token elements            
            if elements[0] == "v":                                      #if the element is a vertex               
                c = [float(i) for i in elements[1:]]                    #get the point coordinates                
                vertices.append(c)                                      #add the coordinates to the vertex list            
            if elements[0] == "l":                                      #if the element is a line                
                idx = [int(i) for i in elements[1:]]                    #get the indices for the points that make up the line                
                lines.append(idx)                                       #add this line to the line list

        self.header = "nwtfileformat "                                  #assign a header and description
        self.desc = "File generated from OBJ"
                                                                        #insert the first and last vertex ID for each line into a set
        vertex_set = set()                                              #create an empty set
        for line in lines:                                              #for each line in the list of lines
            vertex_set.add(line[0])                                     #add the first and last vertex to the vertex set (this will remove redundancies)
            vertex_set.add(line[-1])
        
        obj2nwt = dict()                                                #create a new dictionary - will be used to map vertex IDs in the OBJ to IDs in the NWT object

        #create a mapping between OBJ vertex indices and NWT vertex indices
        vi = 0                                                          #initialize a vertex counter to zero
        for si in vertex_set:                                           #for each vertex in the set of vertices
            obj2nwt[si] = vi                                            #assign the mapping
            vi = vi + 1                                                 #increment the vertex counter

        #iterate through each line (edge), assigning them to their starting and ending vertices
        v_out = [list() for _ in range(len(vertex_set))]                #create an array of empty lists storing the inlet and outlet edges for each vertex
        v_in = [list() for _ in range(len(vertex_set))]

        self.e = []                                                     #create an empty list storing the NWT vertex IDs for each edge (inlet and outlet)
        for li in range(len(lines)):                                    #for each line
            v0 = obj2nwt[lines[li][0]]                                  #get the NWT index for the starting and ending points (vertices)
            v1 = obj2nwt[lines[li][-1]]

            v_out[v0].append(li)                                        #add the line index to a list of inlet edges
            v_in[v1].append(li)                                         #add the line index to a list of outlet edges

            p = []                                                      #create an emptu array of points used to store point positions in the NWT graph
            for pi in range(1, len(lines[li]) - 1):                     #for each point in the line that is not an end point (vertex)
                p.append(np.array(vertices[lines[li][pi]-1]))              #add the coordinates for that point as a tuple into the point list
            self.e.append(edge(v0, v1, p))                              #create an edge, specifying the inlet and outlet vertices and all defining points

        #for each vertex in the set, create a NWT vertex containing all of the necessary edge information
        self.v = []                                                     #create an empty list to store the vertices
        for s in vertex_set:                                            #for each OBJ vertex in the vertex set
            vi = obj2nwt[s]                                             #calculate the corresponding NWT index
            self.v.append(vertex(vertices[s-1][0], vertices[s-1][1], vertices[s-1][2], v_out[vi], v_in[vi]))    #create a vertex object, consisting of a position and attached edges

    #return a set of line segments connecting all points in the network
    def linesegments(self):

        s = []                                                          #create an empty list of line segments
        for e in self.e:                                                #for each edge in the graph
            p0 = self.v[e.v[0]].p                                       #load the first point (from the starting vertex)

            for p in e.p:                                               #for each point in the edge
                p1 = np.array([p[0], p[1], p[2]])                       #get the second point for the line segment
                s.append(linesegment(p0, p1))                           #append the line segment to the list of line segments
                p0 = p1                                                 #update the start point for the next segment to the end point of this one
            
            p1 = self.v[e.v[1]].p                                       #load the last point (from the ending vertex)
            s.append(linesegment(p0, p1))                               #append the last line segment for this edge to the list
        return s

    #return a point cloud sampling the centerline of the network at the given spacing
    def pointcloud(self, spacing):
        ls = self.linesegments()
        pc = []
        for l in ls:
            pc = pc + l.pointcloud(spacing)
        return pc

def gaussian(X, sigma):
    return np.exp(-0.5 * (X ** 2 / sigma ** 2))

#set the input parameters
sigma = 10
GTfile = "04_GT.obj"
Tfile = "04_Tb.obj"

GTfile = "03_GT.obj"
Tfile = "03_T.obj"


GTfile = "00_GT.obj"
Tfile = "00_T.obj"


GTfile = "sample.obj"
Tfile = "sample_alt.obj"


GTfile = "sample_FULL.obj"
Tfile = "sample_FULL_alt.obj"



GTfile = "1to1pair_b_series_t1_eLOSTpairbbslnannot-005_object.obj"
Tfile = "1to1pair_b_series_t1_seed_end_1_input_0_output.obj"


#tunable constants
subdiv = 4                                                            #fraction of sigma used to sample each network
shadow = 10                                                             #thickness of the shadow network when displaying geometric results

#load the ground truth and test case networks
GT = NWT(GTfile)
T = NWT(Tfile)

#generate point clouds representing both networks
P_T = np.array(T.pointcloud(sigma/subdiv))
P_GT = np.array(GT.pointcloud(sigma/subdiv))

#generate KD trees for each network
GT_tree = sp.spatial.cKDTree(P_GT)
T_tree = sp.spatial.cKDTree(P_T)

#query each KD tree to get the corresponding geometric distances
[T_dist, _] = GT_tree.query(P_T)
[GT_dist, _] = T_tree.query(P_GT)

#convert distances to Gaussian metrics
T_metric = gaussian(T_dist, sigma)
GT_metric = gaussian(GT_dist, sigma)

#calculate the TPR and FPR
print("FNR = " + str(1 - np.mean(GT_metric)))
print("FPR = " + str(1 - np.mean(T_metric)))

plt.subplot(1, 2, 1)
plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma*shadow, c="grey")
plt.scatter(P_T[:, 0], P_T[:, 1], s=sigma, c=T_metric, cmap = "plasma")

plt.subplot(1, 2, 2)
plt.scatter(P_T[:, 0], P_T[:, 1], s=sigma*shadow, c="grey")
plt.scatter(P_GT[:, 0], P_GT[:, 1], s=sigma, c=GT_metric, cmap = "plasma")

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(P_GT[:, 0], P_GT[:, 1], P_GT[:, 2], s=sigma*shadow, c="grey")
ax.scatter(P_T[:, 0], P_T[:, 1], P_T[:, 2], s=sigma * shadow * 10, c=T_metric, cmap = "plasma")


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(P_T[:, 0], P_T[:, 1], P_T[:, 2], s=sigma*shadow, c="grey")
ax.scatter(P_GT[:, 0], P_GT[:, 1], P_GT[:, 2], s=sigma, c=GT_metric, cmap = "plasma")










