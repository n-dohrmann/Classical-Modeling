#! /Users/Noah/opt/anaconda3/envs/moleng37/bin/python3
"""
functions is the main module for the Molecular Dynamics simulation Code
"""

import numpy as np
import numba
from numba import jit
import sys
import os
from numpy import random
from time import time
import datetime as dt
# import HOOMD-Blue

# initialize and record seed selection
date = dt.datetime.now().strftime("%a %d %b %Y %X")
seed = int(time())
np.random.seed(seed)

### GLOBAL VARIABLES
rc = 2.5 # cut off radius
rho = 0.9 # density of particles
# dt = 1e-2 # time step (1e-3 for NVE)

# writes the runtime date and seed
with open("../Output/seed_log.txt", "a") as sl:
    sl.write(f"{date}, {seed}\n")

# might rewrite this differently later
@jit(nopython=True)
def ljts_potential(r: np.float64) -> np.float64:
    """
    Takes a scalar distance r and returns the truncated and shifted
    Lennard-Jones potential
    """
    # MANUAL ESCAPE TO PREVENT ENERGY DIVERGING
    # if (r < 0.4):
    #     r = 0.4

    # the potential
    lj: np.float64
    # the shift to make the potential continous with zero at rc
    shift =  0.0163

    lj = 4 * ((1/r)**12 - (1/r)**6) + shift 
    # print(lj)
    return lj

@jit(nopython=True)
def get_dist(A: np.array, B: np.array) -> np.float64:
    """
    get the distance between two points in a 6 vector geometry
    """
    dist = np.float64(0.0)
    for i in range(3):
        dist += (A[i] - B[i])**2
    dist = np.sqrt(dist)
    # if (dist < 1e-20):
    #     # print("too close")
    #     dist += 1e-10
    return dist


@jit(nopython=True)
def mag_of_vec(A: np.array) -> np.float64:
    """
    Takes a 3-vector and returns the magnitude of the vector
    """
    mag = np.float64(0.0)
    i = 0
    while i < 3:
        mag += (A[i])**2
        i += 1
    
    return np.sqrt(mag)

@jit(nopython=True)
def nearest_neighbor_vec(A: np.array, B:np.array, V:np.float64) -> np.float64:
    """
    For two cells that are neighbors in teh cell list, find the 
    nearest neighbor image of the two (this should help in preventing
    the total energy from diverging)

    Takes in the coordinates of two particles, returns a length 
    to the nearest neighbor image using the formula given in the problem
    statement. Also takes the volume V so that the length of the box can
    be found
    """
    L = V**(1/3)
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    dz = B[2] - A[2]


    dx = dx - L * round(dx/L)
    dy = dy - L * round(dy/L)
    dz = dz - L * round(dz/L)


    return np.array([dx,dy,dz])


# @jit(nopython=True)
# def nint(x: np.float64) -> int:
#     """
#     Takes in a real number and returns the nearest int
#     """


@jit(nopython=True)
def get_kinetic(C: np.array) -> np.float64:
    """
    Takes in a p and q geometry array and returns the corresponding
    total kinetic energy of the system (K) and the kinetic energy
    of a single particle K0 for debugging purposes
    """
    K =  np.float64(0.0)
    # K0 = np.float64(0.0)

    for i in range(len(C)):
        kx = C[i][3] ** 2 
        ky = C[i][4] ** 2
        kz = C[i][5] ** 2

        K += 0.5 * (kx + ky + kz)
        # if (i == 0):
        #     K0 = K

    return K#, K0


@jit(nopython=True)
def get_T_components(C: np.array) -> np.float64:
    """
    a manual check on the components of T 
    """
    N = len(C)

    tx = np.float64(0.0)
    ty = np.float64(0.0)
    tz = np.float64(0.0)

    for i in range(N):
        tx += C[i][3] ** 2
        ty += C[i][4] ** 2 
        tz += C[i][5] ** 2

    return np.array([tx, ty, tz]) / (3*N -3)


@jit(nopython=True)
def get_dvec(A: np.array,B:np.array) -> np.float64:
    """
    get vector to B from A's persective
    """
    v = np.zeros(3)
    for i in range(3):
        v[i] = B[i] - A[i]

    return v

@jit(nopython=True)
def grad_lj(r: np.float64) -> np.float64:
    #MANUAL ESCAPE TO PREVENT FORCE DIVERGING
    # if (r < 0.5):
    #     print("too close!")
    #     r = 0.5
    delta = 0 # was added in for debugging, keep at zero
    shift = 0.039 # the shift to make the force go to zero at rc
    return 4 * ((12*(r+delta)**(-13) - 6*(r+delta)**(-7))) + shift


# @jit(nopython=True)
def initialize(N: int, V: np.float64, T: np.float64) -> np.array:
    """
    Takes in the parameters for a simulation and returns the initial coordinates
    and velocity vectors for each of the particles such that 
        C[i][0:3]: position vec for particle i
        C[i][3:6]: velocity vec for particle i

    IN:
        N: number of particles
        V: volume of the box for the minimum image convention
        T: initial temperature for determining velocities
    OUT:
        C: a Nx6 array of the vectors described above

    ~~~ needs to be fixed for double counting and proper lattice 
        idexing !! ~~~
    """
    # make sure we divide the box into enough vertices for the particles
    div = int(1.5 * (N**(1/3)))
    lattice_len = (V**(1/3))  / div

    # print(lattice_len)
    num_vert = div - 1 

    occupied_list = np.zeros(shape=(num_vert,num_vert,num_vert))

    # make initial coordinate vector
    C = np.zeros(shape=(N,6))
# make sure that no place on the lattice gets 
    # filled twice
    count = 0
    while count < N:
        #vertex = np.zeros(3)
        x_index = np.random.randint(1,num_vert)
        y_index = np.random.randint(1,num_vert)
        z_index = np.random.randint(1,num_vert)
        if (occupied_list[x_index][y_index][z_index] == 1.0):
            count -= 1
        else:
            occupied_list[x_index][y_index][z_index] = 1.0
            C[count][0] = x_index * lattice_len
            C[count][1] = y_index * lattice_len
            C[count][2] = z_index * lattice_len

        # for j in range(3):
        #     C[count][j] = np.random.randint(1,div) * lattice_len
        count += 1 
        # add progress counter if necessary
        # print(f"{100*count/N}%")


    # now make initial velocity distribution
    for i in range(N):

        # determine magnitute of the vector
        # sample from distribution
        vs = np.sqrt(T) * np.random.normal(0,1,3)
        for j in range(3,6):
            C[i][j] = vs[j-3]
            
    
    vel_sum = np.zeros(3)
    for i in range(N):
        vel_sum += C[i][3:6]

    vel_sum = vel_sum / N

    for i in range(N):
        C[i][3:6] = C[i][3:6] - vel_sum

    return C 

@jit(nopython=True)
def init_2(N: int, V:np.float64, T: np.float64) -> np.array:
    """
    an alternate attempt at initialization
    """
    div = int(N**(1/3)) + 1 
    lattice_len = V**(1/3) / div
    # print(f"lat len: {lattice_len}")
    num_vert = div - 1 
    occupied_list = np.zeros(shape=(num_vert,num_vert,num_vert))

    C = np.zeros(shape=(N,6))

    xs = np.arange(0,div)
    ys = np.arange(0,div)
    zs = np.arange(0,div)

    count = 0 
    for x in xs:
        for y in ys:
            for z in zs:
                if (count < N):
                    # test adding a bit of a buffer here
                    C[count][0] = x * (lattice_len)# * 1.02)
                    C[count][1] = y * (lattice_len)# * 1.02)
                    C[count][2] = z * (lattice_len)# * 1.02)
                    count += 1 
                else:
                    continue


    # now make initial velocity distribution
    for i in range(N):

        # determine magnitute of the vector
        # sample from distribution
        vs = np.sqrt(T) * np.random.normal(0,1,3)
        for j in range(3,6):
            C[i][j] = vs[j-3]
            
    
    vel_sum = np.zeros(3)
    for i in range(N):
        vel_sum += C[i][3:6]

    vel_sum = vel_sum / N

    for i in range(N):
        C[i][3:6] = C[i][3:6] - vel_sum

    return C 


@jit(nopython=True)
def image_tracker(C: np.array, V: np.float64) -> np.array:
    """
    An adaptation of the 'image_tracker' from PSET 1 to work with the 
    new coordinate dimensions and the new problem statement. 
    This should be used in the making of the neighbor/cell list

    IN:
        C: real space geometry of the system
        V: volue of cubic box for the virtual image
    OUT:
        virtual_C: a collection of the coordinates in the 
        virual box
    """
    N = len(C)
    trackers = np.zeros(shape=(N,3))
    box_len = V**(1/3)

    # for i in range(N):
    #     x_track = int(C[i][0]/box_len)
    #     y_track = int(C[i][1]/box_len)
    #     z_track = int(C[i][2]/box_len)
    #     trackers[i] = np.array([x_track,y_track,z_track])

    for i in range(N):
        for j in range(3):
            if (C[i][j] >= 0.0):
                trackers[i][j] = int(C[i][j]/box_len)
            elif (C[i][j] <= -box_len):
                # print("found!")
                trackers[i][j] = int(C[i][j]/box_len) - 1 
            else:
                trackers[i][j] = -1

    virtual_C = np.zeros(shape=(N,6))
    # keep the velocities the same
    for i in range(N):
        for j in range(3,6):
            virtual_C[i][j] = C[i][j]
    
    for i in range(N):
        for j in range(3):
            virtual_C[i][j] = C[i][j] - box_len*trackers[i][j]
            # if (trackers[i][j] >= 0):
            #     virtual_C[i][j] = C[i][j] - box_len*trackers[i][j]
            # else:
            #     virtual_C[i][j] = box_len + (C[i][j] - box_len*trackers[i][j])

    return virtual_C


class ID_LIST:
    """
    ~~~ DEPRECATED ~~~
    A prototype for dealing with ID lists for each cell (ie 
    which cell has what beads in it and for the neighborlist 
    of each cell

    It is meant to be a fixed length container with variable length 
    elements at each index
    """
    def __init__(self, length):
        self.length = length
        self.arr = np.empty(length, dtype=object)
        for i in range(length):
            self.arr[i] = set([])

    def add_elem(self, i, a):
        self.arr[i].add(a)

    def add_set(self, i, a):
        self.arr[i].update(a)
    # get the set of members belonging to a given ID in the list
    def get_members(self, i):
        return self.arr[i]

    # should always know what the length is but I'll 
    # include this anyways
    def get_length(self):
        return self.length

    def find_particle(self, part_id):
        """
        want to find which cell has a particle of interest
        returns an integer id
        """

        for i in range(self.get_length()):
            if (len(self.arr[i]) == 0):
                continue
            if (part_id in self.get_members(i)):
                return i
        
        return -1




# @jit(nopython=True)
def old_neighbor_list(C: np.array, r_buff: np.float64, V: np.float64) -> np.array:
    """
    ~~~ DEPRECATED ~~~
    Takes in a real geometry C then finds the virutal positions 
    of all particles wihitn the sample volume V 

    from here 
    """
    N = len(C)
    box_len = V**(1/3)
    rough_cell_len = box_len / rc
    cell_per_row = int(rough_cell_len)
    print(f"cpr: {cell_per_row}")
    cell_len = box_len / cell_per_row
    print(f"cell len: {cell_len}")
    virtual_C = image_tracker(C, V)
    
    # the first index of the neighbor list 
    # corresponds to the cell ID number (its position in the 
    # C array)
    # neighbor_list = []
    # for i in range(N):
    #     neighbor_list.append([])

    # find the number of cells needed to cover the entire 
    # virtual box 
    # cell_vol = cell_len**3
    # the cells may not exactly line up with the box,
    # but that is ok? since there is a degree of freedom in the size 
    # of the cells
    # cell_per_row = int(box_len / cell_len) + 1
    M = cell_per_row
    n_cells = (cell_per_row) ** 3
    # initialize an ID list for the cells - ie which particles
    # live in a certain cell at the current time
    cell_ID_list = ID_LIST(n_cells)

    # loop over each bead and see what cell it falls into
    for i in range(N):
        # spatial coords of ith particle
        coords = virtual_C[i][0:3]
        x_ind = int(coords[0] / cell_len)
        y_ind = int(coords[1] / cell_len)
        z_ind = int(coords[2] / cell_len)

        cell_index = (cell_per_row**2 - 1)*z_ind +(cell_per_row-1)*y_ind +x_ind

        # add the number of the particle to the corresponding cell ID
        cell_ID_list.add_elem(cell_index,i)

    
    # cell list has been constructed
    # now make neighbor list

    neighbor_list = ID_LIST(N)

    z_levels = [-M**2, 0, M**2]
    y_levels = [-M, 0, M]
    x_levels = [-1, 0, 1]

    for i in range(N):
        j = cell_ID_list.find_particle(i)
        members = cell_ID_list.get_members(j)
        
        for xlv in x_levels:
            for ylv in y_levels:
                for zlv in z_levels:
                    members.update(cell_ID_list.get_members(j+xlv+ylv+zlv))


        members.remove(i)
        neighbor_list.add_set(i,members)



    return neighbor_list


def neighbor_list(C: np.array, r_buff: np.float64, V:np.float64)->np.array:
    """
    an attempt at a redo of the neighborlist
    """
    N = len(C)
    box_len = V**(1/3)
    rough_cell_count = box_len / rc # global constant
    cell_per_row = int(rough_cell_count)
    # print(f"cpr: {cell_per_row}")
    cell_len = box_len / cell_per_row
    # print(f"cell len: {cell_len}")

    # obtain particle images
    virtual_C = image_tracker(C,V)
    # has to be run multiple times ??
    # for i in range(4):
    #     virtual_C = image_tracker(virtual_C,V)

    # give  a less verbose name
    M = cell_per_row
    # cell list - ie which particles live in what cell
    CL = np.empty(shape=(M,M,M), dtype =object)
    for i in range(M):
        for j in range(M):
            for k in range(M):
                CL[i][j][k] = set([])


    NL = np.empty(shape=(N), dtype=object)
    for i in range(N):
        NL[i] = set([])

    for i in range(N):
        # for each particle i, find it's neighbors in the cell cube
        # first find which cell it's in
        x_ind = int(virtual_C[i][0] / cell_len) #- 1 #GUESS
        y_ind = int(virtual_C[i][1] / cell_len) #- 1
        z_ind = int(virtual_C[i][2] / cell_len) #- 1

        if (x_ind >= M or y_ind >= M or z_ind >= M):
            print("Indexing error!")
            print(f"{x_ind} {y_ind} {z_ind}")
            print(f"{virtual_C[i][0]} {virtual_C[i][1]} {virtual_C[i][2]}")
            raise SystemExit
        elif (x_ind < 0 or y_ind < 0 or z_ind < 0):
            print("indexing error!")
            print(f"{x_ind} {y_ind} {z_ind}")
            print(f"{virtual_C[i][0]} {virtual_C[i][1]} {virtual_C[i][2]}")
            raise SystemExit

        CL[x_ind][y_ind][z_ind].add(i)


    for i in range(N):

        # the local neighbor list for cell i
        loc_nl = set([])
        # find cell index 
        # x_ind = 0
        # y_ind = 0
        # z_ind = 0
        for j in range(M):
            for k in range(M):
                for l in range(M):
                    if (i in CL[j][k][l]):
                        x_ind = j
                        y_ind = k
                        z_ind = l
        # make the boundaries of the adjacent cells 
        xn = [-1,0,1]
        yn = [-1,0,1]
        zn = [-1,0,1]
        if (x_ind == 0):
            xn = [0,1]
        elif (x_ind == M-1):
            xn = [-1,0]
        if (y_ind == 0):
            yn = [0,1]
        elif(y_ind == M - 1):
            yn = [-1,0]
        if (z_ind == 0):
            zn = [0,1]
        elif (z_ind == M-1):
            zn = [-1,0]

        potential_neighbors = set([])
        for xb in xn:
            for yb in yn:
                for zb in zn:
                    potential_neighbors.update(CL[xb][yb][zb])

        ptn_copy = potential_neighbors.copy()
        for pn in potential_neighbors:
            nn_vec = nearest_neighbor_vec(virtual_C[i],virtual_C[pn],V)
            r = mag_of_vec(nn_vec)
            if (r > rc):
                ptn_copy.remove(pn)

            if (pn == i):
                ptn_copy.discard(pn)




        NL[i].update(ptn_copy)
        NL[i].discard(i)

    return NL

# @jit(nopython=True)
def nl_2(C:np.array,rb:np.float64,V:np.float64) -> np.array:
    """
    The third attempt at making a functioning an correct neighborlist

    C: input geometry in real space
    rb: buffer distance
    V: volume of the simulation
    """
    N = len(C)
    L = V**(1/3)
    rough_cell_count= L / rc # rc defined globally, see top of file
    cell_per_row = int(rough_cell_count)
    cell_len = L / cell_per_row

    virtual_C = image_tracker(C,V)

    # less verbose name
    M = cell_per_row

    # initialize the cell list as array of empty sets
    CL = np.empty(shape=(M,M,M),dtype=object)
    for i in range(M):
        for j in range(M):
            for k in range(M):
                CL[i][j][k] = set([])

    NL = np.empty(shape=(N),dtype=object)
    for i in range(N):
        NL[i] = set([])

    for i in range(N):
        # make 3 indexes to place the cell
        x_ind = int(virtual_C[i][0] / cell_len)
        y_ind = int(virtual_C[i][1] / cell_len)
        z_ind = int(virtual_C[i][2] / cell_len)


        # report if there are any intial indexing issues
        if (x_ind >= M or y_ind >= M or z_ind >= M):
            print("Indexing error!")
            print(f"{x_ind} {y_ind} {z_ind}")
            print(f"{virtual_C[i][0]} {virtual_C[i][1]} {virtual_C[i][2]}")
            raise SystemExit
        elif (x_ind < 0 or y_ind < 0 or z_ind < 0):
            print("indexing error!")
            print(f"{x_ind} {y_ind} {z_ind}")
            print(f"{virtual_C[i][0]} {virtual_C[i][1]} {virtual_C[i][2]}")
            raise SystemExit
        
        CL[x_ind][y_ind][z_ind].add(i)

    for i in range(N):
        # make a local neighbor list for this cell
        loc_nl = set([])

        # set these to dummy values that should get changed
        x_ind = -1
        y_ind = -1
        z_ind = -1

        for j in range(M):
            for k in range(M):
                for l in range(M):
                    if (i in CL[j][k][l]):
                        x_ind = j
                        y_ind = k 
                        z_ind = l 


        # if they are not changed, then report an error
        if (x_ind == -1 or y_ind == -1 or z_ind == -1):
            print("neighbor list indexing error!")

        # set default boundaries for the cell cube
        xn = [-1,0,1]
        yn = [-1,0,1]
        zn = [-1,0,1]
        # will be adding these to the indexes from above
        if (x_ind == 0):
            xn = [0,1,M-1]
        elif (x_ind == M-1):
            xn = [-1,0,-(M-1)]
        if (y_ind == 0):
            yn = [0,1,M-1]
        elif(y_ind == M - 1):
            yn = [-1,0,-(M-1)]
        if (z_ind == 0):
            zn = [0,1,M-1]
        elif (z_ind == M-1):
            zn = [-1,0,-(M-1)]


        potential_neighbors = set([])
        for xb in xn:
            for yb in yn:
                for zb in zn:
                    potential_neighbors.update(CL[xb+x_ind][yb+y_ind][zb+z_ind])


        ptn_copy = potential_neighbors.copy()
        for pn in potential_neighbors:
            nn_vec = nearest_neighbor_vec(virtual_C[i],virtual_C[pn],V)
            r = mag_of_vec(nn_vec)
            if (r > rc):
                ptn_copy.remove(pn)

            if (pn == i):
                ptn_copy.discard(pn)




        NL[i].update(ptn_copy)
        NL[i].discard(i)

    return NL



# @jit(nopython=True)
def force_calc(C:np.array, NL:np.array, V:np.float64) -> [np.array,np.float64]:
    """
    Find the forces acting on each of the particles in the geometry
    C: real space geometry
    NL: neighbor list in virtual space
    V: volume
    """
    N = len(C)
    virtual_C = image_tracker(C, V)
    # initialize force array
    F = np.zeros(shape=(N,3),dtype=np.float64)

    # total potential energy, will need to divide by 1/2 to 
    # avoid double counting
    U = np.float64(0.0)


    for i in range(N):
        # set of neigbors for this particle
        nset = NL[i]

        for n in nset:
            if (i != n):
                # r = get_dist(virtual_C[i],virtual_C[n])
                # vec = get_dvec(virtual_C[i],virtual_C[n]) / r
                
                # find nearest neighbor vec
                nn_vec = nearest_neighbor_vec(virtual_C[i],virtual_C[n],V)
                r = mag_of_vec(nn_vec)
                if (r < 1e-3):
                    print("Too close!")

                U += ljts_potential(r)

                f_mag = grad_lj(r)
                F[i] += -f_mag * (nn_vec / r)

    U = U / np.float64(2.0)
    return F, U

@jit(nopython=True)
def NVE_int_1(C:np.array, F:np.array, ref_C: np.array, V:np.float64, rb:np.float64) -> [np.array,np.array]:
    """
    Takes in the real space coordinates and the current velocityof the 
    geometry and the forces acting on the coorindates in virtual space. 

    Step 1 of a two step integration. This does the first half step of 
    v and the full step for r. 
    It then returns the partially updated C and the maximum displacements


    ~~~ Using the method found in "Computer Simulation of Liquids" 
    (Allen and Tildesley, Oxford (2017))  ~~~

    C: real space p and q vec
    F: total force vectors acting on each element
    ref_C: the reference geometry at the time of the last neighborlist
        construction
    V: volume
    rb: buffer radius, used for determining when the neighborlist needs
    to be corrected
    """
    dt  = 1e-3
    N = len(C)
    # virtual_C = image_tracker(C)

    # TESTING THE BELOW LINE, REMOVE IF NOT CORRECT
    # C = image_tracker(C,V)


    # find the v(t + 1/2 \delta t) 
    for i in range(N):
        for j in range(3,6):
            # the C and F arrays are indexed slightly differently
            C[i][j] = C[i][j] + 0.5 * dt * F[i][j-3]

    
    # also need to implement a new call on the neighborlist
    # a new list is needed when the two largest displacements since the 
    # time of the last neighborlist construction exceed r_buff

    # first find the new r coordinates
    r_deltas = np.zeros(N)
    for i in range(N):
        rd = np.float64(0.0)
        for j in range(0,3):
            # add the half step velocity to the current position
            C[i][j] = C[i][j] + dt * C[i][j+3]
            rd += (dt * C[i][j+3])**2
        #print(rd)
        r_deltas[i] = np.sqrt(rd)

    r_deltas = np.sort(r_deltas)
    if (r_deltas[-1] + r_deltas[-2] > rb):
        return C, True
    else:
        return C, False

    # now have to find the new forces acting on 

@jit(nopython=True)
def NVE_int_2(C:np.array, F:np.array, V: np.float64, rb: np.float64) -> np.float64: 
    """
    The second half of the NVE integrator. It takes in a C with new q / r 
    but half updated p / v. Depending on wheter the last stage flagged it,
    the neighbor list will be updated

    This returns a fully updated C with the last half of the velocities
    accounted for
    """
    N = len(C) 
    # don't need to put into virtual space since the force calculation
    # should have done that itself 

    for i in range(N):
        for j in range(3,6):
            # F and v of C are indexed differently (off by 3)
            C[i][j] = C[i][j] + 0.5 * dt * F[i][j-3]


    # that is all that is needed
    return C


# @jit(nopython=True)
def NVE_full(N:int,p:np.float64,T:np.float64,rb:np.float64,ITA:int):
    """
    An attempt at putting the whole process from initization to integration in
    one place. 

    Time step is defined globally at the top of the file

    Returns arrays of the potential and kinetic energies over iterations ITA

    N: number of particles
    p: density 
    T: temperature
    rb: buffer dist
    ITA: iterations
    """
    # dt = 1e-3
    dt = 1e-2
    # find the volume of the simulation
    V = N / p

    # initialize coordinates on lattice
    C = init_2(N,V,T)
    # a reference geometry to be compared to 
    ref_C = C.copy() 

    # initialize the neighborlist
    NL = nl_2(C, rb, V)

    # find initial force vector
    F, U0 = force_calc(C,NL,V)

    # second term is a debugging tool which only tracks the first 
    # particle
    K_init, K0_init = get_kinetic(C)
    
    # inialize energies list
    potentials = [U0]
    kinetics = [K_init]
    single_k = [K0_init]

    # flag to update the neighbor list
    update = False

    for ita in range(ITA):
        # if ( ita%5 == 0 ):
            # will commenting this line allow jit?
            # print(f"On iteration {ita+1} of {ITA}")

        # update the firsrt half of the velocities
        for i in range(N):
            for j in range(3,6):
                C[i][j] = C[i][j] + 0.5 * dt * F[i][j-3]

        # update the spatial coordinates

        r_deltas = np.zeros(N)

        for i in range(N):
            rd = np.float64(0.0)
            for j in range(3):
                C[i][j] = C[i][j] + dt * C[i][j+3]
                rd += (C[i][j] - ref_C[i][j])**2

            r_deltas[i] = np.sqrt(rd)


        r_deltas = np.sort(r_deltas)
        if (r_deltas[-1] + r_deltas[-2] > rb):
            print("updating neighborlist...")
            ref_C = C.copy()
            NL = nl_2(C,rb,V)

        F, U = force_calc(C,NL,V)
        potentials.append(U)


        # do the second half of the velocity integration
        for i in range(N):
            for j in range(3,6):
                C[i][j] = C[i][j] + 0.5 * dt * F[i][j-3]

        K, k = get_kinetic(C)
        kinetics.append(K)
        single_k.append(k)

    potentials = np.array(potentials)
    kinetics = np.array(kinetics)
    single_k = np.array(single_k)

    print("Done!")
    return potentials, kinetics, single_k, C

# @jit(nopython=True)
def NVE_MSD(N:int,p:np.float64,T:np.float64,rb:np.float64,ITA:int):
    """
    An attempt at putting the whole process from initization to integration in
    one place. 

    Time step is defined globally at the top of the file

    Returns arrays of the potential and kinetic energies over iterations ITA

    N: number of particles
    p: density 
    T: temperature
    rb: buffer dist
    ITA: iterations
    """
    dt = 1e-3
    # find the volume of the simulation
    V = N / p

    # initialize coordinates on lattice
    C = init_2(N,V,T)
    # a reference geometry to be compared to 
    ref_C = C.copy() 
    initial_C = C.copy()

    # initialize the neighborlist
    NL = nl_2(C, rb, V)

    # find initial force vector
    F, U0 = force_calc(C,NL,V)

    # second term is a debugging tool which only tracks the first 
    # particle
    K_init = get_kinetic(C)

    # MSD LIST
    msds = [np.float64(0.0)]
    
    # inialize energies list
    potentials = [U0]
    kinetics = [K_init]
    # single_k = [K0_init]

    # flag to update the neighbor list
    update = False

    for ita in range(ITA):
        if ( ita%50 == 0 ):
            # will commenting this line allow jit?
            # print(f"On iteration {ita+1} of {ITA}")
            print("On " + str(ita + 1) + " of " + str(ITA))

        # update the firsrt half of the velocities
        for i in range(N):
            for j in range(3,6):
                C[i][j] = C[i][j] + 0.5 * dt * F[i][j-3]

        # update the spatial coordinates

        r_deltas = np.zeros(N)

        for i in range(N):
            rd = np.float64(0.0)
            for j in range(3):
                C[i][j] = C[i][j] + dt * C[i][j+3]
                rd += (C[i][j] - ref_C[i][j])**2

            r_deltas[i] = np.sqrt(rd)


        r_deltas = np.sort(r_deltas)
        if (r_deltas[-1] + r_deltas[-2] > rb):
            print("updating neighborlist...")
            ref_C = C.copy()
            NL = nl_2(C,rb,V)

        # update MSD list
        msds.append(MSD(initial_C,C))


        F, U = force_calc(C,NL,V)
        potentials.append(U)


        # do the second half of the velocity integration
        for i in range(N):
            for j in range(3,6):
                C[i][j] = C[i][j] + 0.5 * dt * F[i][j-3]

        K = get_kinetic(C)
        kinetics.append(K)
        # single_k.append(k)

    potentials = np.array(potentials)
    kinetics = np.array(kinetics)
    temps =  2 * kinetics / (3*N)
    msds = np.array(msds)
    # single_k = np.array(single_k)

    print("Done!")
    return potentials, kinetics, temps, msds, C


@jit(nopython=True)
def get_G1(C:np.array,V:np.float64,T:np.float64,Q1:np.float64) -> np.float64:
    """
    The G1 term in the Nosé-Hoover integration scheme.

    Takes in coordinates C, volume V, and temperature T
    """
    G1 = np.float64(0.0)
    N = len(C)
    L = 3*N #V**(1/3)

    for i in range(N):
        for j in range(3):
            G1 += C[i][j+3] ** 2

    G1 -= L*T
    G1 = G1 / Q1
    return G1
    

@jit(nopython=True)
def get_G2(v_xi_1:np.float64,Q1:np.float64,Q2:np.float64,T:np.float64) -> np.float64:
    """
    method to get the G2 term for the NVT integrator
    """
    return (1/Q2) * (Q1*v_xi_1**2 - T) 



# @jit(nopython=True)
def NVT_full(N:int,p:np.float64,T:np.float64,rb:np.float64,ITA:int):
    """
    A test implementation of the Nose-Hoover Thermostat method to create an
    NVT simulation (also with periodic boundaries as above) 
    
    N: number of particles
    p: density 
    T: temperature
    rb: buffer dist
    ITA: iterations
    """
    print("starting")
    dt = 1e-2

    # find volume 
    V = N / p
    L = V**(1/3)

    # initialize coordinates
    C = init_2(N,V,T)

    ref_C = C.copy()

    # initialize the neighbotlist
    NL = nl_2(C,rb,V)

    # find initial force vector
    F, U0 = force_calc(C,NL,V)

    # second term is a debugging tool which only tracks the first 
    # particle
    K_init = get_kinetic(C)
    
    # inialize energies list
    potentials = [U0]
    kinetics = [K_init]
    # temperature list
    temps = [2* K_init / (3*N)]

    # check the directionality of T
    t_components =  [get_T_components(C)]


    # flag to update the neighbor list
    update = False

    # gets complicated from here

    # set the values for guess and check G1 and G2, first define 
    # Q1 and Q2  with trial values
    Q1 = 1
    Q2 = 1

    # again, make intital values based on a guess
    # p_xi_1 = 1.0
    # p_xi_2 = 1.0
    v_xi_1 = 1.0
    v_xi_2 = 1.0
    xi_1 = 1.0 # verify that these can be guessed too
    xi_2 = 1.0
    # these are supposed to approach a convergent value

    # add terms to nose-H 
    nose_H = []
    # dont need to add time term here since the time is currently 0
    nose_H.append((xi_1**2 * Q1)/2 + (xi_2**2 * Q2)/2)

    G1 = get_G1(C, V, T, Q1)
    G2 = get_G2(v_xi_1,Q1,Q2,T) 

    for ita in range(ITA):
        # not using f strings to I can try jit
        if (ita % 50 == 0):
            print(str(ita+1) + " of " + str(ITA))
        # now follow the checklsit to update the system

        # step 1  + 
        v_xi_2 = v_xi_2 + G2 * dt / 4.

        #step 2 + 
        v_xi_1 = np.exp(-v_xi_2 * dt / 8.) * v_xi_1

        # step 3 +
        v_xi_1 = v_xi_1 +  G1 * dt / 4.
        
        # step 4
        v_xi_1 = np.exp(-v_xi_2 * dt / 8.) * v_xi_1

        # step 5
        xi_1 = xi_1 - v_xi_1 * dt / 2
        xi_2 = xi_2 - v_xi_2 * dt / 2

        # step 6
        for i in range(N):
            C[i][3:6] = C[i][3:6] * np.exp(-v_xi_1 * dt / 2)

        # step 7
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 8
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4.

        # step 9
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 10
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2 + G2 * dt / 4.

        # step 11
        F, U = force_calc(C,NL,V)
        for i in range(N):
            C[i][3:6] = C[i][3:6] + F[i] * dt / 2

        # step 12 - update the spatial coordinates
        r_deltas = np.zeros(N)
        for i in range(N):
            rd = np.float64(0.0)
            C[i][0:3] = C[i][0:3] + C[i][3:6] * dt
            rd += np.sum((C[i][0:3] - ref_C[i][0:3])**2)
            r_deltas[i] = np.sqrt(rd)

        r_deltas = np.sort(r_deltas)
        if (r_deltas[-1] + r_deltas[-2] > rb):
            print("updating neighborlist...")
            ref_C = C.copy()
            NL = nl_2(C,rb,V)


        # step 13 - updated spatial coordinates give new U
        F, U = force_calc(C,NL,V)
        potentials.append(U)
        for i in range(N):
            C[i][3:6] = C[i][3:6] +  F[i] * dt / 2
            
        # step 14
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2  + G2 * dt / 4

        # step 15
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 16
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4

        # step 17
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 18
        xi_1 = xi_1 - v_xi_1 * dt / 2
        xi_2 = xi_2 - v_xi_2 * dt / 2

        # step 19 - update v then K (validate this is is right step to do this)
        for i in range(N):
            C[i][3:6] = np.exp(-v_xi_1 * dt / 2) * C[i][3:6]
        K = get_kinetic(C)
        kinetics.append(K)
        temps.append(2* K / (3*N))
        # t_components.append(get_T_components(C))


        # step 20
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 21
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4
        # v_xi_1 = p_xi_1 * Q1

        # step 22
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 23
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2 + G2 * dt / 4
        # v_xi_2 = p_xi_2 * Q2

        # update the nose hoover addition to the Hamiltonian
        add = 0.5*(xi_1**2 *Q1 + xi_2**2 *Q2) + L*temps[-1]*(ita+1)*dt*(xi_1+xi_2)
        nose_H.append(add)


    # find the temperature list
    # temps = np.array(kinetics) / float((3*N - 3))
    potentials = np.array(potentials)
    kinetics = np.array(kinetics)
    temps = np.array(temps)
    t_components = np.array(t_components)
    nose_H = np.array(nose_H)

    print("done!")
    return potentials, kinetics, temps, t_components, nose_H, C


def NVT_full_no_nl(N:int,p:np.float64,T:np.float64,rb:np.float64,ITA:int):
    """
    IMPLEMENTATION WITHOUT THE NEIGHBORLIST (ie. the neighborlist is 
    a constant and is the full set of particles, still subject to r_C)

    A test implementation of the Nose-Hoover Thermostat method to create an
    NVT simulation (also with periodic boundaries as above) 
    
    N: number of particles
    p: density 
    T: temperature
    rb: buffer dist
    ITA: iterations
    """
    dt = 1e-2
    print("starting")

    # find volume 
    V = N / p
    L = V**(1/3)

    # initialize coordinates
    C = init_2(N,V,T)

    ref_C = C.copy()

    # initialize the neighbotlist to include all the particles
    NL = np.empty(shape=(N),dtype=object)
    for i in range(N):
        NL[i] = set(np.arange(0,N))

    # find initial force vector
    F, U0 = force_calc(C,NL,V)

    # second term is a debugging tool which only tracks the first 
    # particle
    K_init = get_kinetic(C)
    
    # inialize energies list
    potentials = [U0]
    kinetics = [K_init]
    # temperature list
    temps = [2* K_init / (3*N)]

    # check the directionality of T
    t_components =  [get_T_components(C)]


    # flag to update the neighbor list
    update = False

    # gets complicated from here

    # set the values for guess and check G1 and G2, first define 
    # Q1 and Q2  with trial values
    Q1 = 1
    Q2 = 1

    # again, make intital values based on a guess
    # p_xi_1 = 1.0
    # p_xi_2 = 1.0
    v_xi_1 = 1.0
    v_xi_2 = 1.0
    xi_1 = 1.0 # verify that these can be guessed too
    xi_2 = 1.0
    # these are supposed to approach a convergent value

    # add terms to nose-H 
    nose_H = []
    # dont need to add time term here since the time is currently 0
    nose_H.append((xi_1**2 * Q1)/2 + (xi_2**2 * Q2)/2)

    G1 = get_G1(C, V, T, Q1)
    G2 = get_G2(v_xi_1,Q1,Q2,T) 

    for ita in range(ITA):
        # not using f strings to I can try jit
        if (ita % 50 == 0):
            print(str(ita+1) + " of " + str(ITA))
        # now follow the checklsit to update the system

        # step 1  + 
        v_xi_2 = v_xi_2 + G2 * dt / 4.

        #step 2 + 
        v_xi_1 = np.exp(-v_xi_2 * dt / 8.) * v_xi_1

        # step 3 +
        v_xi_1 = v_xi_1 +  G1 * dt / 4.
        
        # step 4
        v_xi_1 = np.exp(-v_xi_2 * dt / 8.) * v_xi_1

        # step 5
        xi_1 = xi_1 - v_xi_1 * dt / 2
        xi_2 = xi_2 - v_xi_2 * dt / 2

        # step 6
        for i in range(N):
            C[i][3:6] = C[i][3:6] * np.exp(-v_xi_1 * dt / 2)

        # step 7
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 8
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4.

        # step 9
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 10
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2 + G2 * dt / 4.

        # step 11
        F, U = force_calc(C,NL,V)
        for i in range(N):
            C[i][3:6] = C[i][3:6] + F[i] * dt / 2

        # step 12 - update the spatial coordinates
        # r_deltas = np.zeros(N)
        for i in range(N):
            # rd = np.float64(0.0)
            C[i][0:3] = C[i][0:3] + C[i][3:6] * dt
            # rd += np.sum((C[i][0:3] - ref_C[i][0:3])**2)
            # r_deltas[i] = np.sqrt(rd)

        # r_deltas = np.sort(r_deltas)
        # if (r_deltas[-1] + r_deltas[-2] > rb):
        #     print("updating neighborlist...")
        #     ref_C = C.copy()
        #     NL = nl_2(C,rb,V)


        # step 13 - updated spatial coordinates give new U
        F, U = force_calc(C,NL,V)
        potentials.append(U)
        for i in range(N):
            C[i][3:6] = C[i][3:6] +  F[i] * dt / 2
            
        # step 14
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2  + G2 * dt / 4

        # step 15
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 16
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4

        # step 17
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 18
        xi_1 = xi_1 - v_xi_1 * dt / 2
        xi_2 = xi_2 - v_xi_2 * dt / 2

        # step 19 - update v then K (validate this is is right step to do this)
        for i in range(N):
            C[i][3:6] = np.exp(-v_xi_1 * dt / 2) * C[i][3:6]
        K = get_kinetic(C)
        kinetics.append(K)
        temps.append(2* K / (3*N))
        # t_components.append(get_T_components(C))


        # step 20
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 21
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4
        # v_xi_1 = p_xi_1 * Q1

        # step 22
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 23
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2 + G2 * dt / 4
        # v_xi_2 = p_xi_2 * Q2

        # update the nose hoover addition to the Hamiltonian
        add = 0.5*(xi_1**2 *Q1 + xi_2**2 *Q2) + L*temps[-1]*(ita+1)*dt*(xi_1+xi_2)
        nose_H.append(add)


    # find the temperature list
    # temps = np.array(kinetics) / float((3*N - 3))
    potentials = np.array(potentials)
    kinetics = np.array(kinetics)
    temps = np.array(temps)
    t_components = np.array(t_components)
    nose_H = np.array(nose_H)

    print("done!")
    return potentials, kinetics, temps, t_components, nose_H, C



def NVT_MSD(N:int,p:np.float64,T:np.float64,rb:np.float64,xi:np.float64,ITA:int):
    """
    A test implementation of the Nose-Hoover Thermostat method to create an
    NVT simulation (also with periodic boundaries as above) with the 
    addition of tracking the MSD of the system
    
    N: number of particles
    p: density 
    T: temperature
    rb: buffer dist
    xi: The friction constants to be used for the thermal mass
    ITA: iterations
    """
    print("starting")
    dt = 1e-2

    # find volume 
    V = N / p
    L = V**(1/3)

    # initialize coordinates
    C = init_2(N,V,T)

    # make reference and initial C's
    ref_C = C.copy()
    initial_C = C.copy()

    # initialize the neighbotlist
    NL = nl_2(C,rb,V)

    # find initial force vector
    F, U0 = force_calc(C,NL,V)

    # second term is a debugging tool which only tracks the first 
    # particle
    K_init = get_kinetic(C)
    
    # inialize energies list
    potentials = [U0]
    kinetics = [K_init]
    # temperature list
    temps = [2* K_init / (3*N)]

    # check the directionality of T
    t_components =  [get_T_components(C)]

    # MSD list
    msds = [np.float64(0.0)]

    # flag to update the neighbor list
    update = False

    # gets complicated from here

    # set the values for guess and check G1 and G2, first define 
    # Q1 and Q2  with trial values
    Q1 = 1
    Q2 = 1

    # again, make intital values based on a guess
    # p_xi_1 = 1.0
    # p_xi_2 = 1.0
    v_xi_1 = 1.0
    v_xi_2 = 1.0
    xi_1 = xi 
    xi_2 = xi
    # these are supposed to approach a convergent value

    # add terms to nose-H 
    nose_H = []
    # dont need to add time term here since the time is currently 0
    nose_H.append((xi_1**2 * Q1)/2 + (xi_2**2 * Q2)/2)

    G1 = get_G1(C, V, T, Q1)
    G2 = get_G2(v_xi_1,Q1,Q2,T) 

    for ita in range(ITA):
        # not using f strings to I can try jit
        if (ita % 50 == 0):
            print(str(ita+1) + " of " + str(ITA))
        # now follow the checklsit to update the system

        # step 1  + 
        v_xi_2 = v_xi_2 + G2 * dt / 4.

        #step 2 + 
        v_xi_1 = np.exp(-v_xi_2 * dt / 8.) * v_xi_1

        # step 3 +
        v_xi_1 = v_xi_1 +  G1 * dt / 4.
        
        # step 4
        v_xi_1 = np.exp(-v_xi_2 * dt / 8.) * v_xi_1

        # step 5
        xi_1 = xi_1 - v_xi_1 * dt / 2
        xi_2 = xi_2 - v_xi_2 * dt / 2

        # step 6
        for i in range(N):
            C[i][3:6] = C[i][3:6] * np.exp(-v_xi_1 * dt / 2)

        # step 7
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 8
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4.

        # step 9
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 10
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2 + G2 * dt / 4.

        # step 11
        F, U = force_calc(C,NL,V)
        for i in range(N):
            C[i][3:6] = C[i][3:6] + F[i] * dt / 2

        # step 12 - update the spatial coordinates
        r_deltas = np.zeros(N)
        for i in range(N):
            rd = np.float64(0.0)
            C[i][0:3] = C[i][0:3] + C[i][3:6] * dt
            rd += np.sum((C[i][0:3] - ref_C[i][0:3])**2)
            r_deltas[i] = np.sqrt(rd)

        r_deltas = np.sort(r_deltas)
        if (r_deltas[-1] + r_deltas[-2] > rb):
            print("updating neighborlist...")
            ref_C = C.copy()
            NL = nl_2(C,rb,V)


        # step 12 A - add the current MSD to list
        # (need to get out of the habit of using lists when they 
        #  are not necessary) 
        msds.append(MSD(initial_C,C))


        # step 13 - updated spatial coordinates give new U
        F, U = force_calc(C,NL,V)
        potentials.append(U)
        for i in range(N):
            C[i][3:6] = C[i][3:6] +  F[i] * dt / 2
            
        # step 14
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2  + G2 * dt / 4

        # step 15
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 16
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4

        # step 17
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 18
        xi_1 = xi_1 - v_xi_1 * dt / 2
        xi_2 = xi_2 - v_xi_2 * dt / 2

        # step 19 - update v then K (validate this is is right step to do this)
        for i in range(N):
            C[i][3:6] = np.exp(-v_xi_1 * dt / 2) * C[i][3:6]
        K = get_kinetic(C)
        kinetics.append(K)
        temps.append(2* K / (3*N))
        # t_components.append(get_T_components(C))


        # step 20
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 21
        G1 = get_G1(C, V, T, Q1)
        v_xi_1 = v_xi_1 + G1 * dt / 4
        # v_xi_1 = p_xi_1 * Q1

        # step 22
        v_xi_1 = np.exp(-v_xi_2 * dt / 8) * v_xi_1

        # step 23
        G2 = get_G2(v_xi_1,Q1,Q2,T) 
        v_xi_2 = v_xi_2 + G2 * dt / 4
        # v_xi_2 = p_xi_2 * Q2

        # update the nose hoover addition to the Hamiltonian
        add = 0.5*(xi_1**2 *Q1 + xi_2**2 *Q2) + L*temps[-1]*(ita+1)*dt*(xi_1+xi_2)
        nose_H.append(add)


    # find the temperature list
    # temps = np.array(kinetics) / float((3*N - 3))
    potentials = np.array(potentials)
    kinetics = np.array(kinetics)
    temps = np.array(temps)
    msds = np.array(msds)
    # t_components = np.array(t_components)
    nose_H = np.array(nose_H)

    print("done!")
    return potentials, kinetics, temps, msds, nose_H, C






@jit(nopython=True)
def delta(diff:np.float64, tol:np.float64) -> np.float64:
    if (diff < tol):
        return 1.0
    else:
        return 0.0


# @jit(nopython=True)
def rad_dist(C:np.array, V:np.float64) -> np.float64:
    """
    Takes in a coordinate geometry and returns the radial distribution
    results across a range of r values
    """
    N = len(C)

    r_vals = np.linspace(0., 3.5,1000)
    rdf_vals = np.zeros(shape=(1000))

    tol = 3.5/ (1000*2)

    #option 1 
    for n, r in enumerate(r_vals):
        summ = 0.0
        for i in range(N-1):
            for j in range(i+1,N):
                r_loc = get_dist(C[i],C[j])
                diff = abs(r - r_loc)

                summ += delta(diff,tol)

        rdf_vals[n] = (V/N**2) * (summ/len(r_vals))
                

    # for best results, divide rdf_vals by 4pi r^2 to account for 
    # speherical character
    return r_vals, rdf_vals

def rad_dist_2(C:np.array, V:np.float64) -> np.float64:
    """
    Alternate formulation of the radial distrubtion function that includes 
    the image convention
    """
    N = len(C)

    r_vals = np.linspace(0,3.5,1000)
    rdf_vals = np.zeros(shape=(1000))

    virtual_C = image_tracker(C,V)

    tol = 3.5 / (1000*2)

    # option 2 
    for n, r in enumerate(r_vals):
        summ = 0.0
        for i in range(N-1):
            for j in range(i+1,N):
                r_loc = nearest_neighbor_vec(virtual_C[i],virtual_C[j],V)
                mag_loc = mag_of_vec(r_loc)

                diff = abs(r - mag_loc)

                summ += delta(diff,tol)
        rdf_vals[n] = (V / N**2) * (summ/len(r_vals))

    # for best results, divide rdf_vals by 4pi r^2 (if at least a high-enough
    # non-zero value) to account for spherical character
    return r_vals, rdf_vals




@jit(nopython=True)
def MSD(ref_C: np.array, C:np.array) -> np.float64:
    """
    Question 4: MSD for system vs time. 

    IN:
        ref_C: The geometry of the system at time t = 0
        C: the *real space* geometry of the system at time 
           t = t'
    OUT:
        The value of the mean squared displacement at that the time C 
        was made.
    """
    N = len(C) 
    msd = np.float64(0.0)

    for i in range(N):
        msd += (ref_C[i][0] - C[i][0]) ** 2
        msd += (ref_C[i][1] - C[i][1]) ** 2 
        msd += (ref_C[i][2] - C[i][2]) ** 2

    return  msd / N





def main():
    """
    do anything here that should be run as a stand alone operation
    when this file is called from the command line (ie not from being
    imported alone)
    """

    # do test initializer
    test = initialize(1000,1111,1.0)

    print("Done!")


if __name__ == "__main__":
    main()
