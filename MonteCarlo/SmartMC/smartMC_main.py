#! /Users/Noah/opt/anaconda3/envs/moleng37/bin/python3
"""
This is the main source file for the ``Smart Monte Carlo" term project for 
MENG 255

Author: Noah Dohrmann
Date Due: Thurs. 17 March 2022 (11:59PM CST)
"""

import numpy as np
import pandas as pd
import numba
from numpy.random import random
from numba import jit
from time import time
import datetime as dt
from numpy import random

# could consider importing code from the original MC assignment?

# # initialize and record seed selection
# date = dt.datetime.now().strftime("%a %d %b %Y %X")
# seed = int(time())
# np.random.seed(seed)

# # writes the runtime date and seed
# with open("../Output/seed_log.txt", "a") as sl:
#     sl.write(f"{date}, {seed}\n")


@jit(nopython=True)
def coord_initializer(N: int) -> np.array:
    """
    An initializer for making a new polymer with random coordinates in a chain

    IN:
        int "N": number of monomers in the chain
        float "scale": multiplier for the random coordinate generation
    OUT:
        np.array of the randomized initial coordinates for each monomer such
        that they form a loose chain in the x direction
    """
    coords_list = []
    scale = 1.0 # change as needed
    
    for i in range(N):
        nth_coord = []
        for j in range(3):
            # The coord on the x axis should be shifted 
            if (False): # change to 'false' if needed for general testing
                nth_coord.append(np.float64(scale * (float(i/np.sqrt(3)) +
                                                     random.random()/np.sqrt(3))))
            else:
                nth_coord.append(np.float64(scale * random.random()/np.sqrt(3)))
        coords_list.append(nth_coord)

    return np.array(coords_list)

@jit(nopython=True)
def coord_init_2(N: int) -> np.array:
    """
    An alternate code (hopefully faster) for initializing coordinates

    N: number of monomers in the polymer
    """
    coords = np.zeros(shape=(N,3))
    scale = 1.0

    for i in range(N):
        # put the coords in the middle of the box
        center = np.array([4.5,4.5,4.5])
        nth_coord = center + (random.random(3) / (np.sqrt(3)))
        coords[i] = nth_coord


    return coords



@jit(nopython=True)
def coord_init_3(N: int) -> np.array:
    """
    linear initializer
    """

    coords = np.zeros(shape=(N,3))
    scale = 1.0

    for i in range(N):
        # place on string in the x direction (will be rather close)
        xr = i * 0.95
        yr = np.random.random() / 5
        zr = np.random.random() / 5
        coords[i] = np.array([xr, yr, zr])


    return coords

@jit(nopython=True)
def coord_init_4(N: int) -> np.array:
    """
    linear initializer
    """

    coords = np.zeros(shape=(N,3))
    scale = 1.0

    for i in range(N):
        # place on string in the x direction (will be rather close)
        xr = (-N + 2*i) * 0.95 / np.sqrt(3)
        yr = (-N + 2*i) * 0.95 / np.sqrt(3)
        zr = (-N + 2*i) * 0.95 / np.sqrt(3)
        # yr = np.random.random() / 5
        # zr = np.random.random() / 5
        coords[i] = np.array([xr, yr, zr])


    return coords




@jit(nopython=True)
def bond_length(A: np.array, B: np.array) -> np.float64:
    """
    Takes two coordinate positions are returns the bond length (not squared)
    """
    return np.linalg.norm(B-A)
    # dist = np.float64(0.0)

    # for i in range(3):
    #     dist += (A[i] - B[i])**2

    # return np.sqrt(dist)



@jit(nopython=True)
def harmonic_energy(A: np.array, B: np.array) -> np.float64:
    """
    Takes 2 coordinate vectors and finds the potential energy for that bond
    Give single row 3-vectors such that A[0] = x coordinate of A

    Does not include LJ energy
    """
    dist = np.linalg.norm(A-B) ** 2

    # for i in range(3):
    #     dist += (A[i] - B[i])**2

    # return 3 * 1/2 r^2 
    # since k was found to be 3
    return np.float64(1.5 * dist)


@jit(nopython=True)
def harmonic_gradient(r:np.float64) -> np.float64:
    """
    The magnitude of the gradient for the Harmonic potential
    (positive sign returned) for a given separation $r$. 
    """
    # yes, this is a silly function but in terms of bookkeeping for me
    # its just easy to have this operation labeled as a function
    return 3*r 


@jit(nopython=True)
def ljts_gradient(r: np.float64) -> np.float64:
    """
    return the magnitude of the gradient of the LJTS potential
    """
    #MANUAL ESCAPE TO PREVENT FORCE DIVERGING
    # if (r < 0.5):
    #     print("too close!")
    #     r = 0.5
    delta = 0 # was added in for debugging, keep at zero
    shift = 0.039 # the shift to make the force go to zero at rc
    return 4 * ((12*(r+delta)**(-13) - 6*(r+delta)**(-7))) + shift


@jit(nopython=True)
def lj_potential(A: np.array, B: np.array) -> np.float64:
    """
    Takes in the coordinates of two beads and returns a Lennard Jones
    Potential
    For the purposes of this excercise this function should be able to give LJ
    potentials between both real and virtual coordinates


    NON SHIFTED VERSION
    """
    r = bond_length(A,B)
    cut_off = 2**(1/6)
    if (r > cut_off):
        #employ a cutoff radius
        return np.float64(0.0)
    else:
        # Lennard Jones
        return 4 * (r**(-12) - r**(-6))


@jit(nopython=True)
def ljts_potential(A: np.array, B: np.array) -> np.float64:
    """
    Takes a two coordinates and returns the truncated and shifted
    Lennard-Jones potential

    SHIFTED VERSION
    """

    # the potential
    r = bond_length(A,B)
    # MANUAL ESCAPE TO PREVENT ENERGY DIVERGING
    if (r < 0.01):
        print("r too small")
        print(r)
        # r += 0.01
    lj: np.float64
    # the shift to make the potential continous with zero at rc
    shift =  0.0163

    lj = 4 * ((1/r)**12 - (1/r)**6) + shift 
    # print(lj)
    return lj

@jit(nopython=True)
def get_dvec(A: np.array,B:np.array) -> np.float64:
    """
    get vector to B from A's persective
    """
    return B - A
    # v = np.zeros(3)
    # for i in range(3):
    #     v[i] = B[i] - A[i]

    # return v


@jit(nopython=True)
def get_norm_dvec(A: np.array,B:np.array) -> np.float64:
    """
    get unit vector to B from A's persective
    """
    # vec = get_dvec(A,B)

    # mag = mag_of_vec(vec)
    # if (mag < 0.01):
    #     print("mag too small, err")
    #     print(mag)
    #     mag += 0.01

    # return vec / mag
    if ((A == B).all()):
        print("same vec!!")
    l = np.linalg.norm(B-A)
    if l < 0.001:
        print("norm vec err")
        print(l)
        l += 0.001
    val = (B-A) / l 
    # print("got val!")
    return val



@jit(nopython=True)
def mag_of_vec(A: np.array) -> np.float64:
    """
    Takes a 3-vector and returns the magnitude of the vector
    """
    # faster to use numpy? 
    return np.linalg.norm(A)
    # mag = np.float64(0.0)
    # i = 0
    # while i < 3:
    #     mag += (A[i])**2
    #     i += 1
    
    # return np.sqrt(mag)



@jit(nopython=True)
def image_tracker(C: np.array, L: np.float64) -> np.array:
    """
    Another implementation of the image tracker that takes in a coordinate
    array of Nx3 in real space. Returns the virtual coordinates 
    inside the cube of length L as "virtual_C"
    """
    N = len(C)

    trackers = np.zeros(shape=(N,3))
    
    for i in range(N):
        for j in range(3):
            if (C[i][j] >= 0.0):
                trackers[i][j] = int(C[i][j]/L)
            elif (C[i][j] <= -L):
                trackers[i][j] = int(C[i][j]/L) - 1
            else:
                trackers[i][j] = -1

    virtual_C = np.zeros(shape=(N,3))

    for i in range(N):
        for j in range(3):
            virtual_C[i][j] = C[i][j] - L * trackers[i][j]

    return virtual_C

@jit(nopython=True)
def nearest_neighbor_vec(A: np.array, B:np.array, L:np.float64) -> np.float64:
    """
    For two cells that are neighbors in the cell list, find the 
    nearest neighbor image of the two (this should help in preventing
    the total energy from diverging)

    Takes in the coordinates of two particles, returns a length 
    to the nearest neighbor image using the formula given in the problem
    statement.  Also takes the length of the box L
    """
    # L = V**(1/3)
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    dz = B[2] - A[2]


    dx = dx - L * round(dx/L)
    dy = dy - L * round(dy/L)
    dz = dz - L * round(dz/L)


    return np.array([dx,dy,dz])


@jit(nopython=True)
def delta_maker(a: np.float64) -> np.array:
    """
    The Standard Monte Carlo deltas maker (trial displacement in r)
    """
    return -a + 2*a*random.random(3)


@jit(nopython=True)
def combo_energy(r: np.float64) -> np.float64:
    """
    return the combination of Harmonic and LJTS potential for a
    given separation r - this is mainly used in debugging
    """
    A = np.array([r,0.0,0.0])
    B = np.zeros(3)
    return harmonic_energy(A,B) + ljts_potential(A,B)

@jit(nopython=True)
def ljts_total_energy(C: np.array,L:np.float64) -> np.float64:
    """
    Returns the total energy of the polymer ie harmonic potential between 
    bonds as well as the 12-6 LJTS potential 
    """
    N = len(C)
    # virtual_C = image_tracker(C,L)

    tot_en = np.float64(0.0)

    for i in range(N-1):
        tot_en += harmonic_energy(C[i],C[i+1])
        # tot_en += ljts_potential(virtual_C[i],virtual_C[i+1])
        tot_en += ljts_potential(C[i],C[i+1])

    return tot_en


@jit(nopython=True)
def standard_single_mover(C: np.array, a:np.float,L:np.float64):
    """
    standard MC one particle LJTS mover

    Takes real space geom. C, attempts a single move based on Harmonic + LJTS
    energy

    Returns next geometry (not necessarily different) as well as next energy
    (not necessarily different)
    """
    N = len(C)
    
    # make selection index 
    # index = int(round(N * random.random()))
    index = int(N * random.random())
    frozen_coords = C.copy()

    # standard deltas
    deltas = delta_maker(a)

    old_energy = ljts_total_energy(frozen_coords,L)

    # make trial move
    C[index] = C[index] + deltas

    new_energy = ljts_total_energy(C,L)
    en_change = new_energy - old_energy
    accept_prob = min(1.0,np.exp(-en_change))
    rand_val = np.random.random()

    if (rand_val <= accept_prob):
        # accept
        return C, new_energy
    else:
        return frozen_coords, old_energy


@jit(nopython=True)
def avg_rsq(C:np.array) -> np.float64:
    """
    Takes a geometry C and reports the average r^2 bond distance over the 
    polymer.
    """
    N = len(C) # number of monomers in polymer

    rsq_sum = np.float64(0.0)

    for i in range(N-1):
        bond = bond_length(C[i],C[i+1])
        rsq_sum += bond**2

    rsq_sum = rsq_sum / np.float64(N-1)
    return rsq_sum


@jit(nopython=True)
def standard_loop_mover(C:np.array, a:np.float64, L:np.float64):
    """
    wrapper for calling 'standard_single_mover' based on the length of the 
    polymer

    C: absolute coordinate array
    a: scaling factor for the displacement
    """
    N = len(C)

    energies = np.zeros(N)
    rsqs = np.zeros(N)

    # make N trial moves for a polymer of length N
    for i in range(N):
        C, energy = standard_single_mover(C, a,L)
        loq_rsq = avg_rsq(C)
        energies[i] = energy
        rsqs[i] = loq_rsq


    # note that there are multiple objects returned here unlike the 
    # first version of this function in pset 1
    return C, energies, rsqs


@jit(nopython=True)
def force_maker(C: np.array) -> np.array:
    """
    given a geometry of a chain C, give a matrix F of the net force vector 
    acting on each element C_i

    Will account for both LJTS as well as harmonic gradients

    End pieces will only have 1 interaction ie. source of fources, while 
    all others will have 2
    """
    N = len(C)

    # initialize F matrix
    F = np.zeros(shape=(N,3))

    # do first element
    r_01 = bond_length(C[0],C[1])
    unit_v_01 = get_norm_dvec(C[0],C[1])
    f_lj_mag = ljts_gradient(r_01)
    f_0_ljts = -f_lj_mag * unit_v_01

    f_harmonic_mag = harmonic_gradient(r_01)
    f_0_h = -f_harmonic_mag * unit_v_01

    F[0] = f_0_ljts + f_0_h

    for i in range(1,N-1):
        # units A and B
        r_ab = bond_length(C[i-1],C[i])
        unit_ab = get_norm_dvec(C[i],C[i-1])
        f_lj_ab_mag = ljts_gradient(r_ab)
        f_ab_ljts = -f_lj_ab_mag * unit_ab

        f_harmonic_ab_mag = harmonic_gradient(r_ab)
        f_ab_h = -f_harmonic_ab_mag * unit_ab

        # units B and C
        r_bc = bond_length(C[i],C[i+1])
        unit_bc = get_norm_dvec(C[i],C[i+1])
        f_lj_bc_mag = ljts_gradient(r_bc)
        f_bc_ljts = -f_lj_bc_mag * unit_bc

        f_harmonic_bc_mag = harmonic_gradient(r_bc)
        f_bc_h = -f_harmonic_bc_mag * unit_bc

        F[i] = f_ab_ljts + f_ab_h + f_bc_ljts + f_bc_h

    # do the last element
    r_last = bond_length(C[N-2],C[N-1])
    unit_last = get_norm_dvec(C[N-1],C[N-2])
    f_lj_mag = ljts_gradient(r_last)
    f_l_ljts = -f_lj_mag * unit_last

    f_harmonic_mag = harmonic_gradient(r_last)
    f_l_h = -f_harmonic_mag * unit_last

    F[N-1] = f_l_ljts + f_l_h

    # check the sign on this
    # return -F
    return F

@jit(nopython=True)
def specific_force_vec(C:np.array, index:int) ->np.array:
    """
    Find the forces acting on one particle only - don't find the entire 
    matrix each time if it's not necessary
    """
    N = len(C)

    # simple 3 vec
    # F = np.zeros(3)

    if (index == 0):
        r_01 = bond_length(C[0],C[1])
        # print("A")
        unit_v_01 = get_norm_dvec(C[0],C[1])
        f_lj_mag = ljts_gradient(r_01)
        # print("B")
        f_0_ljts = -f_lj_mag * unit_v_01

        f_harmonic_mag = harmonic_gradient(r_01)
        f_0_h = -f_harmonic_mag * unit_v_01

        # return -(F_0_LJTS + F_0_H)
        return (f_0_ljts + f_0_h)
    elif (index == N - 1):
        r_last = bond_length(C[N-2],C[N-1])
        # print("A")
        unit_last = get_norm_dvec(C[N-1],C[N-2])
        f_lj_mag = ljts_gradient(r_last)
        # print("B")
        f_l_ljts = -f_lj_mag * unit_last

        f_harmonic_mag = harmonic_gradient(r_last)
        f_l_h = -f_harmonic_mag * unit_last

        # return -(F_L_LJTS + F_L_H)
        return (f_l_ljts + f_l_h)
    else:
        # units A and B
        r_ab = bond_length(C[index],C[index-1])
        # print("A") # not the A of A and B
        unit_ab = get_norm_dvec(C[index],C[index-1])
        f_lj_ab_mag = ljts_gradient(r_ab)
        # print("B")
        f_ab_ljts = -f_lj_ab_mag * unit_ab

        f_harmonic_ab_mag = harmonic_gradient(r_ab)
        f_ab_h = -f_harmonic_ab_mag * unit_ab

        # units B and C
        r_bc = bond_length(C[index],C[index+1])
        unit_bc = get_norm_dvec(C[index],C[index+1])
        f_lj_bc_mag = ljts_gradient(r_bc)
        f_bc_ljts = -f_lj_bc_mag * unit_bc

        f_harmonic_bc_mag = harmonic_gradient(r_bc)
        f_bc_h = -f_harmonic_bc_mag * unit_bc

        # return -(F_AB_LJTS + F_AB_H + F_BC_LJTS + F_BC_H)
        return (f_ab_ljts + f_ab_h + f_bc_ljts + f_bc_h)



@jit(nopython=True)
def R_sampler(A: np.float64) -> np.array:
    """
    Gaussian random force vector for smart monte carlo trial displacements.
    To be used in coordination with the force vector. 

    A: the scaling parameter from Rossky et al. use trial value of dt^2 / 2

    Returns vector R from SMC
    """
    W = np.random.normal(loc=0.0, scale=np.sqrt(2*A), size = 3)

    return np.sqrt(1 / (4 * np.pi * A)) * W

@jit(nopython=True)
def smart_mc_move_acceptor(C_old: np.array,C_new:np.array,F_old:np.array,index:int,A:np.float64) -> bool:
    """
    The smart monte carlo move acceptor : returns a boolean for whether a 
    move should be accepted.

    IN:
        C_old: pre-move geometry
        C_new: post-move geometry
        F_old: the force on i before the particle is moved
        index: index of moving particle within the array
        A: distribution parameter
    OUT:
        boolean: True if move is accepted
        updated energy
    """
    N = len(C_old)

    # default value of L
    L = 10.0

    # find energies
    en_old = ljts_total_energy(C_old,L)
    en_new = ljts_total_energy(C_new,L)
    en_diff = en_new - en_old

    # change in r_i
    d_ri = C_new[index] - C_old[index]

    # method with no optimization for forces
    # (given F_old from the function that calls this function)
    # F_old = specific_force_vec(C_old,index)
    F_new = specific_force_vec(C_new,index)
    F_diff = F_new - F_old

    # following the formula from Allen and Tildesley (perform at T = 1.0)
    dwsmc = (A/4) * (np.dot(F_diff,F_diff) + np.dot(2*F_diff,F_old))
    exp_psi_nm = -(en_diff + np.dot(0.5*(F_new + F_old),d_ri) + dwsmc)

    prob = min(1.0,np.exp(exp_psi_nm))
    rand = np.random.random()

    if (rand <= prob):
        return True, en_new
    else:
        return False, en_old


@jit(nopython=True)
def smart_single_mover(C:np.array, A:np.float64,L:np.float64=10.0):
    """
    smart MC one particle mover

    Takes real space geometry C, attemps a single displacement. returns 
    updated (but not necessarily different) geometries and energies
    """
    N = len(C)
    
    # index = int(round(N * np.random.random()))
    index = int(N * np.random.random())
    frozen_coords = C.copy()

    # find force in particle i
    F_i = specific_force_vec(frozen_coords,index)
    # if np.linalg.norm(F_i) > 100: 
    #     print("large force")

    # get random gaussian variable
    R = R_sampler(A)

    # delta r: m to n
    beta = 1.0 # adjust if sims are run at T != 1.0
    drmn =  beta * A * F_i + R

    # move the particle
    C[index] = C[index] + drmn

    # find if accepted
    acc, updated_en = smart_mc_move_acceptor(frozen_coords,C,F_i,index,A)

    if acc:
        return C, updated_en
    else:
        # still return "updated_en", since this is the old en if 
        # no changes are made
        return frozen_coords, updated_en


@jit(nopython=True)
def smart_loop_mover(C:np.array, A:np.float64, L:np.float64 = 10.0):
    """
    wrapper for calling the smart single mover based on the length of the 
    polymer

    C: absolute coordinate array
    A: gaussian parameter for "R" in SMC trial moves
    L: length of box (if used)
    """
    N = len(C)

    # these will be returned and subsequently added to the longer list 
    # of total energies and rsqs for the length of the iterations performed
    energies = np.zeros(N)
    rsqs = np.zeros(N)
    
    # debugging, want to track the minimum bond lengths
    # mins = np.zeros(N)

    # make N trial moves on the polymer
    for i in range(N):
        # print("trial move: " + str(i))
        C, energy = smart_single_mover(C, A, L)
        # mini = 5.0
        # for j in range(N-1):
        #     if (np.linalg.norm(C[j]-C[j+1])) < mini: 
        #         mini = np.linalg.norm(C[j]-C[j+1])
        energies[i] = energy
        rsqs[i] = avg_rsq(C)
        # mins[i] = mini



    return C, energies, rsqs#, mins




@jit(nopython=True)
def standard_energy_rsq_recorder(N: int, L:np.float64,ita: int, a:np.float64):
    """
    an energy and average rsq recorder for the standard MC method

    N: size of polymer
    ita: number of total-chain update cycles to run
    a: scaling factor for displacements
    """

    # C = coord_init_2(N)
    C = coord_init_3(N)

    start_en = ljts_total_energy(C,L)

    if (ita > 20):
        progress = int(ita/20)
    else:
        progress = np.sqrt(2)

    length = N * ita
    # print(f"len {length}")
    long_energies = [] #np.zeros(shape=length)
    long_rsqs = [] #np.zeros(shape=length)
    # long_mins = []

    for i in range(ita):
        # records total energy and average rsq
        C, curr_en, curr_rsqs = standard_loop_mover(C, a,L)
        # print(len(curr_en))
        # print(len(curr_rsqs))
        # print(len(long_energies[N*ita:N*(ita+1)]))
        # long_energies[N*ita:N*(ita+1)] = curr_en
        # long_rsqs[N*ita:N*(ita+1)] = curr_rsqs
        long_energies.extend(curr_en)
        long_rsqs.extend(curr_rsqs)
        # long_mins.extend(curr_mins)


    return C, np.array(long_energies), np.array(long_rsqs)


@jit(nopython=True)
def smart_energy_rsq_recorder(N:int,L:np.float64,ita:int,A:np.float64):
    """
    an energy and average rsq recorder for smart monte carlo

    N: size of polymer
    L: length of box (if used)
    ita: number of total-chain updates to run
    A: gaussian scaling factor
    """

    C = coord_init_3(N)

    start_en = ljts_total_energy(C,L)

    if (ita > 20):
        progress = int(ita/20)
    else:
        progress = np.sqrt(2)

    # length = N * ita
    long_energies = []
    long_rsqs = []
    # long_mins = []

    # perform the iterations, at each ita call the loop mover and get a 
    # list of energies and average rsqs's
    for i in range(ita):
        C, curr_en, curr_rsqs = smart_loop_mover(C,A,L)
        long_energies.extend(curr_en)
        long_rsqs.extend(curr_rsqs)
        # long_mins.extend(curr_mins)
        # print(i)

    return C, np.array(long_energies), np.array(long_rsqs)#, np.array(long_mins)



@jit(nopython=True)
def MSD(ref_C: np.array, C:np.array) -> np.float64:
    """
    MSD for a given C as compared to initial geometry ref_C

    IN:
        ref_C: geom. at t = 0
        C: current geometry
    OUT:
        MSD: float64 of the MSD
    """
    N = len(C)

    msd = np.float64(0.0)

    # for curr, ref in np.nditer([C,ref_C]):
    #     msd += np.linalg.norm(curr - ref)**2

    # diff = C - ref_C

    # return np.sum(np.linalg.norm(diff,axis=1)**2)

    for i in range(N):
        msd += np.linalg.norm(C[i]-ref_C[i])**2

    return msd


@jit(nopython=True)
def standard_msd_loop_mover(C: np.array, ref_C:np.array,a:np.float64,L:np.float64):
    """
    wrapper for calling standard_single_mover which also records MSDS
    """
    N = len(C)

    energies = np.zeros(N)
    msds = np.zeros(N)

    for i in range(N):
        C, energy = standard_single_mover(C, a, L)
        energies[i] = energy

        msd = MSD(ref_C,C) 
        msds[i] = msd


    return C, energies, msds


@jit(nopython=True)
def standard_msd_recorder(N:int,L:np.float64,ita:int,a:np.float64):
    """
    MSD recorder for standard monte carlo
    
    N: size of polymer (int)
    ita: number of total chain interations to run
    a: scaling factor for displacements
    """

    C = coord_init_3(N)
    ref_C = C.copy()
    if (ita > 20):
        progress = int(ita/20)
    else:
        progress = np.sqrt(2)
    
    long_energies = []
    long_msds = []

    for i in range(ita):
        C, curr_en, curr_msds = standard_msd_loop_mover(C,ref_C,a,L)
        long_energies.extend(curr_en)
        long_msds.extend(curr_msds)


    return C, np.array(long_energies),np.array(long_msds)

@jit(nopython=True)
def smart_msd_loop_mover(C:np.array,ref_C:np.array,A:np.float64,L:np.float64):
    """
    wrapper for calling smart_single_mover which also records MSDS
    """
    N = len(C)

    energies = np.zeros(N)
    msds = np.zeros(N)

    for i in range(N):
        C, energy = smart_single_mover(C,A,L)
        energies[i] = energy

        msd = MSD(ref_C,C)
        msds[i] = msd


    return C, energies, msds

@jit(nopython=True)
def smart_msd_recorder(N:int,L:np.float64,ita:int,A:np.float64):
    """
    MSD recorder for smart monte carlo

    N: size of polymer
    ita: number of total chain iterations to run
    A: gaussian variable parameter for displacements
    """

    C = coord_init_3(N)
    ref_C  = C.copy()

    long_energies = []
    long_msds = []

    for i in range(ita):
        C, curr_en, curr_msds = smart_msd_loop_mover(C,ref_C,A,L)
        long_energies.extend(curr_en)
        long_msds.extend(curr_msds)

    return C, np.array(long_energies), np.array(long_msds)


### NOW NEED TO ANALYZE ROUSE MODES

@jit(nopython=True)
def rouse_mode(C:np.array) -> np.float64:
    """
    Return the value of the pth Rouse mode (p >= 1) for a geometry 
    at the current time t
    """
    N = len(C)

    # array of 5 vectors
    xp = np.zeros(shape=(5,3))

    # testing the first four Rouse modes
    for p in range(1,6):
        temp = np.zeros(3)
        for i in range(N):
            temp += C[i] * np.cos((p*(np.pi/N) * (i-0.5)))
        xp[p-1] = temp
    
    # recale by prefactor
    xp *= np.sqrt(2) * (N**(-0.5))

    return xp

@jit(nopython=True)
def standard_rouse_single_mover(C:np.array, a: np.float, L:np.float64):
    """
    single mover for standard MC which keeps track of the first four 
    Rouse modes
    """
    N = len(C)

    # make selection index
    index = int(N * random.random())
    frozen_coords = C.copy()

    # standard deltas
    deltas = delta_maker(a)

    old_energy = ljts_total_energy(C,L)

    C[index] = C[index] + deltas

    new_energy = ljts_total_energy(C,L)


    en_change = new_energy - old_energy
    accept_prob = min(1.0,np.exp(-en_change))
    rand_val = np.random.random()

    if (rand_val <= accept_prob):
        # remember that rouse mode returns a vector quantity that is later
        # dotted into the rouse mode at time t = 0
        return C, rouse_mode(C)
    else:
        return frozen_coords, rouse_mode(frozen_coords)



@jit(nopython=True)
def standard_rouse_loop(C:np.array, orig_rouse:np.array,a:np.float64,L:np.float64):
    """
    wrapper to call standard rouse mover based on the length of the polymer

    C: absolute coordinate array
    a: scaling factor for the displacement
    """
    N = len(C)

    # array of each of the modes 1 -5 (results placed here to be 
    # dotted with the original rouse mode at each of N iterations
    dotted_rouses = np.zeros(shape=(N,5))

    for i in range(N):
        # this rouse is a 4,3
        C, rouse = standard_rouse_single_mover(C,a,L)

        local_dotted = np.zeros(5)
        for j in range(5):
            local_dotted[j] = np.dot(rouse[j],orig_rouse[j])



        dotted_rouses[i] = local_dotted


    return C, dotted_rouses

@jit(nopython=True)
def standard_rouse_recorder(N:int, L:np.float64,ita:int,a:np.float64):
    """
    The recorder of the autocorrelation of the Rouse modes for standard 
    Monte Carlo

    This will depend on a predetermined value of <r^2> for the standard MC
    chains
    """

    C = coord_init_4(N)

    # time t = 0 rouse modes
    orig_rouse = rouse_mode(C)

    # progress tracker, usually turned off in favor of notebook-level counters
    if (ita > 20):
        progress = int(ita/100)
    else:
        progress = np.sqrt(2)
    
    long_rouses = np.zeros(shape=(5,ita*N))

    rouse_1 =  [] #np.zeros(ita*N)
    rouse_2 =  [] #np.zeros(ita*N)
    rouse_3 =  [] #np.zeros(ita*N)
    rouse_4 =  [] #np.zeros(ita*N)
    rouse_5 =  [] #np.zeros(ita*N)

    # need to check that the standard number of iterations is being
    # done
    for i in range(ita):

        # if (ita % progress == 0):
        #     print(str(float(i) / ita))

        C, rouse_curr = standard_rouse_loop(C,orig_rouse,a,L)
        # print(len(rouse_curr))
        # check if the typing is right on this
        # long_rouses.extend(rouse_curr)
        # if (len(rouse_curr != N)):
        #     print("length error")
        # rouse_1[i*N:(i+i)*N] = rouse_curr[0]
        # rouse_2[i*N:(i+i)*N] = rouse_curr[1]
        # rouse_3[i*N:(i+i)*N] = rouse_curr[2]
        # rouse_4[i*N:(i+i)*N] = rouse_curr[3]
        rouse_1.extend(rouse_curr[:,0])
        rouse_2.extend(rouse_curr[:,1])
        rouse_3.extend(rouse_curr[:,2])
        rouse_4.extend(rouse_curr[:,3])
        rouse_5.extend(rouse_curr[:,4])

    """ vestigial code left in should I need to change anything back"""

    # rouse_1 = np.zeros(ita*N)
    # rouse_2 = np.zeros(ita*N)
    # rouse_3 = np.zeros(ita*N)
    # rouse_4 = np.zeros(ita*N)

    # for i in range(ita):
    #     rouse_1[i] = long_rouses[i][0]
    #     rouse_2[i] = long_rouses[i][1]
    #     rouse_3[i] = long_rouses[i][2]
    #     rouse_4[i] = long_rouses[i][3]

    # rouse_1 = long_rouses[0]
    # rouse_2 = long_rouses[1]
    # rouse_3 = long_rouses[2]
    # rouse_4 = long_rouses[3]


    # found r^2 from previous calculations
    r2 = 1.62
    # Scaling behaviour is included in these lines as prescribed by Kopf,
    # Dünweg, and Paul (see report references)
    rouse_1 = np.array(rouse_1) * (4*np.sin(1*np.pi/(2*N)**2)) / r2
    rouse_2 = np.array(rouse_2) * (4*np.sin(2*np.pi/(2*N)**2)) / r2
    rouse_3 = np.array(rouse_3) * (4*np.sin(3*np.pi/(2*N)**2)) / r2
    rouse_4 = np.array(rouse_4) * (4*np.sin(4*np.pi/(2*N)**2)) / r2
    rouse_5 = np.array(rouse_5) * (4*np.sin(5*np.pi/(2*N)**2)) / r2

    return C, [rouse_1,rouse_2,rouse_3,rouse_4,rouse_5]

# NOW SMART MONTE CARLO ROUSE MODES
# (still check the above for errors)

@jit(nopython=True)
def smart_rouse_single_mover(C:np.array,A:np.float64,L:np.float64):
    """
    single mover for rouse modes with smart monte carlo
    """
    N = len(C)

    index = int(N * random.random())
    frozen_coords = C.copy()

    # find force on particle i
    F_i = specific_force_vec(frozen_coords,index)


    # get random gaussian variable
    R = R_sampler(A)

    # delta r: m to n
    beta = 1.0 # adjust if sims are run at T != 1.0
    drmn =  beta * A * F_i + R

    # move the particle
    C[index] = C[index] + drmn

    # find if accepted
    acc, updated_en = smart_mc_move_acceptor(frozen_coords,C,F_i,index,A)

    if acc:
        return C, rouse_mode(C)
    else:
        # still return "updated_en", since this is the old en if 
        # no changes are made
        return frozen_coords, rouse_mode(frozen_coords)


@jit(nopython=True)
def smart_rouse_loop(C:np.array,orig_rouse:np.array,A:np.float64,L:np.float64):
    """
    wrapper to call the smart MC single mover with rouse mode recordingrecordgin
    """
    N = len(C)

    dotted_rouses = np.zeros(shape=(N,5)) 

    for i in range(N):
        C, rouse = smart_rouse_single_mover(C,A,L)

        local_dotted = np.zeros(5)
        for j in range(5):
            local_dotted[j] = np.dot(rouse[j],orig_rouse[j])

        dotted_rouses[i] = local_dotted

    return C, dotted_rouses

@jit(nopython=True)
def smart_rouse_recorder(N:int, L:np.float64,ita:int,A:np.float64):
    """
    Recorder of the autocorrelation of the Rouse modes for smart monte carlo
    """
    C = coord_init_4(N)

    orig_rouse = rouse_mode(C) 

    long_rouses = np.zeros(shape=(5,ita*N))


    rouse_1 =  [] #np.zeros(ita*N)
    rouse_2 =  [] #np.zeros(ita*N)
    rouse_3 =  [] #np.zeros(ita*N)
    rouse_4 =  [] #np.zeros(ita*N)
    rouse_5 =  [] #np.zeros(ita*N)

    for i in range(ita):

        C, rouse_curr = smart_rouse_loop(C,orig_rouse,A,L)

        rouse_1.extend(rouse_curr[:,0])
        rouse_2.extend(rouse_curr[:,1])
        rouse_3.extend(rouse_curr[:,2])
        rouse_4.extend(rouse_curr[:,3])
        rouse_5.extend(rouse_curr[:,4])

    r2 = 1.498
    rouse_1 = np.array(rouse_1) * (4*np.sin(1*np.pi/(2*N)**2)) / r2
    rouse_2 = np.array(rouse_2) * (4*np.sin(2*np.pi/(2*N)**2)) / r2
    rouse_3 = np.array(rouse_3) * (4*np.sin(3*np.pi/(2*N)**2)) / r2
    rouse_4 = np.array(rouse_4) * (4*np.sin(4*np.pi/(2*N)**2)) / r2
    rouse_5 = np.array(rouse_5) * (4*np.sin(5*np.pi/(2*N)**2)) / r2

    return C, [rouse_1,rouse_2,rouse_3,rouse_4,rouse_5]



def main():
    """
    in-file main method for debugging purposes (not used)
    """




    print("Done!")

if __name__ == "__main__": main()
