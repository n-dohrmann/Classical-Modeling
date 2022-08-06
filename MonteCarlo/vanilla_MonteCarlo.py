"""
This is the main file for this project that will call the necessary functions
and write the necessary output as asked in parts 2 - 5 of the problem set.
Please see other files that are referenced from here 

Plots will be made with a separate file that calls on the written data 

This code is written with the assumption that there exists a directory ../Output/*
with the ouput files
"""

# import libraries 
from time import time
import datetime as dt
import numba
from numba import jit
import pandas as pd
#import cython # does this work here?
import os
import numpy as np
from numpy import random

# initialize and record seed selection
date = dt.datetime.now().strftime("%a %d %b %Y %X")
seed = int(time())
np.random.seed(seed)

# writes the runtime date and seed
with open("../Output/seed_log.txt", "a") as sl:
    sl.write(f"{date}, {seed}\n")


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
def bond_length(A: np.array, B: np.array) -> np.float64:
    """
    Takes two coordinate positions are returns the bond length (not squared)
    """
    dist = np.float64(0.0)

    for i in range(3):
        dist += (A[i] - B[i])**2

    return np.sqrt(dist)

@jit(nopython=True)
def bond_energy(A: np.array, B: np.array) -> np.float64:
    """
    Takes 2 coordinate vectors and finds the potential energy for that bond
    Give single row 3-vectors such that A[0] = x coordinate of A
    """
    dist = np.float64(0.0)

    for i in range(3):
        dist += (A[i] - B[i])**2

    # return 3 * 1/2 r^2 
    # since k was found to be 3
    return np.float64(1.5 * dist)


@jit(nopython=True)
def lj_potential(A: np.array, B: np.array) -> np.float64:
    """
    Takes in the coordinates of two beads and returns a Lennard Jones
    Potential
    For the purposes of this excercise this function should be able to give LJ
    potentials between both real and virtual coordinates
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
def image_tracker(C: np.array) -> np.array:
    """
    Keeps track of the real and image parts of the geometry using the suggested
    method of tracking how many times a coordinate has passed the boundary

    Takes the absoloute geometey (ie geometry in free space with no box), wrt the
    origin, and returns the virual coordinates within the box for use with LJ
    potential
    """
    N = len(C)
    trackers = np.zeros(shape=(N,3))

    for i in range(N):
        x_track = int(C[i][0]/10.0)
        y_track = int(C[i][1]/10.0)
        z_track = int(C[i][2]/10.0)
        trackers[i] = np.array([x_track,y_track,z_track])

    virtual_C = np.zeros(shape=(N,3))
    for i in range(N):
        for j in range(3):
            virtual_C[i][j] = C[i][j] - 10*trackers[i][j]

    return virtual_C

@jit(nopython=True)
def lj_total_energy(C: np.array) -> np.float64:
    """
    Takes in the absolute (non-image) coordinates of a geometry and returns 
    the LJ potential energy using the virtual image points found for the input 
    geometry
    """
    N = len(C)
    virtual_C = image_tracker(C)
    tot_en = np.float64(0.0)

    for i in range(N):
        j = i + 1
        while (j <= N-1):
            tot_en += lj_potential(virtual_C[i],virtual_C[j])
            j += 1

    # add in the harmonic
    tot_en += total_energy(C)
    return tot_en

@jit(nopython=True)
def lj_single_mover(C: np.array, a:np.float64) -> (np.array,np.float64):
    """
    An analagous function to 'corrected_mover'. Takes the absolute geometry
    array C and attempts to perform single move on the geometry based on energy 
    changes. Returns and updated array (not necessarily different) and the
    updated energy

    Also can give different results based on the value given for 'a'
    """
    N = len(C)
    index = int(N * random.random())

    frozen_coords = C.copy()
    deltas = delta_maker(a)

    old_energy = lj_total_energy(frozen_coords)

    for i in range(3):
        C[index][i] += deltas[i]

    new_energy = lj_total_energy(C)
    en_change = new_energy - old_energy
    accept_prob = min(1.00,np.exp(-en_change))
    rand_val = np.random.random()

    if (rand_val <= accept_prob):
        # accept
        return C, new_energy
    else:
        return frozen_coords, old_energy

#@jit(nopython=True)
def lj_loop_mover(C: np.array, a:np.float64, en_file: str, rsq_file:str) -> np.array:
    """
    A wrapper for calling 'lj_single_mover' based on the length of the string

    C: absolute coordinate array
    a: scaling factor
    en_file: file path for energy log, null if i/o not needed
    rsq_file: file path for rsq, null if i/o not needed
    """
    N = len(C)

    if (en_file != "null"):
        f = open(en_file, "a+")
    if (rsq_file != "null"):
        rf = open(rsq_file, "a+")

    for i in range(N):
        C, energy = lj_single_mover(C, a)
        loc_rsq = avg_rsq(C)
        if ((en_file != "null") and (rsq_file != "null")):
            f.write(str(energy) + "\n")
            rf.write(str(loc_rsq) + "\n")

    if ((en_file != "null") and (rsq_file != "null")):
        f.close()
        rf.close()

    return C



@jit(nopython=True)
def total_energy(C: np.array) -> np.float64:
    """
    returns the total energy of the sequence of bonds from a given geometry C
    """
    N = len(C)
    en = np.float64(0.0)

    for i in range(N - 1):
        en += bond_energy(C[i],C[i+1])

    return en


@jit(nopython=True)
def delta_maker(a: np.float64) -> np.array:
    """
    Generates a delta list to offset the coordinates by, 
    scaled by 'a'
    """
    deltas = []
    for i in range(3):
        deltas.append(-a + 2*a*np.random.random())
    return np.array(deltas) 


#@jit(nopython=True)
def move_acceptor(in_coords: np.array, pos: int, a: np.float64, file_path: str) -> np.array:
    """
    ~~~ DEPRECATED ~~~
    Takes a set of in-coordinates, a new proposed coordinate change, and its
    position. The proposed change is scaled by 'a'

    The set of in coords should be either length 3 (inner) or 2 (edge). pos will
    indicate if an edge is the start (0) or finish (1) if len == 2

    Returns a set of out-coordinates based on wheter the move happens
    """

    frozen_coords = in_coords.copy() # freeze the input coords

    N = len(in_coords)
    if (N == 2):
        if (pos == 0):
            neighbors = [1]
            index = pos
        elif (pos == 1):
            neighbors = [-1]
            index = pos
        else:
            raise ValueError("'move_acceptor': 'pos' must be 0 or 1!")
    elif (N == 3):
        neighbors = [-1,1]
        index = 1
    else:
        raise ValueError("'move_acceptor': invalid coordinate slice")


    old_en = np.float64(0.0)

    for neighbor in neighbors:
        old_en += bond_energy(in_coords[index], in_coords[index+neighbor])

    deltas  = delta_maker(a)

    for i in range(3):
        in_coords[index][i] = deltas[i]

    new_en = np.float64(0.0)

    for neighbor in neighbors:
        new_en += bond_energy(in_coords[index], in_coords[index+neighbor])

    en_delta = new_en - old_en
    acceptance_prob = min(1.0,np.exp(-(new_en - old_en)))
    rand = np.random.random()

    if (rand < acceptance_prob): 
        # accepted
        line = os.popen(f"tail -1 {file_path}").read()
        print(f"~~~\n\n This is line val: {str(line)}\n\n")
        reference_energy = np.float64(line)
        changed_energy = reference_energy + en_delta
        with open(file_path, "a+") as f:
            f.write(str(changed_energy) + "\n")
        return in_coords # modified coordinates
    else:
        # denied
        print("denied!")
        line = os.popen(f"tail -1 {file_path}").read()
        print(f"This is line: {line}")
        reference_energy = np.float64(line)
        # I know this looks silly, but this *did* fix a bug by adding zero
        # maybe casting issues, im not sure
        changed_energy = reference_energy + np.float64(0.0)
        with open(file_path, "a+") as f:
            f.write(str(changed_energy) + "\n")
        return frozen_coords
        



#@jit(nopython=True)
def total_geometry_update(C: np.array, a: np.float64, file_path: str) -> np.array:
    """
    ~~~ DEPRECATED ~~~
    Random displacement method for coordinate arrays
    IN:
        np.array "C": Coordinate array of polymer
        np.float64 "a": scaling factor for displacement
    OUT: 
        np.array: new array with updated coordinates
    """
    N = len(C)

    if ((N%2) == 0): 
        #  group 1 
        for i in range(0,N-1,2):
            if(i == 0):
                C[0:2] = move_acceptor(C[0:2], 0, a, file_path)
            else:
                C[i-1:i+2] = move_acceptor(C[i-1:i+2], 1, a, file_path)
        # group 2 
        for i in range(1,N,2):
            if(i == N-1):
                C[i-1:N] = move_acceptor(C[i-1:N], 1, a, file_path)
            else:
                C[i-1:i+2] = move_acceptor(C[i-1:i+2], 1, a, file_path)
    else:
        # group 1 
        for i in range(0, N, 2):
            if (i == 0):
                C[0:2] = move_acceptor(C[0:2], 0, a, file_path)
            elif (i == N-1):
                C[i-1:N] = move_acceptor(C[i-1:N], 1, a, file_path)
            else:
                C[i-1:i+2] = move_acceptor(C[i-1:i+2], 1, a, file_path)
        # group 2
        for i in range(1,N-1,2):
            C[i-1:i+2] = move_acceptor(C[i-1,i+2], 1, a, file_path)

    return C


@jit(nopython=True)
def corrected_mover(C: np.array, a: np.float64) -> (np.array, np.float64):
    """
    The code development had to be corrected due to an incorrect selection
    scheme that had to be modified. The above methods 'move_acceptor' and 
    'total_geometry_update' are left in for my reference, but are not used in
    any of the main methods

    IN: "C" a coordinate array of an N-bead polymer. A random index is selected
    and the corresponding bead is considered for a trial move which may or may
    not happen

        "a": scaling factor, float 
    """
    N = len(C)
    # index of bead to move
    index = int((N) * np.random.random())

    frozen_coords = C.copy()

    # delta[i] = -a + 2*a*np.random.random()
    # len(deltas) = 3
    deltas = delta_maker(a) 

    # initialized energy
    # tot_en sums 0.5* r^2 over each bond
    old_energy = total_energy(frozen_coords) 

    for i in range(3):
        C[index][i] += deltas[i]

    new_energy = total_energy(C)
    en_change = new_energy - old_energy
    accept_prob = min(1.00,np.exp(-en_change))
    rand_val = np.random.random()

    if (rand_val <= accept_prob):
        # accept the move
        return C, new_energy
    else:
        # deny move
        return frozen_coords, old_energy


#@jit(nopython=True)
def corrected_mover_loop(C: np.array, a: np.float64, file_path: str, rsq_log:str) -> np.array:
    """
    A wrapper for calling the  corrected mover funtion

    Will call the mover function by a number of times equal to the
    length of the polymer

    C: coordinate array 
    a: scaling factor
    file_path: path of energy log
    rsq_log: path of rsq log
    """
    N = len(C) # length of chain

    if (file_path != "null"):
        f = open(file_path, "a+")
    if (rsq_log != "null"):
        rf = open(rsq_log, "a+")

    for i in range(N):
        C, energy = corrected_mover(C, a)
        loc_rsq = avg_rsq(C)
        if ((file_path != "null") and (rsq_log != "null")):
            f.write(str(energy) + "\n")
            rf.write(str(loc_rsq) + "\n")


    if ((file_path != "null") and (rsq_log != "null")):
        f.close()
        rf.close()


    return C


@jit(nopython=True)
def avg_rsq(C:np.array) -> np.float64:
    """
    Takes a geometry C and reports the average r^2 bond distance over the 
    polymer.
    """
    N = len(C) # number of beads

    rsq_sum = np.float64(0.0)

    for i in range(N-1):
        bond = bond_length(C[i],C[i+1])
        rsq_sum += bond**2

    rsq_sum = rsq_sum / np.float64(N-1)
    return rsq_sum

@jit(nopython=True)
def MSD(C: np.array, C_0: np.array) -> np.float64:
    """
    Function for MSD (question 3) 

    IN:
        C: geometry at current iteration
        C_0: initial geometry
    OUT:
        msd_val: MSD for the two above geometries
    """
    msd_val = np.float64(0.0)

    N = len(C) 
    assert N == len(C_0)

    for i in range(N):  
        # can reuse "bond_legnth(A, B)" here since it just 
        # returns the absolute distance between two coordinates
        msd_val += (bond_length(C[i],C_0[i]))**2

    return msd_val / np.float64(N)

@jit(nopython=True)
def end_to_end_vec(C: np.array) -> np.array:
    """
    end to end distance finder for question 4
    IN:
        C: geometry of chain
    OUT:
        Vector from first to last bead
    """
    N = len(C)
    first = C[0]
    last = C[N-1]
    vec = np.zeros(3)

    for i in range(3):
        vec[i] = first[i] - last[i]

    return vec

@jit(nopython=True)
def end_to_end_rsq(C: np.array) -> np.array:
    """
    gives the rsq value between the first and last points 
    of the geometry
    IN:
        C: geometry
    OUT:
        rsq, scalar
    """

    vec = end_to_end_vec(C)
    rsq = np.float64(0.0)
    for i in range(len(vec)):
        rsq += vec[i]**2

    return rsq



def main_2():
    """
    The main method for question 2. Should make the necessary parts of the
    simulation as well as output the desired data.
    """

    """ note: initialization successful """ 

    # shell = os.popen("echo $0").read()
    # print(f"This is the used shell: {shell}")
    num_cycles = 500000
    # a = 1.0/np.sqrt(3)
    a = 1.0

    C = coord_initializer(2)
    print(C)
    start_en = total_energy(C)
    print(f"This config has starting energy: {start_en}")

    file_path = "../Output/q2_energy_plot.csv" 
    os.system(f"> {file_path}")

    rsq_log = "../Output/q2_rsq_log.csv"
    os.system(f"> {rsq_log}")


    with open (file_path, "w+") as f:
        f.write(f"Tot_E\n{start_en}\n")

    rf = open(rsq_log, "w+")
    rf.write(f"Rsq_Avg\n")
    rf.close()

    if (num_cycles > 20):
        progress = int(num_cycles/20)
    else:
        progress = np.sqrt(2) # make it irrational so mod is never 0



    for i in range(num_cycles):
        C = corrected_mover_loop(C, a, file_path, rsq_log)
        if ((i % progress) == 0):
            print(f"{100*i/num_cycles}%")


    stop_en = total_energy(C)

    print("100.0%")
    print(f"Ending energy: {stop_en}")
    print("Done!")


def main_3():
    """
    The main method for question 3
    """
    num_cycles = 100000
    a_list = [0.1,  0.25, 0.50, 0.75, 1, 2]
    a_names = ["01",  "025", "05", "075","1", "2"]
    C = coord_initializer(32)
    C_0 = C.copy()

    avg_list = [] # list of the average of each series
    avg_count = 100

    file_name = "../Output/avg_msd_data.csv"
    os.system(f"> {file_name}") 
    for i, a in enumerate(a_list):
        a_avg = []
        iteration_sum = np.zeros(shape=(avg_count,num_cycles))
        for j in range(avg_count):
            print(f"On step: {i+1} of {len(a_list)}, iteration: {j+1} of {avg_count}")

            # msd_file = f"../Output/msd_{a_names[i]}.csv"
            # os.system(f"> {msd_file}")
            # msf = open(msd_file, "w+") 
            # msf.write("MSD\n") 

            for n in range(num_cycles):
                # dont want to record E or total rsq here
                C = corrected_mover_loop(C, a, "null", "null")
                loc_msd = MSD(C, C_0)
                iteration_sum[j,n] += loc_msd
                # msf.write(str(loc_msd) + "\n")

            del C
            C = C_0.copy()
        for i in range(num_cycles):
            a_avg.append(np.sum(iteration_sum[:,i:i+1])/ np.float64(avg_count))
        avg_list.append(a_avg)

    # hopefully this does work and a loop isnt needed
    df = pd.DataFrame(np.array(avg_list).T,columns=[a_names])
    df.to_csv(file_name,index=False)

    print("Done!")


def main_4():
    """
    The main method for question 4
    """
    file_path_q4 = "../Output/q4_data.csv"
    os.system(f"> {file_path_q4}")


    N_list = [8,16,32,64]
    N_names = [] 
    # make a list of names for the dataframe
    for N in N_list:
        N_names.append(str(N))

    a = 0.50 # trial a to use (0.75?)
    total_rsq_of_N = [] # list of lists of rsq

    for i, N in enumerate(N_list):
        print(f"On N {i+1} of {len(N_list)}")
        # make coordinate array for this N 
        C = coord_initializer(N)
        # dont need C_0 for this run
        #C_0 = C.copy()
        iterations = 1000000
        loc_rsq_list = []

        for i in range(iterations):
            C = corrected_mover_loop(C,a,"null","null")
            loc_rsq = end_to_end_rsq(C)
            loc_rsq_list.append(loc_rsq)

        total_rsq_of_N.append(loc_rsq_list)

    df = pd.DataFrame(np.array(total_rsq_of_N).T,columns=[N_names])
    df.to_csv(file_path_q4,index=False)
    print("Done!")

    ## Data analyis done in jupyer notebook




def main_5():
    """
    The main method for question 5
    """
    ## implement the Lennard Jones potential and apply the minimum image
    ## convention
    stage = int(input("Which part of Q5? [2-4]: "))
    if (stage == 2):
        main_5_2()
    elif (stage == 3):
        main_5_3()
    elif (stage == 4):
        main_5_4()
    else:
        print("Please run again with a valid number")



def main_5_2():
    """
    main method for question 5 version of question 2 (with LJ potential)
    """
    num_cycles = 500000
    a = 1.0

    C = coord_initializer(2)
    print(C)
    start_en = lj_total_energy(C)
    print(f"This config has starting energy: {start_en}")

    en_file = "../Output/q5_2_energy.csv"
    os.system(f"> {en_file}")

    rsq_file = "../Output/q5_2_rsq.csv"
    os.system(f"> {rsq_file}")

    with open(en_file, "w+") as f:
        f.write(f"Tot_E\n{start_en}\n")

    with open(rsq_file, "w+") as rf:
        rf.write(f"Rsq_Avg\n")

    if(num_cycles > 20):
        progress = int(num_cycles/20)
    else:
        # make irrational to disable progress tracking
        progress = np.sqrrt(2)


    for i in range(num_cycles):
        C = lj_loop_mover(C, a, en_file, rsq_file)
        if ((i % progress) == 0):
            print(f"{100*i/num_cycles}%")

    stop_en = lj_total_energy(C)
    print("100%")
    print(f"Ending energy: {stop_en}")
    print("Done!")


def main_5_3():
    """
    the main method for question 5 part 3 
    """
    num_cycles = 10000
    a_list = [0.1,  0.25, 0.50, 0.75, 1, 2]
    a_names = ["01",  "025", "05", "075","1", "2"]
    C = coord_initializer(32)
    C_0 = C.copy()

    avg_list = []
    # adjusted to reflect cost
    avg_count = 50

    file_name = "../Output/lj_avg_msd_data.csv"
    os.system(f"> {file_name}")

    for i, a in enumerate(a_list):
        a_avg = []
        iteration_sum = np.zeros(shape=(avg_count,num_cycles))
        for j in range(avg_count):
            print(f"On step: {i+1} of {len(a_list)}, iteration {j+1} of {avg_count}")


            for n in range(num_cycles):
                C = lj_loop_mover(C,a,"null","null")
                loc_msd = MSD(C,C_0)
                iteration_sum[j,n] += loc_msd

            del C
            C = C_0.copy()

        for i in range(num_cycles):
            a_avg.append(np.sum(iteration_sum[:,i:i+1]) /np.float64(avg_count))
        avg_list.append(a_avg)

    df = pd.DataFrame(np.array(avg_list).T,columns=[a_names])
    df.to_csv(file_name,index=False)

    print("Done!")



def main_5_4():
    """
    main method for question 5.4
    """
    file_path_q5_4 = "../Output/q5_4_data.csv"
    os.system(f"> {file_path_q5_4}")

    N_list = [8,16,32,64]
    N_names = []

    for N in N_list:
        N_names.append(str(N))

    a = 0.5 # trial a to use
    total_rsq_of_N = []

    for i, N in enumerate(N_list):
        print(f"On N {i+1} of {len(N_list)}")
        C = coord_initializer(N)

        iterations = 100000
        loc_rsq_list = []

        for i in range(iterations):
            C = lj_loop_mover(C,a,"null","null")
            loc_rsq = end_to_end_rsq(C)
            loc_rsq_list.append(loc_rsq)

        total_rsq_of_N.append(loc_rsq_list)

    df = pd.DataFrame(np.array(total_rsq_of_N).T,columns=[N_names])
    df.to_csv(file_path_q5_4,index=False)
    print("Done!")



def test_main():
    """
    misc. tester to check methods
    """

    C = coord_initializer(2)
    print(f"C coords: \n {C}\n")

    bond = bond_length(C[0], C[1])
    print(f"bond length: {bond}") 

    bond_sq = bond**2
    print(f"rsq: {bond_sq}")

    single_en = bond_energy(C[0],C[1])
    print(f"single bond energy: {single_en}")

    energy = total_energy(C)
    print(f"total en: {energy}")


# Please run this as the main file if you would like to test the simulation
# Also note the required file structure for read/write operations above (or
# comment them out) 
if __name__ == "__main__":
    stage = int(input("Which question to run? [2-5]: "))
    if (stage == 2):
        main_2()
    elif (stage == 3):
        main_3()
    elif (stage == 4):
        main_4()
    elif (stage == 5):
        main_5()
    elif (stage == 0):
        # tester for debugging
        test_main()
