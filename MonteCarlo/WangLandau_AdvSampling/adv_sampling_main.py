#! /Users/Noah/opt/anaconda3/envs/applied-ai/bin/python3
"""
Main python file for PSET 3 in MENG 255

Author: Noah Dorhmann
Date: (For) 9 March 2022
"""

import numpy as np
import pandas as pd
import numba 
from numba import jit
import sys
import os
import hoomd
# need to figure out hoomd.dlext

#@jit(nopython=True)
def V(x):
    return x**4 - 4 * x ** 2 + 1


#@jit(nopython=True)
def is_flat(H: np.array) -> bool:
    """
    determine whether a histogram is flat based on its values
    """
    
    ### needs to be fixed since the edge bins (0 and N-1) are never getting filled,
    ### which means that this function never says that h is flat
    
    mean = H.mean()

    for i, h in enumerate(H):
        if (i == 0):
            continue
        elif (h < 0.8 * mean):
            return False

    return True


#@jit(nopython=True)
def g_ln_finder():

    # global definition of bins
    bins = np.linspace(-3., 3., 101)

    # initialize the H(E) and g(E) histograms 
    H = np.zeros(101)
    g = np.ones(101)

    # natural log form of the density of states
    g_ln = np.zeros(101)


    # define the staring f parameter (will be updated as f <- np.sqrt(f) 
    f = np.e

    # trial displacements - several methods to test 

    cycle = 0

    # log iterations
    ita = 0


    # do 27 iterations to get f down to a small enough value
    while (cycle < 27):


        print(str(ita))
        # choose x_0  and update bins
        x_curr = -3.0 + 6 * np.random.random()
        E_curr = V(x_curr)

        index = np.digitize(x_curr, bins)
        H[index] += 1

        g_ln[index] += np.log(f)

        while(True):

            x_new = -3.0 + 6 * np.random.random()

            # if the code is changed for x to move freely, be careful with the
            # indexing here
            index_new = np.digitize(x_new,bins)
            E_new = V(x_new)

            prob = np.exp(-(E_new-E_curr)) * np.exp(g_ln[index] - g_ln[index_new])

            rand = np.random.random()

            if (rand < prob):
                # acc 
                x_curr = x_new
                E_curr = E_new
                index  = index_new

                H[index] += 1
                g_ln[index] += np.log(f)
            else:
                H[index] += 1
                g_ln[index] += np.log(f)



            if (ita > 10000 and is_flat(H)):
                break


            ita += 1




        # update cycle count, reset H, update f
        cycle += 1
        H = np.zeros(101)
        f = np.sqrt(f)


    return g_ln

def disp():
    """
    generate a displacement
    """

    return -0.1 + 0.2 * np.random.random()


def fixed_weight_walk(g_ln:np.array, bins:np.array) -> np.array:
    """
    Takes in the fixed ln(g) and performs a biased random walk
    """
    H = np.zeros(101) 

    x_curr = -3.0 + 6.0 * np.random.random() 
    en_curr = V(x_curr)

    index = np.digitize(x_curr,bins)
    H[index] += 1 


    ita =  10000000

    for i in range(ita):
        loc_disp = disp()
        x_new = x_curr + loc_disp
        # x_new = -3.0 + 6.0 * np.random.random()
        if (x_new > bins[-1]):
            x_new -= 2 * loc_disp
        elif (x_new < bins[0]):
            x_new += 2 * loc_disp


        index_new = np.digitize(x_new,bins)

        en_new = V(x_new)
        prob = np.exp(-(en_new-en_curr)) * np.exp(-(g_ln[index] - g_ln[index_new]))
        # prob = np.exp(-(en_new-en_curr)) * (-(g_ln[index] - g_ln[index_new]))

        rand = np.random.random()

        if (rand < prob):
            x_curr = x_new
            en_curr = en_new
            index = index_new

            H[index] += 1
        else:
            H[index] += 1

    return H



def main():
    """
    main function for debugging
    """

    print("Done!")



# control the execution of the program
if __name__ == "__main__":
    main()
