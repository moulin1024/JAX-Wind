import numpy as np
import math
import fatpack
import matplotlib.pyplot as plt
import pandas as pd

#Create a function that reutrns the Goodman correction:
def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar

def Equivalent_bending_moment(M_ar,Neq,m):
    P = M_ar.shape
    M_sum = 0
    j = P[0] 
    for i in range(j):
        M_sum = math.pow(M_ar[i],m) + M_sum
    M_eq = math.pow((M_sum/Neq),(1/m))
    return M_eq

def get_DEL(y,Neq,m):
    S, Sm = fatpack.find_rainflow_ranges(y.flatten(), return_means=True, k=256)
    data_arr  = np.array([Sm , S ]).T
    M_ar = Goodman_method_correction(data_arr[:,1],data_arr[:,0],np.max(S))
    print(sum(M_ar.shape))
    M_eq = Equivalent_bending_moment(M_ar,Neq,m)
    return M_eq

