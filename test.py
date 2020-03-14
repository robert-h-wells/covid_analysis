#=======================================================================================#
import sys
import os
import csv
import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from decimal import Decimal
import multiprocessing as mp
from itertools import repeat

import plots as pl
import tools as tl
#=======================================================================================#


confirmed_data = []
death_data = []
recovery_data = []
tl.get_data([confirmed_data,death_data,recovery_data])

sizer = np.shape(confirmed_data) ; print(sizer)
print(confirmed_data[0])

# sort by countries
country_list = []

num = 0

if 1==0:
    fig, ax = plt.subplots()
    for i in range(sizer[0]):
        if confirmed_data[i][1] =='US':
            num += 1
            vals = np.array([int(i) for i in confirmed_data[i][4:]])
            legend = str(confirmed_data[i][0])+','+str(confirmed_data[i][1])
            pl.scatter_plot(range(sizer[1]-4),vals,['Confirmed Cases','Days','Num Affected',legend])

    plt.show()
    print(num)

for i in range(sizer[0]):
    if confirmed_data[i][1] =='US':
        print(confirmed_data[i][0],confirmed_data[i][0][1])