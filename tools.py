#=============================================================================================================#
import sys
import os
import csv
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from decimal import Decimal
from scipy.optimize import curve_fit
#=============================================================================================================#

#=============================================================================================================#
def get_data(fils):
    
    with open('time_series_19-covid-Confirmed.csv', 'rt') as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            fils[0].append(row)

    f.close()
    death_data = []
    with open('time_series_19-covid-Deaths.csv', 'rt') as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            fils[1].append(row)

    f.close()
    recovery_data = []
    with open('time_series_19-covid-Recovered.csv', 'rt') as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            fils[2].append(row)

    f.close()
#=============================================================================================================#