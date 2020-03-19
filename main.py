#======================================================================================================#
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
#======================================================================================================#
type_val = [1,1]  # [0] 1-show confirmed plots 2- show extrapolated plots, [1] - show state plots

# lists that will be used by the full program
confirmed_data = []
death_data = []
recovery_data = []
country_list = []

#wanted = ["China","US","Italy"]  # "Germany","Korea, South"
#wanted_num = np.size(wanted)
#wanted_pop = [1401710720.,329431067.,60252824.,83149300.,51780579.]
#wanted_pop_density = [418.,200.,200.,233.,517.]
#wanted_index = [0]*wanted_num
#wanted_fit_param = np.zeros((wanted_num,3))

wanted = []
wanted_num = 0 # np.size(wanted)
wanted_index = [0]*wanted_num

# actual China pop density 145, but will use an average of south, east and central
# actual US pop density 34, but will use number for city pop density

#======================================================================================================#
def get_country_index(num_countries):

    for i in range(num_countries):
        if country_list[i] in wanted:
            num = wanted.index(country_list[i])
            wanted_index[num] = i
#======================================================================================================#
def get_fit_param(x,y,namer):

    for i in range(np.shape(x)[0]):
        popt = tl.fit_function(namer,x[i],y[i])
        wanted_fit_param[i,:] = popt
#======================================================================================================#
def get_countries(num):

    for i in range(1,num):
        namer = confirmed_data[i][1]
        if namer not in country_list:
            country_list.append(namer)
#======================================================================================================#
def get_country_rates(num_countries,entries,num_days,country_rates):

    for i in range(num_countries):
        for j in range(1,entries):
            if country_list[i] == confirmed_data[j][1]:
                for k in range(4,num_days+4):  # offset by other data in file
                    country_rates[i,k-4] += float(confirmed_data[j][k])

    return country_rates
#======================================================================================================#
def get_same_start(country_rates,num_days,val):

    rates_resize = []

    for i in range(wanted_num):
        for j in range(num_days):
            if country_rates[wanted_index[i],j] > val:
                num = j
                break

        x_resize = np.array(range(num_days-num))
        y_resize = np.array(country_rates[wanted_index[i],num:])
        rates_resize.append([x_resize,y_resize])

    return(rates_resize)
#======================================================================================================#
def main():

    # read in data from csv file
    tl.get_data([confirmed_data,death_data,recovery_data])

    # size constants of files   ~~~ will need to make larger for all 3 files
    sizer = np.shape(confirmed_data) ; print(sizer)
    entries = sizer[0]
    num_days = sizer[1]-4   # first 4 columns are state,country,lat,long

    # sort data by countries 
    get_countries(entries)
    #print(country_list)

    num_countries = np.size(country_list)
    country_rates = np.zeros((num_countries,num_days))   # will make this a 3 dim vector for other dat

    # find the rates in each country per day
    country_rates = get_country_rates(num_countries,entries,num_days,country_rates)

    sorted_country = []
    for i in range(num_countries):
        sorted_country.append([country_list[i],country_rates[i,:]])

    sorted_country_list = sorted(sorted_country,key=lambda l:float(l[1][-1]), reverse=True)

    wanted = []
    for i in range(10):
        wanted.append(sorted_country_list[i][0])

    print('wanted',wanted)   
    wanted_num = np.size(wanted)
    wanted_index = [0]*wanted_num
    wanted_fit_param = np.zeros((wanted_num,3))

    # determine index of wanted countries from total array
    get_country_index(num_countries)

    print('index',wanted_index)

    # resize data to start at similar times
    rates_resize = get_same_start(country_rates,num_days,200)

    # find the fit parameters for the wanted countries
    x = [lis[0] for lis in rates_resize]
    y = [lis[1] for lis in rates_resize]

    for i in range(np.size(y)):    # scale rate by pop density
        y[i] = y[i] #/ wanted_pop_density[i]

    get_fit_param(x,y,pl.logistic)
    print(wanted_fit_param) ; print(wanted_index)

    # plot wanted data with extrapolation fit
    if type_val[0] > 0:
        fig, ax = plt.subplots()
        for i in range(len(wanted)):
            title = ['Confirmed Cases','Days','Num Affected',wanted[i]]
            pl.scatter_plot((x[i]),y[i],title,'.')
            if type_val[0] == 2:
                num_sim = 50
                modelPredictions = pl.logistic((range(num_sim)), *wanted_fit_param[i,:])
                pl.scatter_plot(range(num_sim),modelPredictions,title,'-')
        plt.show()

    # get state / province data
    state_list = []
    for i in range(entries):
        if confirmed_data[i][1] =='US': # US
            state_list.append([confirmed_data[i][0],i,(confirmed_data[i][4:])])
    
    # removing cruises
    i = 0
    while i < len(state_list):
        if ' Princess' in state_list[i][0]:
            print(state_list[i][0])
            state_list.remove(state_list[i])
            i -= 1
        i += 1

    state_list_resize = get_same_start(country_rates,num_days,1)

    sorted_state_list = sorted(state_list,key=lambda l:float(l[2][-1]), reverse=True)

    if type_val[1] == 1:
        fig, ax = plt.subplots()
        for i in range(10): 
            title = ['Confirmed Cases','Days','Num Affected',sorted_state_list[i][0]]
            x = range(num_days)
            y = np.array(sorted_state_list[i][2], dtype=np.float) #/ wanted_pop_density[1]
            pl.scatter_plot(x,y,title,'-')
        plt.show()

#======================================================================================================#

#================================#
if __name__ == '__main__':
  main()
#================================#