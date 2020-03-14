#=============================================================================================================#
import sys
import os
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from decimal import Decimal
from scipy.optimize import curve_fit
#=============================================================================================================#

#=============================================================================================================#
def scatter_plot(xdata,ydata,title):
  # simple scatter plot with same xdata but different y values

  #fig, ax=plt.subplots()
  if ydata.ndim > 1:
    for i in range(0,np.shape(ydata)[1]):
      plt.plot(xdata,ydata[:,i],'-',label=title[3+i])
  else:
    plt.plot(xdata,ydata,'-',label=title[3])

  plt.title(title[0])
  plt.xlabel(title[1])
  plt.ylabel(title[2])
  plt.legend(loc=5)
#=============================================================================================================#
