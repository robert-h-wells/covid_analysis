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
def scatter_plot(xdata,ydata,title,point):
  # simple scatter plot with same xdata but different y values

  #fig, ax=plt.subplots()
  if ydata.ndim > 1:
    for i in range(0,np.shape(ydata)[1]):
      plt.plot(xdata,ydata[:,i],'-',label=title[3+i])
  else:
    plt.plot(xdata,ydata,point,label=title[3])
    #plt.semilogy(xdata,ydata,'-',label=title[3])

  plt.title(title[0])
  plt.xlabel(title[1])
  plt.ylabel(title[2])
  plt.legend(loc=2)
#=============================================================================================================#

def exp_1(t,A,k1,B):
  return A*np.exp(-k1*t)+B

def exp_2(t,A,B,k1,k2,C):
  return A*np.exp(-k1*t)+B*np.exp(-k2*t)+C

def exp_3(t,A,B,C,k1,k2,k3,D):
  return A*np.exp(-k1*t)+B*np.exp(-k2*t)+C*np.exp(-k3*t)+D

def power_law_2(t,A,B,k1):
  return A*t**(k1)+B 
  #return A*t**(1./2.)+B   

def poly_2(t,A,B):
  return A*t**(-1/2)+B

def poly_3(t,A,B,C,D):
  return A*t**3+B*t**2+C*t+D

def poly_4(t,A,B,C,D,E):
  return A*t**4+B*t**3+C*t**2+D*t+E

def poly_5(t,A,B,C,D,E,F):
  return A*t**5+B*t**4+C*t**3+D*t**2+E*t+F

def logistic(t,A,B,C):
  return  B/(C+np.exp(-A*t))

#=============================================================================================================#
#def scatter_fit_plot(func,xdata,ydata,init,title,val):
def scatter_fit_plot(func,xdata,ydata,title):
  
  #popt, pcov = (curve_fit(func, xdata, ydata,p0=init,maxfev = 100400))
  popt, pcov = (curve_fit(func, xdata, ydata,maxfev = 100400))
  modelPredictions = func((xdata), *popt) 
  fit_val = np.zeros(int(np.size(popt)))
  for i in range(0,np.size(fit_val)):
    fit_val[i] = popt[i]

  print(title[3],popt)
  
  plt.plot(xdata,ydata,'.',label=title[3])
  plt.plot(xdata,modelPredictions,'-')  # label=title[3])
  #plt.text(max(xdata)/1.5,2.,'Fit Parameters')
  #plt.text(max(xdata)/1.8,1.85-val*0.15,' '.join(['%.2e' % (i,) for i in fit_val]))
  plt.title(title[0])
  plt.xlabel(title[1])
  plt.ylabel(title[2])
  plt.legend(loc=5)
#=============================================================================================================#
