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
      plt.plot(xdata,ydata[:,i],'-',label=title[3+i],linewidth=10,markersize=10)
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
def extrap_plot(x,y,title):

  sizer = np.size(country_rates[1,:]) ; print(sizer)

  for i in range(wanted_num):
        legend = wanted[i]
        
        for j in range(num_days):  # shift time to where day 0 has > 200 confirmed cases
            if country_rates[wanted_index[i],j] > 200.:
                val = j
                break

        #x = np.array(range(np.size(country_rates[wanted_index[i],:])-j))  # orig

        x_resize = np.array(range(sizer-j)) 
        y_resize = np.array(country_rates[wanted_index[i],j:])
        pl.scatter_plot(x,(y)/wanted_pop_density[i],['Confirmed Cases','Days','Num Affected',legend],'.')
        num_sim = 40
        modelPredictions = pl.logistic((range(num_sim)), *wanted_fit_param[i,:])   # *wanted_pop_density[num]
        pl.scatter_plot(range(num_sim),modelPredictions,['Confirmed Cases','Days','Num Affected',legend],'-')

#=============================================================================================================#
def us_map():
  import plotly.plotly as py 
  import plotly.graph_objs as go 
  import pandas as pd 
    
  # some more libraries to plot graph 
  from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot 
    
  # To establish connection 
  init_notebook_mode(connected = True) 
    
    
  # type defined is choropleth to 
  # plot geographical plots 
  data = dict(type = 'choropleth', 
    
              # location: Arizoana, California, Newyork 
              locations = ['AZ', 'CA', 'NY'], 
                
              # States of USA 
              locationmode = 'USA-states', 
                
              # colorscale can be added as per requirement 
              colorscale = 'Portland', 
                
              # text can be given anything you like 
              text = ['text 1', 'text 2', 'text 3'], 
              z = [1.0, 2.0, 3.0], 
              colorbar = {'title': 'Colorbar Title Goes Here'}) 
                
  layout = dict(geo ={'scope': 'usa'}) 
    
  # passing data dictionary as a list  
  choromap = go.Figure(data = [data], layout = layout) 
    
  # plotting graph 
  iplot(choromap)
#=============================================================================================================#
