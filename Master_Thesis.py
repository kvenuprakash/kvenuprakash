# Non Isothermal data of Epoxy and PU can used for KAS model_Verified on 12.08.2022

from numpy.testing._private.utils import tempdir	# Importing all the required librarires
from numpy.ma.core import append
from numpy.lib.function_base import append
import pandas as pd
import xlrd
import openpyxl
import matplotlib.pyplot as plt
import math
from math import cos, exp, pi
from scipy.integrate import quad
import numpy as np
from math import e
from statistics import mean
from matplotlib import pyplot
from tabulate import tabulate

loc = ("D1.xlsx")                           	# Load excel file 
wb = xlrd.open_workbook(loc)                	# Open workbook
t25 = []; t5 = []; t8 = []; t10 = [];       	# Empty lists creation, tHeatingRate

#*************Calculation of Connversion*************#  
for s in range (0,4):             	          # Loop for no.of sheets, Each sheet stands for a heating rate           
  sheet = wb.sheet_by_index(s)    
  step = 0.025                              	# Step size for conversion
  temp = []                                 	# Creation of empty list to store temperature values
  for i in range(2,sheet.nrows):            	# Loop for no.of rows in a sheet
    try:
      if (float(sheet.cell_value(i,3)) >= step):  	# Checking whether the value in respective sheet cell is greater than step value
        c=sheet.cell_value(i,0)             	# Temperature in degree Centigrade                       
        k= c+273.3                          	# Temperature in Kelvin
        #print(k)
        temp.append(k)                      	# Addition of integers to a list
        step = step + 0.025                 	# Conversion step increament
      if(s == 0):			                      	# if sheet number equals 0				
        t25 = temp 				                    # Import all the values from temp list to t25 list
      elif(s==1):				
        t5 = temp;				
      elif(s==2):
        t8 = temp;
      else:
        t10 = temp
    except:
      continue
print("t2.5", t25)				                    # Print all the values of 't25 list' temperature at each covnersion after the completion of 'for loop'
print("t5", t5)
print("t8", t8)
print("t10", t10)

conv = np.linspace(0.025, 0.975, 39)		      # Declaration of conversion values from 0.025 to 0.975
print("conversion", conv)

#*************Graph plots: Temperature vs Connversion*************#  
plt.plot(t25, conv)
plt.plot(t5, conv)
plt.plot(t8, conv)
plt.plot(t10, conv)
plt.xlabel('Temperature')                     # Plot temperature on x-axis 
plt.ylabel('Conversion')                      # Plot conversion on y-axis 
plt.title('Temp dependent Conversion')
plt.show

#*************Calculation: Kissinger Akahira-Sunose model*************# 
activation_energy=[]
qlist = [2.5, 5, 8, 10]                     	# Heatramp declaration
figure, axes=plt.subplots(nrows=2,ncols=2)  
for j in range (0,len(t25)):                
    list_inv = []; list_log = []; 
    inv25 = 1/t25[j]                        	# X label calculation: (1/T)
    inv5 = 1/t5[j]
    inv8 = 1/t8[j]
    inv10 = 1/t10[j]

    sq25 = t25[j]*t25[j]                   	  # Y label calculation: ln(q/T²)
    sq5 = t5[j]*t5[j]
    sq8 = t8[j]*t8[j]
    sq10 = t10[j]*t10[j]
    div_sq25 = 2.5/sq25                     
    div_sq5 = 5/sq5
    div_sq8 = 8/sq8
    div_sq10 = 10/sq10
    l_y25 = math.log(div_sq25)              
    l_y5 = math.log(div_sq5)
    l_y8 = math.log(div_sq8)
    l_y10 = math.log(div_sq10)

    list_inv.append(inv25)                  	# Add individual results into a list
    list_inv.append(inv5)
    list_inv.append(inv8)
    list_inv.append(inv10)
    list_log.append(l_y25)
    list_log.append(l_y5)
    list_log.append(l_y8)
    list_log.append(l_y10)

    plt.subplot(1, 2, 1)                    	# Graph Subplot(row 1, col 2, index 1)
    plt.plot(list_inv,list_log)
    plt.xlabel('1/T')
    plt.ylabel('ln(q/T²)')
    plt.title('Kissinger-Akahira-Sunose(KAS)')
    plt.show

#*************Calculation of Slope and Activation energy *************# 
    model = np.polyfit(list_inv,list_log,1)	  # Polyfit the isoconversional lines and calculation slope
    slope = model[0]
    act_energy = slope * -8.3145		          # Multiplication of slope with universal gas constant
    #print(slope)
    activation_energy.append(act_energy)
print("Activation energy(Ea)", activation_energy)

plt.subplot(1, 2, 2)                        	
plt.plot(conv, activation_energy)		          # Plot the graph for activation energy vs conversion
figure.tight_layout(pad=3.0)
plt.xlabel('Conversion')			                # Plot Conversion on X-axis 
plt.ylabel('Activation energy)')		          # Plot Activation energy on Y-axis
plt.title('Conv vs Activation energy')
plt.show

#*************Calculation of Time Prediction*************# 
b=5; R=8.3145;              			          # b is sleceted heating rate
sum = 0;					
def func(t):					                      # Integral function declaration of exponential equation
    return exp(-Ea/(R*t))			              
t=[]						
for r in range(1, len(activation_energy)):	
  Ea= activation_energy[r]			
  l1=t5[r-1]					                      # Lower limit of the integral function
  l2=t5[r]					                        # Upper limit of the integral function
  #print(l1)
  solution, err = quad(func, l1, l2)		    # Integrating the function mentioned above (exponential factor) 
  ca = solution / ((b)*(exp(-Ea/(R*363.3))))	# Calculation of cure time prediction
  #print(ca)					
  t.append(ca)  				
print("Predicted time: t =", t)			        # Print the predicted cure time
a= len(t)
b= len(conv)
print("length of t", a)
print("length of conversion", b)
plt.plot(t, conversion)                     # Error due to varying x-axis and y-axis lengths. Therfore 
figure.tight_layout(pad=3.0)                # Code can be executed without plotting this  
plt.xlabel('time')
plt.ylabel('Conversion)')
plt.title('time vs Conversion')
plt.show
figure.tight_layout(pad=3.0)
