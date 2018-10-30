#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[24]:


###Punto 3###
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from cmath import exp, pi
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

incompletos=np.genfromtxt('incompletos.dat')
signal=np.genfromtxt('signal.dat')
x=signal[:,0]
y=signal[:,2]

###Inciso b###
plt.figure()
plt.title('Grafico signal')
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

###Inciso c###
def fourier(x):
    long= x.shape[0]
    if (long <= 1): 
        return x
    else:
        pares = fourier(x[0::2])
        impares =  fourier(x[1::2])
        T= [exp(-2j*pi*k/long)*impares[k] for k in range(long//2)]
        return [pares[k] + T[k] for k in range(long//2)] + [pares[k] - T[k] for k in range(long//2)]
 

    ###Inciso d###

plt.figure()
plt.title("Transformada de fourier")
plt.xlabel('Frecuencia')
plt.ylabel('Transformada')
dx=y[1]-y[0]
freq = fftfreq(len(y),dx)*10000 # Frecuencias
nuevoa=fourier(y)
tfourier=[]
filtrada=[]
for i in range(y.shape[0]):
    tfourier.append(abs(nuevoa[i]/ y.shape[0]))
    filtrada.append(abs(nuevoa[i]/ y.shape[0]))
    
plt.plot(freq,tfourier)
plt.grid(True)
plt.show()

###Inciso e###
maximos=argrelextrema(freq, np.greater)
print(maximos)

###Inciso f###

plt.figure()
plt.title("Transformada de fourier")
plt.xlabel('Frecuencia')
plt.ylabel('Transformada')

for j in range(len(filtrada)):
    if(freq[j]>=1000):
        filtrada[j]=0
    else:
        filtrada[j]=filtrada[j]
        
plt.plot(freq,filtrada)
plt.grid(True)
plt.show()

###Inciso g###

print("En los datos incompletos no se puede realizar la transformada de Fourier, ya que el numero de datos es mucho menos al numero de datos de signal; tambien se puede saber de antemano que la funcion puede que no sea continua")

###Inciso h###
x0=incompletos[0]
y0=incompletos[1]

quad=interp1d(x0,y0,kind='quadratic')
qbic=interp1d(x0,y0,kind='cubic')
lin=np.linspace(min(x0),max(x0),512)


# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook
