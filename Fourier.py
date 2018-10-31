#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[14]:


# In[16]:


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
plt.savefig('MartinezNicolas_signal.pdf')

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
plt.savefig('MartinezNicolas_TF.pdf')

###Inciso e###
altas=[]
for i in range(len(freq)):
    if(tfourier[i]>=0.7):
        altas.append(freq[i])
print('Las frecuencias principales de la señal son')
print(altas)

###Inciso f###

plt.figure()
plt.title("Transformada de fourier filtrada")
plt.xlabel('Frecuencia')
plt.ylabel('Transformada')

for j in range(len(filtrada)):
    if(freq[j]>=1000):
        filtrada[j]=0
    else:
        filtrada[j]=filtrada[j]
        
plt.plot(freq,filtrada)
plt.grid(True)
plt.savefig('MartinezNicolas_filtrada.pdf')

###Inciso g###

print("En los datos incompletos no se puede realizar la transformada de Fourier, ya que el numero de datos es mucho menos al numero de datos de signal; tambien se puede saber de antemano que la funcion puede que no sea continua")

###Inciso h###
x0=incompletos[:,0]
y0=incompletos[:,2]
lin=np.linspace(min(x0),max(x0),512)
quad=interp1d(x0,y0,kind='quadratic')
fquad=quad(lin)
qbic=interp1d(x0,y0,kind='cubic')
fqbic=qbic(lin)

###Inciso i###
dl=lin[1]-lin[0]
frec = fftfreq(len(fquad),dl)
frec0=fftfreq(len(x0),dl)
fourierq=fourier(fquad)
fourierc=fourier(fqbic)
fouriery0=fft(y0)



f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(frec0, fouriery0,c='c')
ax2.plot(frec, fourierq,c='g')
ax3.plot(frec, fourierc)
ax1.set_title("Transformada de los datos originales")
ax2.set_title("Transformada de la interpolacion cuadratica")
ax3.set_title("Transformada de la interpolacion cubica")
plt.savefig('MartinezNicolas_TF_interpola.pdf')


###Inciso h### Explicar diferencias
print('La interpolacion de datos cuadratico, centra la mayoria de ruido en su centro=0. De la misma manera la cubica; solo que el intervalo en el que se encuentra el ruido es mucho mas pequeño. En los datos originales, las perturbaciones se encuentran a traves de toda la grafica y de una forma mucho menos simetrica')
###Inciso i###
cuadraticaf=fourier(fquad)
cuadraticafB=fourier(fquad)
cubicaf=fourier(fqbic)
cubicafB=fourier(fqbic)
y0f=fft(y0)
y0fB=fft(y0)


###FILTRO 1000###
for j in range(len(cuadraticaf)):
    if(frec[j]>=1000):
        cuadraticaf[j]=0
        cubicaf[j]=0
    else:
        cuadraticaf[j]=cuadraticaf[j]
        cubicaf[j]=cubicaf[j]

for j in range(len(y0f)):
    if(frec0[j]>=1000):
        y0f[j]=0
    else:
        y0f[j]=y0f[j]

###FILTRO 500###
for j in range(len(cuadraticafB)):
    if(frec[j]>=500):
        cuadraticafB[j]=0
        cubicafB[j]=0
    else:
        cuadraticafB[j]=cuadraticafB[j]
        cubicafB[j]=cubicafB[j]

for j in range(len(y0fB)):
    if(frec0[j]>=500):
        y0fB[j]=0
    else:
        y0fB[j]=y0fB[j]


plt.subplot(321)
plt.plot(frec0,y0f,c='g',label="Orig")
plt.title('Filtro 1000Hz')
plt.legend()
plt.subplot(323)
plt.plot(frec,cuadraticaf,label="Cuad")
plt.legend()
plt.subplot(325)
plt.plot(frec,cubicaf,c='c',label="Cubica")
plt.legend()
plt.subplot(322)
plt.plot(frec0,y0fB,c='g',label="Orig")
plt.title('Filtro 500Hz')
plt.legend()
plt.subplot(324)
plt.plot(frec,cuadraticafB,label="Cuad")
plt.legend()
plt.subplot(326)
plt.plot(frec,cubicafB,c='c',label="Cubica")
plt.legend()
plt.savefig('MartinezNicolas_2Filtros.pdf')


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
