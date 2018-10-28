#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

###Punto 2###
###Inciso a###
datos0=np.genfromtxt('WDBC.dat',delimiter=',')

datos1=datos0[:,2:]
print(np.shape(datos1))


N=np.empty_like(datos1)
for j in range(N.shape[1]):
    media=np.mean(datos1[:,j])
    for i in range(N.shape[0]):
        N[i,j]=datos1[i,j]-media
        
###Inciso b###

def var(j):
    acumulado=0.
    for i in range(datos1.shape[0]):
        resta=((datos1[i,j]-np.mean(datos1[:,j]))*(datos1[i,j]-np.mean(datos1[:,j])))
        acumulado=acumulado+resta
    
    acumuladof=acumulado/(datos1.shape[0])
    return acumuladof

def var1(j,c):
    acumulado=0.
    
    for i in range(datos1.shape[0]):
        resta=((datos1[i,c]-np.mean(datos1[:,c]))*(datos1[i,j]-np.mean(datos1[:,j])))
        acumulado=acumulado+resta
    
    acumuladof=acumulado/(datos1.shape[0])
    return acumuladof
###Inciso d###
id=[]
pos=[]
##############
def matriz():
    respuesta=np.zeros((30,30))
    
    c=0
    for i in range(respuesta.shape[0]):
        for j in range(datos1.shape[1]):
            if(i==j):
                respuesta[i,j]=var(j)
                id.append(var(j))
                if(respuesta[i,j]==323597.6708928503 or respuesta[i,j]==123625.90307986432 or respuesta[i,j]==2065.794620508682):
                    pos.append(j)
                    
                
            elif(i!=j):
                
                respuesta[i,j]=var1(j,c)
        c=c+1      
                
    return respuesta

matrix=matriz()

###Inciso c###

valores,vectores=np.linalg.eig(matrix)
print("Inciso b")
datos=[]

for i in range(30):
    valor=valores[i]
    vector=vectores[i]
    #print("El valor es")
    #print(valor)
    #print("y su vector asociado es")
    #print(vector)
    if(i==0 or i==1):
        datos.append(np.matmul(N,vector))
        
###Inciso d###
#Las columnas con mayor  autovalor son las mas importantes. Tienen mayor varianza. Se creo un arreglo ordenado de menor a mayor en donde se obtuvieron los amyores valores de varianza; su posicion exacta se agrego en el arreglo llamado pos.
ids=sorted(id)
#print (pos)
#A partir de la impresion de pos, es posible saber que los parametros mas importantes son los encontrados en las columnas 3,13 y 23, puesto que tienen mayor varianza.

###Inciso e###
plt.figure()
plt.title("Proyeccion de datos")
plt.scatter(datos[0],datos[1])
plt.grid(True)
plt.show()

###Inciso f###


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
