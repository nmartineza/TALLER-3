#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[8]:


import numpy as np 

A=np.array([[3.,6.,7.],[1.,7.,8.],[1.,2.,7.]])


def var(j):
    acumulado=0.
    for i in range(A.shape[1]):
        resta=((A[i,j]-np.mean(A[:,j]))*(A[i,j]-np.mean(A[:,j])))
        acumulado=acumulado+resta
    
    acumuladof=acumulado/(A.shape[1])
    return acumuladof

def var1(j,c):
    acumulado=0.
    
    for i in range(A.shape[1]):
        resta=((A[i,c]-np.mean(A[:,c]))*(A[i,j]-np.mean(A[:,j])))
        acumulado=acumulado+resta
    
    acumuladof=acumulado/(A.shape[1])
    return acumuladof

def matriz():
    respuesta=np.empty_like(A)
    c=0
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if(i==j):
                respuesta[i,j]=var(j)
                
            elif(i!=j):
                respuesta[i,j]=var1(j,c)
        c=c+1      
                
    return respuesta

print(matriz())
print(np.cov(A))


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
