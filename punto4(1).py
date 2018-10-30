#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[ ]:



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq,ifft,fft2,ifft2



imagen=Image.open('Arboles.png')
X=np.array(imagen)
plt.figure()
plt.imshow(X)
plt.colorbar()
plt.show()

#print(np.shape(X))

#def fourier(arreglo):
    #transformada=np.empty_like(arreglo)
    #for i in range(transformada.shape[1]):
        #transformada[i,:]=fft(arreglo[i,:])
        
    #return transformada

#def fourieri(arreglo):
    #transformada=np.empty_like(arreglo)
    #for i in range(transformada.shape[1]):
        #transformada[i,:]=ifft(arreglo[i,:])
        
   # return transformada

#print(fourier(X))


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
