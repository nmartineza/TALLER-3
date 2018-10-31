#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[20]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft2,ifft2, fftshift, ifftshift

###Inciso a###
imagen=Image.open('Arboles.png')
X=np.array(imagen)

###Inciso b###
transf=fft2(X)
stransf=(abs(fftshift(transf)))
plt.figure()
plt.title('Transformada')
plt.imshow(stransf,cmap='gray')
plt.colorbar()
plt.savefig('MartinezNicolas_FT2D.pdf')

###Inciso c###
mayor=0
stransf2=fft2(X)

for i in range(stransf2.shape[0]):
    for j in range(stransf2.shape[0]):
        if(stransf2[i,j]>=970000.):
            stransf2[i,j]=0.
            
        else:
            stransf2[i,j]= stransf2[i,j]
            
filtrada=np.log(abs(fftshift(stransf2)))

###Inciso d###
plt.figure()
plt.title('Transformada filtrada')
plt.imshow(filtrada,cmap='gray')
plt.colorbar()
plt.savefig('MartinezNicolas_FT2D_filtrada.pdf')

###Inversa###
plt.figure()
plt.title('Imagen final')
plt.imshow(abs(ifft2((ifftshift(stransf2)))),cmap='gray')
plt.colorbar()
plt.savefig('MartinezNicolas_Imagen_filtrada.pdf')


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
