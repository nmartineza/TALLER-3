# TALLER-3
import numpy as np
A=np.array([[3.,6.,7.],[1.,7.,8.],[2.,2.,7.]])


### Matriz Covarianza###
def sumatoria(j):
    respuesta=0
    resta=0
    for  i in range(A.shape[1]):
        
        respuesta=respuesta+A[i,j]-np.mean(A[:,j])
    return respuesta

print(sumatoria(2))
def cov(A):
    respuesta=np.empty_like(A)
    print(respuesta)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if(i==j):
                respuesta[i,j]=((sumatoria(j))**2)/A.shape[1]-1
            #else:
                #respuesta[i,j]=sum(((A[:,j]-np.mean(A[:,j]))*(A[:,j+1]-np.mean(A[:,j+1]))))/A.shape[1]-1
    return respuesta

print(cov(A))
    
#for i in range(A.shape[1]):
    #for c in range(A.shape[1]):
     #   A[i,:]=A[i,j]
    #for j in range(A.shape[1]):
        
