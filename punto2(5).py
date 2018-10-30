import numpy as np
import matplotlib.pyplot as plt

###Punto 2###
###Inciso a###
datos0=np.genfromtxt('WDBC.dat',delimiter=',',dtype='U16')
datosx=np.genfromtxt('WDBC.dat',delimiter=',')
MALIGNOS=[]
BENIGNOS=[]
copia=datos0[:,1]
datos1=datosx[:,2:]

for i in range(copia.shape[0]):
    if(copia[i]=='M'):
        MALIGNOS.append(datos1[i,:])
    elif(copia[i]=='B'):
        BENIGNOS.append(datos1[i,:])


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
print(matrix)

###Inciso c###

valores,vectores=np.linalg.eig(matrix)
print("Inciso b")

#for i in range(30):
    #vector=vectores[i]
    #print("El valor es")
    #print(valor)
    #print("y su vector asociado es")
    #print(vector)

PC1=np.matmul(MALIGNOS,vectores)
PC2=np.matmul(BENIGNOS,vectores)

###Inciso d###
#Las columnas con mayor  autovalor son las mas importantes. Tienen mayor varianza. Se creo un arreglo ordenado de menor a mayor en donde se obtuvieron los amyores valores de varianza; su posicion exacta se agrego en el arreglo llamado pos.
ids=sorted(id)
#print (pos)
#A partir de la impresion de pos, es posible saber que los parametros mas importantes son los encontrados en las columnas 3,13 y 23, puesto que tienen mayor varianza.
print np.shape(PC1)
###Inciso e###
plt.figure()
plt.title("Proyeccion de datos")
plt.scatter(PC1[:,0],PC1[:,1],c='c',label='Malignos')
plt.scatter(PC2[:,0],PC2[:,1],c='r',label='Benignos')
plt.legend()
plt.grid(True)
plt.show()

###Inciso f###
