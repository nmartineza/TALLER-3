
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
x0=incompletos[:,0]
y0=incompletos[:,2]
lin=np.linspace(min(x0),max(x0),512)
quad=interp1d(x0,y0,kind='quadratic')
fquad=quad(lin)
qbic=interp1d(x0,y0,kind='cubic')
fqbic=qbic(lin)


dl=lin[1]-lin[0]
frec = fftfreq(len(fquad),dl)*10000
frec0=fftfreq(len(x0),dl)*10000
fourierq=fourier(fquad)
fourierc=fourier(fqbic)
fouriery0=fft(y0)



f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(frec0, fouriery0)
ax2.plot(frec, fourierc)
ax3.plot(frec, fourierq)
ax1.set_title("Transformada de los datos originales")
ax2.set_title("Transformada de la interpolacion cuadratica")
ax3.set_title("Transformada de la interpolacion cubica")
plt.savefig('MartinezNicolas_TF_interpola.pdf')





