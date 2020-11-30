#!/usr/bin/env python
# coding: utf-8

# ---
# 
# ## Universidad de Costa Rica
# ### Escuela de Ingeniería Eléctrica
# #### IE0405 - Modelos Probabilísticos de Señales y Sistemas
# 
# Segundo semestre del 2020
# 
# ---
# 
# * Estudiante: **Karol Liseth Hernández Morera**
# * Carné: **B63367**
# * Grupo: **1**
# 
# ---

# ### 4.1. - Modulación QPSK
# 
# * (50%) Realice una simulación del sistema de comunicaciones como en la sección 3.2., pero utilizando una modulación QPSK en lugar de una modulación BPSK. Deben mostrarse las imágenes enviadas y recuperadas y las formas de onda.
# 

# In[7]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import fft
def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales
    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)

def rgb_a_bit(imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).
    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = imagen.shape
    
    # Número total de pixeles
    n_pixeles = x * y * z

    # Convertir la imagen a un vector unidimensional de n_pixeles
    pixeles = np.reshape(imagen, n_pixeles)

    # Convertir los canales a base 2
    bits = [format(pixel,'08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

#---------------------------------------------------- Función modulador------------------------------------------------------

def modulador_IQ(bits, fc, mpp, x):
    '''Un método que simula el esquema de 
    modulación digital BPSK.
    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    
    Tc = 1 / fc  # periodo [s]
    
    tperiodo = np.linspace(0, Tc, mpp)
    if(x==1):
        portadora = np.sin(2*np.pi*fc*tperiodo)
    else:
        portadora = np.cos(2*np.pi*fc*tperiodo)
        
    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_TIQ = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # señal de información
 
    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bits):
        if bit == 1:
            senal_TIQ[i*mpp : (i+1)*mpp] = portadora
            moduladora[i*mpp : (i+1)*mpp] = 1
        else:
            senal_TIQ[i*mpp : (i+1)*mpp] = portadora * -1
            moduladora[i*mpp : (i+1)*mpp] = 0
    
    # 5. Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_TIQ, 2), t_simulacion)
    
    return senal_TIQ, Pm, portadora, moduladora  

def canal_ruidoso(senal_TIQ, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.
    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_TIQ.shape)

    # Señal distorsionada por el canal ruidoso
    senal_RIQ = senal_TIQ + ruido

    return senal_RIQ

def demodulador(senal_RIQ, portadora, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.
    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_RIQ
    M = len(senal_RIQ)

    # Cantidad de bits en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_RIQ = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)
    # Energía de un período de la portadora
    Es = np.sum(portadora* portadora)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_RIQ[i*mpp : (i+1)*mpp] * portadora
        Ep = np.sum(producto) 
        senal_demodulada[i*mpp : (i+1)*mpp] = producto
     

        # Criterio de decisión por detección de energía
        if Ep > 0:
            bits_RIQ[i] = 1
        else:
            bits_RIQ[i] = 0
        
    return bits_RIQ.astype(int), senal_demodulada


def bits_a_rgb(bits_RIQ, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación
    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_RIQ)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_RIQ, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)



#--------------------------- Parámetros---------------------------------------------------------------------

fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 25   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_I, Pm1, portadora1, moduladora1 = modulador_IQ(bits_Tx, fc, mpp,1)
senal_Q, Pm2, portadora2, moduladora2 = modulador_IQ(bits_Tx, fc, mpp,0)

Pm = Pm1 + Pm2

# SE SUMAN LAS SEÑALES 
senal_TIQ = senal_I + senal_Q
portadora = portadora1+portadora2
moduladora = moduladora1 + moduladora2

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_RIQ = canal_ruidoso(senal_TIQ, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_RIQ, senal_demodulada  = demodulador(senal_RIQ, portadora, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_RIQ, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_RIQ))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)



# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_TIQ[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_RIQ[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()



# ### 4.2. - Estacionaridad y ergodicidad
# 
# * (30%) Realice pruebas de estacionaridad y ergodicidad a la señal modulada `senal_Tx = senal_TIQ ` y obtenga conclusiones sobre estas.
# 

# In[14]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Creación del vector de tiempo
T = 150			# número de elementos (usamos solo 150, no el total de 500)
t_final = 500	# tiempo en segundos
t = np.linspace(0, t_final, T)

P = [np.mean(senal_TIQ[i]) for i in range(150)]  
plt.xlabel('Realizaciones del proceso')
plt.ylabel('Valor esperado del proceso')
plt.plot(P, lw=2)


#Se convierte en un entero, ya que la función mean devuelve un float
Promedio = np.mean(P, dtype=int)  
print("El valor promedio es de: ", Promedio)

# T valores de desplazamiento tau
desplazamiento = np.arange(T)
taus = desplazamiento/t_final
N = 10


# Inicialización de matriz de valores de correlación para las N funciones
corr = np.empty((N, len(desplazamiento)))

# Nueva figura para la autocorrelación
plt.figure()


# ### 4.3. - Densidad espectral de potencia
# 
# * (20%) Determine y grafique la densidad espectral de potencia para la señal modulada `senal_Tx = senal_TIQ`.
# 
# 

# In[13]:


# Transformada de Fourier

senal_f = fft(senal_TIQ )

# Muestras de la señal

Nm = len(senal_TIQ )

# Número de símbolos

Ns = Nm//mpp

# Tiempo del símbolo es igual periodo de la onda portadora

Tc = 1/fc

# Tiempo entre muestras (período de muestreo)

Tm = Tc / mpp

# Tiempo de la simulación

T = Ns * Tc

# Espacio de frecuencias

f = np.linspace(0.0,1.0/(2.0*Tm),Nm//2) 

# Gráfica

plt.plot(f, 2.0/Nm*np.power(np.abs(senal_f[0:Nm//2]),2))
plt.xlim(0,20000)
plt.grid()
plt.show()


# In[ ]:




