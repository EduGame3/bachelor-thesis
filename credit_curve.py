# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:05:43 2022

@author: pc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def CDS(hazard_ini,spreads,temp_spreads,IR):
    fecha_final= temp_spreads[-1]  #en años
    tam_periodo = 4*1 #tamaño del periodo (fracciones de año)
    IR = IR
    RR = 0.4
    delta = 1/tam_periodo
    tamanio = fecha_final*tam_periodo
    tiempo = list(np.arange(delta, fecha_final+delta,delta))
    df = [np.exp(-delta*i*IR) for i in range(1,tamanio+1)]
    p_supervivencia = []
    p_supervivencia.append(np.exp(-hazard_ini[0]*delta))
    i=0
    j=0
    while i<temp_spreads[-1]:
        while j<tamanio-1 and tiempo[j+1]<=temp_spreads[i]:
            p_supervivencia.append(p_supervivencia[j]*np.exp(-hazard_ini[i]*delta))
            j+=1
        i+=1
    p_default = [1-p for p in p_supervivencia] 
    dq = []
    dq.append(p_default[0])
    for p in range(1,tamanio):
        dq.append(p_default[p]-p_default[p-1])
    pata_fija = [delta*df[t]*p_supervivencia[t] for t in range(tamanio)]
    pata_contingente = [(1-RR)*df[c]*dq[c] for c in range(tamanio)]
    pata_fija_valuacion = [sum(pata_fija[0:tam_periodo*temp_spreads[a]])*spreads[a]/10000 for a in range(len(temp_spreads))]
    pata_contingente_valuacion = [sum(pata_contingente[0:tam_periodo*temp_spreads[a]]) for a in range(len(temp_spreads))]
    return [pata_fija_valuacion[s]-pata_contingente_valuacion[s] for s in range(len(temp_spreads))]

#Comprobamos los resultados obtenidos, con los ejemplos:
    
#IBM
IR = 0.02
spreads = [6.576,10.23,13.915,16.748,19.581]
temp_spreads = [1,2,3,4,5]

#Brigo 1
IR = 0.05
spreads = [16,29,45,50,58]
temp_spreads = [1,3,5,7,10]

#Brigo2
IR = 0.05
spreads = [397,315,277,258,240]
temp_spreads = [1,3,5,7,10]

#Resolvemos
sol = root(fun=CDS, x0=(0.001, 0.002, 0.0036, 0.004, 0.005),args=(spreads,temp_spreads,IR), method='hybr')
hazar_final = [round(num, 5) for num in sol.x.tolist()]
hazar_final  #obtenemos las hazar que hacen que el CDS valga cero.

#graficamos las hazard obtenidas
plt.step(temp_spreads, hazar_final,drawstyle='steps-mid')
plt.xticks(temp_spreads)
plt.yticks(np.arange(min(hazar_final), max(hazar_final)+0.001, 0.002))
plt.title("Hazard rates", fontdict=None, loc='center', pad=None)
plt.xlabel('Temporalidades Spreads')
plt.show()


def probas_CDS(hazar_final,temp_spreads,tipo_proba,t):
    t_a_calcular_survivencia_o_default = t  #es más facil manejar todo en años
    if t_a_calcular_survivencia_o_default <= temp_spreads[0]: k=0
    elif t_a_calcular_survivencia_o_default > temp_spreads[-1]:  #esta condición se podría eliminar, no puede haber t > temp_spreads[-1]
        k = len(temp_spreads)-1
    else: 
        k = 0
        while t_a_calcular_survivencia_o_default > temp_spreads[k]: k+=1
    l = k
    integrated_hazard_rate = []
    if t_a_calcular_survivencia_o_default <= temp_spreads[0]:
        integrated_hazard_rate.append(hazar_final[0]*t_a_calcular_survivencia_o_default)
    else:
        while k>0:
            integrated_hazard_rate.append(hazar_final[0])
            k-=1
            if k<=0:
                break
            while k>0:
                integrated_hazard_rate.append((temp_spreads[k]-temp_spreads[k-1])*hazar_final[k])
                k-=1
                if k<=0:
                    break
        integrated_hazard_rate.append((t_a_calcular_survivencia_o_default-temp_spreads[l-1])*hazar_final[l])
    if tipo_proba == 0: #default
        return 1-np.exp(-sum(integrated_hazard_rate))
    else: #supervivencia
        return np.exp(-sum(integrated_hazard_rate))

#el último argumento es un número en años; es el instante para el cual se quiere calcular la proba de default o supervivencia
probas_CDS(hazar_final,temp_spreads,0,5)  