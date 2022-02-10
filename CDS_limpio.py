# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:09:16 2022

@author: pc

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root
"""

"""
def CDS(hazard_ini,fecha_final,tam_periodo,IR,RR,spreads):
    delta = 1/tam_periodo
    tamanio = fecha_final*tam_periodo
    tiempo = list(np.arange(delta, fecha_final+delta,delta))
    df = [np.exp(-delta*i*IR) for i in range(1,tamanio+1)]
    p_supervivencia = list(range(tamanio))
    p_supervivencia[0] = np.exp(-hazard_ini[0]*delta)

    cont = 0
    for i in tiempo:
        if delta<i and i<=1:
            cont += 1
            p_supervivencia[cont] = p_supervivencia[cont-1]*np.exp(-hazard_ini[0]*delta)
        if 1<i and i<=2:
            cont += 1
            p_supervivencia[cont] = p_supervivencia[cont-1]*np.exp(-hazard_ini[1]*delta)
        if 2<i and i<=3:
            cont += 1
            p_supervivencia[cont] = p_supervivencia[cont-1]*np.exp(-hazard_ini[2]*delta)
        if 3<i and i<=4:
            cont += 1
            p_supervivencia[cont] = p_supervivencia[cont-1]*np.exp(-hazard_ini[3]*delta)
        if 4<i and i<=5:
            cont += 1
            p_supervivencia[cont] = p_supervivencia[cont-1]*np.exp(-hazard_ini[4]*delta)

    p_default = [1-p for p in p_supervivencia]

    dq = []
    dq.append(p_default[0])
    for p in range(1,tamanio):
        dq.append(p_default[p]-p_default[p-1])
    
    pata_fija = [delta*df[t]*p_supervivencia[t] for t in range(tamanio)]
    pata_contingente = [(1-RR)*df[c]*dq[c] for c in range(tamanio)]
    pata_fija_valuacion = [sum(pata_fija[0:tam_periodo*(a+1)])*spreads[a]/10000 for a in range(fecha_final)]
    pata_contingente_valuacion = [sum(pata_contingente[0:tam_periodo*(a+1)]) for a in range(fecha_final)]

    return [pata_fija_valuacion[s]-pata_contingente_valuacion[s] for s in range(fecha_final)]
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
sol = root(fun=CDS, x0=(0.0011, 0.00234, 0.00364, 0.00436, 0.00542),args=(spreads,temp_spreads,IR), method='hybr')
hazar_final = [round(num, 5) for num in sol.x.tolist()]
hazar_final  #obtenemos las hazar que haven que el CDS valga cero.

plt.step(temp_spreads, hazar_final,drawstyle='steps-mid')
plt.xticks(temp_spreads)
plt.yticks(np.arange(min(hazar_final), max(hazar_final)+0.001, 0.002))
plt.title("Hazard rates", fontdict=None, loc='center', pad=None)
plt.xlabel('Temporalidades Spreads')
plt.show()

#esto realmente no sirve, lo queremos generalizar!
def probas_estandar_CDS(hazar_final,temp_spreads,tipo_probas):  #tipo_probas = 0 -> supervivencia; tipo_probas = 1 -> default
    fecha_final= temp_spreads[-1]
    tam_periodo = 4*1 #tamaño del periodo (fracciones de año), trimestres por default
    delta = 1/tam_periodo
    tamanio = fecha_final*tam_periodo
    tiempo = list(np.arange(delta, fecha_final+delta,delta))
    p_supervivencia = []
    p_supervivencia.append(np.exp(-hazar_final[0]*delta))
    i=0
    j=0
    while i<temp_spreads[-1]:
        while j<tamanio-1 and tiempo[j+1]<=temp_spreads[i]:
            p_supervivencia.append(p_supervivencia[j]*np.exp(-hazar_final[i]*delta))
            j+=1
        i+=1
    proba_surv = [p_supervivencia[tam_periodo*k-1] for k in temp_spreads]
    if tipo_probas == 0:
        return proba_surv
    else:
        return [1-p for p in proba_surv] 

probas_estandar_CDS(hazar_final,temp_spreads,tipo_probas=1)


t_a_calcular_survivencia_o_default = 0.25  #es más facil manejar todo en años
if t_a_calcular_survivencia_o_default <= temp_spreads[0]:
    k=0
elif t_a_calcular_survivencia_o_default > temp_spreads[-1]:  #esta condición se podría eliminar, no puede haber t > temp_spreads[-1]
    k = len(temp_spreads)-1
else: 
    k = 0
    while t_a_calcular_survivencia_o_default > temp_spreads[k]: 
        k+=1
integrated_hazard_rate = []
l = k  
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
integrated_hazard_rate #esta es la integral de 0 a t, de lambda(u)du
sum(integrated_hazard_rate)
print("La proba de default al tiempo {} es {}.".format(t_a_calcular_survivencia_o_default, 1-np.exp(-sum(integrated_hazard_rate) ) ))


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

probas_CDS(hazar_final,temp_spreads,0,10)

#[0.0011, 0.00234, 0.00364, 0.00436, 0.00542]