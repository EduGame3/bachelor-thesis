# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy import exp as exp

"""
Fórmula 3.6 (ecuación 1.19 en el overleaf)
Tabla 3.1

1 pb = 0.01 % = 0.0001
"""

#Típicamente las T están espaciadas trimestralmente.
#10 de julio de 2007
#1Y -> (T_a,T_b]
a=0
b=4
#(T_0,T_1,T_2,T_3,T_4]

S_i= 0.003 #0.00016 #16bps
delta1 = 90/360    #30/360 
delta2 = 90/360
delta3 = 90/360
delta4 = 90/360

T1 = delta1
T2 = T1+delta2
T3 = T2+delta3
T4 = T3+delta4

"""
Factores de descuento
P(0,(T_0+T_1)/4) = p1 = P(0,T1)
P(0,(T_1+T_2)/4) = p2 = P(0,T2)
P(0,(T_2+T_3)/4) = p3 = P(0,T3)
P(0,(T_3+T_4)/4) = p4 = P(0,T4)
"""

p1 = 0.05
p2 = 0.05
p3 = 0.05
p4 = 0.05

"""
Probabilidades de supervivencia
Q(\tau > t) = e{-t*\lambda(u)}
Q(\tau > T_0) = e{-T_0*lambda_i}
"""

lambda1=0.00267    #lambda(u) = 0.267%
lambda2=0.00267
lambda3=0.00267
lambda4=0.00267

q1 = np.exp(-T1*lambda1)
q2 = np.exp(-T2*lambda2)
q3 = np.exp(-T3*lambda3)
q4 = np.exp(-T4*lambda4) #0.997, coincide con el libro

#dq1= dQ(\tau \in (T_0,T_1))=P(T_0 < \tau < T_1) = F(T_1) - F(T_0) = 1-exp(-T_1*lambda1)
#dq2= dQ(\tau \in (T_1,T_2))=P(T_1<\tau<T_2)=F(T_2)-F(T_1)=exp(-T_1*lambda1)-exp(-T_2*lambda2)
#dq3= dQ(\tau \in (T_2,T_3))=exp(-T_2*lambda2)-exp(-T_3*lambda3)
#dq4= dQ(\tau \in (T_3,T_4))=exp(-T_3*lambda3)-exp(-T_4*lambda4) 

dq1 = 1-exp(-T1*lambda1)
dq2 = exp(-T1*lambda1)-exp(-T2*lambda2)
dq3 = exp(-T2*lambda2)-exp(-T3*lambda3)
dq4 = exp(-T3*lambda2)-exp(-T4*lambda3)

LGD = 0.6

s1 = -S_i*(p1*T1*dq1+p2*(T2-T1)*dq2+p3*(T3-T2)*dq3+p4*(T4-T3)*dq4)

s2 = -S_i*(p1*delta1*q1+p2*delta2*q2+p3*delta3*q3+p4*delta4*q4) 

s3 = LGD*(p1*dq1+p2*dq2+p3*dq3+p4*dq4)

print("Valor del CDS con lambda {} y S {} es de: {}".format(lambda1,S_i,s1+s2+s3))



"""
##### pdf risky bond and cds valuation
mca mbsiguetcmWocd(E,tblcs,r`tc,p`r`ls,X)>
blpgrt eulpy `s ep
blpgrt sibpy.betcdr`tc `s betcd
s<:
f0<o`lhm` t>mcesbtyWaueitbge(tblcs,p`r`ls,t)
f8 < o`lhm` t>ep.cxp(-betcd.qu`m(r`tc,:,t)[:\)
f < o`lhm` t>f0(t)*f8(t)
rcture (0-X)*E*betcd.qu`m(f,:,tblcs[-0\)[:\
mca prclbulWocd(i,E,tblcs,r`tc,p`r`ls,X)>
blpgrt eulpy `s ep
blpgrt sibpy.betcdr`tc `s betcd
s< :
f0 < o`lhm` t>survbv`oWaueitbge(tblcs,p`r`ls,t)
s < s+i*E*tblcs[:\*ep.cxp(-betcd.qu`m(r`tc,:,tblcs[:\)[:\)*f0(tblcs[:\)
"""


spreads = [16,29,45,50,58]#[397,315,277,258,240] #[16,29,45,50,58]#[6.576,10.23,13.915,16.748,19.581]  #estos son los buenos, ejemplo IBM
temp_spreads = [1,3,5,7,10] #datos en años

nuevos_spreads = [int(np.repeat(spreads[0],temp_spreads[0]))]
for i in range(1,len(spreads)):
    a = np.repeat(spreads[i],temp_spreads[i]-temp_spreads[i-1]).tolist()
    nuevos_spreads = nuevos_spreads + a


fecha_final= temp_spreads[-1]  #en años
tam_periodo = 4*1 #tamaño del periodo (fracciones de año)
IR = 0.05
RR = 0.4
delta = 1/tam_periodo
tamanio = fecha_final*tam_periodo

#hazard_ini = [0.001100,0.002348,0.003646,0.004364,0.005421]  #estos son los buenos
hazard_ini =  [0.001100,0.002348,0.003646,0.004364,0.005421]

tiempo = list(np.arange(delta, fecha_final+delta,delta))

df = [np.exp(-delta*i*IR) for i in range(1,tamanio+1)]

p_supervivencia = []

p_supervivencia.append(np.exp(-hazard_ini[0]*delta))

i=0
j=0
while i<temp_spreads[-1]:
    #print("-- Estamos en el spread --",temp_spreads[i])
    while j<tamanio-1 and tiempo[j+1]<=temp_spreads[i]:
        p_supervivencia.append(p_supervivencia[j]*np.exp(-hazard_ini[i]*delta))
        #print("j vale",j)
        j+=1
    i+=1
#    while temp_spreads[i]<tiempo[j-1] and tiempo[j-1]<=temp_spreads[i+1]:
#        p_supervivencia.append(p_supervivencia[j-1]*np.exp(-hazard_ini[i+1]*delta))
#        print("Segundo nivel, tiempo",j)
#        j+=1
    
p_supervivencia

p_default = [1-p for p in p_supervivencia]

dq = []
dq.append(p_default[0])
for p in range(1,tamanio):
    dq.append(p_default[p]-p_default[p-1])
    
pata_fija = [delta*df[t]*p_supervivencia[t] for t in range(tamanio)]
pata_contingente = [(1-RR)*df[c]*dq[c] for c in range(tamanio)]
pata_fija_valuacion = [sum(pata_fija[0:tam_periodo*temp_spreads[a]])*spreads[a]/10000 for a in range(len(temp_spreads))]
pata_contingente_valuacion = [sum(pata_contingente[0:tam_periodo*temp_spreads[a]]) for a in range(len(temp_spreads))]
mtm = [pata_fija_valuacion[s]-pata_contingente_valuacion[s] for s in range(len(temp_spreads))]
mtm

#buscando el solver
from scipy.optimize import fsolve

def mtm(lista1,lista2):
    return [a_i - b_i for a_i, b_i in zip(lista1, lista2)]

starting_guess = spreads

print(fsolve(mtm(pata_fija_valuacion,pata_contingente_valuacion),starting_guess))

#vamos
def solver(hazard_ini,spreads,temp_spreads):
    #hazard_ini[0],hazard_ini[1],hazard_ini[2],hazard_ini[3],hazard_ini[4] = parametros
    fecha_final= temp_spreads[-1]  #en años
    tam_periodo = 4*1 #tamaño del periodo (fracciones de año)
    IR = 0.05
    RR = 0.4
    delta = 1/tam_periodo
    tamanio = fecha_final*tam_periodo

    #spreads = [6.576,10.23,13.915,16.748,19.581]
    
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
    
    pata_fija = list(range(tamanio))
    for t in range(tamanio):
        pata_fija[t] = delta*df[t]*p_supervivencia[t]
    
    pata_contingente = list(range(tamanio))
    for c in range(tamanio):
        pata_contingente[c] = (1-RR)*df[c]*dq[c]
        
    pata_fija_valuacion = [sum(pata_fija[0:tam_periodo*temp_spreads[a]])*spreads[a]/10000 for a in range(len(temp_spreads))]
    pata_contingente_valuacion = [sum(pata_contingente[0:tam_periodo*temp_spreads[a]]) for a in range(len(temp_spreads))]
    return [pata_fija_valuacion[s]-pata_contingente_valuacion[s] for s in range(len(temp_spreads))]


root = fsolve(solver, [0.0015,0.0025,0.004,0.005,0.006])
root



import scipy.optimize as optim
initial_gues = [0.0015,0.0025,0.004,0.005,0.006]
result = optim.minimize(solver,initial_gues)
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
    

res = optim.minimize(solver,(0.0015,0.0025,0.004,0.005,0.006),method='TNC',tol=1e-15)
fsolve(solver,)


from scipy.optimize import minimize
 
minimize(solver, initial_gues, method='nelder-mead')

#jueves, se logró!
from scipy.optimize import root
sol = root(fun=solver, x0=(0.00267, 0.00601, 0.01211, 0.0109, 0.0139),args=(spreads,temp_spreads), method='hybr')
hazar_final = [round(num, 5) for num in sol.x.tolist()]
hazar_final
[0.001100,0.002348,0.003646,0.004364,0.005421]


0.00200,0.003500,0.005500,0.006000,0.006500
