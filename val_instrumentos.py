# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:20:31 2022

@author: pc
"""
import numpy as np
from scipy.stats import norm

#SWAPTION
delta  = 30/360 #igual para todos los periodos
t_swap = 0.05
sigma2 = 0.8
k = 0.7
n_pagos = 5 
N = 1000 #constante
t_0 =  2*delta #primera fecha de intercambio de tasas, en meses
t = delta

d1 = (np.log(t_swap / k) + 0.5*t_0*sigma2) / np.sqrt(t_0*sigma2)
d2 = d1 - 0.5*t_0*sigma2
Nd1 = norm.cdf(d1 , loc = 0 , scale = 1 )
Nd2 = norm.cdf(d2 , loc = 0 , scale = 1 )

swaption_pay = n_pagos*N*delta*(t_swap*Nd1-k*Nd2)

#P: bonos cupón cero para traer a VP los flujos.
def swaption(t,t_0,delta,t_swap,sigma2,n_pagos,N,P,payer):
    d1 = (np.log(t_swap / k) + 0.5*(t_0-t)*sigma2) / np.sqrt((t_0-t)*sigma2)
    d2 = d1 - 0.5*(t_0-t)*sigma2
    Nd1 = norm.cdf(d1 , loc = 0 , scale = 1 )
    Nd2 = norm.cdf(d2 , loc = 0 , scale = 1 )
    if payer==1:
        enes = t_swap*Nd1-k*Nd2
    else:
        enes = k*Nd2-t_swap*Nd1
    flujos = [i*N*delta*enes for i in P]
    return sum(flujos)

swaption(t,t_0=t_0,delta=delta, t_swap=t_swap, sigma2=sigma2, n_pagos=n_pagos, N=N, P=[0.95,0.9,0.85,0.8,0.75], payer=0)

#CAPLET y FLOORLET
delta  = 30/360 
t_forward = 0.05
sigma2 = 0.8
k = 0.7 
N = 1000 
T =  2*delta
P = 0.7  #factor de descuento de t a T (de la fecha de valuacion al flujo)
t = delta

d1 = (np.log(t_forward / k) + 0.5*T*sigma2) / np.sqrt(T*sigma2)
d2 = d1 - 0.5*T*sigma2
Nd1 = norm.cdf(d1 , loc = 0 , scale = 1 )
Nd2 = norm.cdf(d2 , loc = 0 , scale = 1 )

caplet =  N*P*delta*(t_forward*Nd1-k*Nd2)
floorlet =N*P*delta*(k*Nd2-t_forward*Nd1)

def caplet_floorlet(t,T,N,k,sigma2,delta,P,t_forward,caplet):
    d1 = (np.log(t_forward / k) + 0.5*(T-t)*sigma2) / np.sqrt((T-t)*sigma2)
    d2 = d1 - 0.5*(T-t)*sigma2
    Nd1 = norm.cdf(d1 , loc = 0 , scale = 1 )
    Nd2 = norm.cdf(d2 , loc = 0 , scale = 1 )
    if caplet==1:
        return N*P*delta*(t_forward*Nd1-k*Nd2)
    else:
        return N*P*delta*(k*Nd2-t_forward*Nd1)
    
caplet_floorlet(t,T=T,N=N,k=k,sigma2=sigma2,delta=delta,P=P,t_forward=t_forward,caplet=1)

#CAP y FLOOR
n_flujos = 2
deltas  = [30/360]*n_flujos  #vector de deltas
t_forwards = [0.05 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
sigmas2 =     [0.8 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
ks =          [0.7 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)] 
Ns = [1000]*n_flujos 
Ts =  [2*delta]*n_flujos
Ps =          [0.7 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
tp_ins = 0  #tipo de instrumento, 1 caplets, 0 floorlets
t = Ts[0]/2

instrumentos = []
for i in range(0,n_flujos):
    #print(Ts[i])
    instrumentos.append(caplet_floorlet(Ts[i],Ns[i],ks[i],sigmas2[i],deltas[i],Ps[i],t_forwards[i],caplet=ind_instrumento[i]))
instrumentos

sum(instrumentos)

def cap_floor(t,n_flujos,Ts,Ns,ks,sigmas2,deltas,Ps,t_forwards,tp_ins):
    instrumentos = []
    if n_flujos==1:
        instrumentos.append(caplet_floorlet(t,Ts[0],Ns[0],ks[0],sigmas2[0],deltas[0],Ps[0],t_forwards[0],caplet=tp_ins))
    else:
        for i in range(0,n_flujos):
            instrumentos.append(caplet_floorlet(t,Ts[i],Ns[i],ks[i],sigmas2[i],deltas[i],Ps[i],t_forwards[i],caplet=tp_ins))
    return sum(instrumentos)

cap_floor(t,n_flujos=n_flujos,Ts=Ts,Ns=Ns,ks=ks,sigmas2=sigmas2,deltas=deltas,Ps=Ps,t_forwards=t_forwards,tp_ins=tp_ins)

#FRA
N = 1000
Pt = 0.08
delta = 30/360
t_forward = 0.9
K = 0.7
payer = 1

N*P*delta*(t_forward-K)

def FRA(N,Pt,K,delta,t_forward,payer=1):
    if payer == 1:
        return N*Pt*delta*(t_forward-K)
    else:
        return N*Pt*delta*(K-t_forward)
    
FRA(N,P,K,delta,t_forward,payer=1)

#IRS
n_flujos = 5
Ns = [1000]*n_flujos
Ps = [0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
deltas = [30/360]*n_flujos
t_forwards = [0.9 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
Ks = [0.7 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
tipo = [1]*n_flujos

instrumentos = []
for i in range(0,n_flujos):
    instrumentos.append(FRA(Ns[i],Ps[i],Ks[i],deltas[i],t_forwards[i],payer=tipo[i]))
sum(instrumentos)

def IRS(n_flujos,Ns,Ps,deltas,t_forwards,Ks,tipo):
    instrumentos = []
    for i in range(n_flujos):
        instrumentos.append(FRA(Ns[i],Ps[i],Ks[i],deltas[i],t_forwards[i],payer=tipo[i]))
    return sum(instrumentos)

IRS(n_flujos,Ns,Ps,deltas,t_forwards,Ks,tipo)

#CMS
n_flujos = 5
t_swaps = [0.11]*n_flujos
sigmas2 = [0.04]*n_flujos
bs = [3]*n_flujos
deltas = [30/360]*n_flujos
Ps = [0.084651763,0.083699237,0.082200935,0.084811791,0.082386364]
     
     #0.085,0.084,0.082,0.085,0.082]
     #+ 0.01*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]

CMS_s = []
for i in range(n_flujos):
    CMS_s.append(Ps[i]*t_swaps[i]*np.exp((i+1)*deltas[i]*sigmas2[i]*t_swaps[i]*bs[i]))
r_CMS_s = [round(CMS_s[i],6) for i in range(n_flujos)]    
sum(r_CMS_s)

def CMS(n_flujos,t_swaps,sigmas2,bs,deltas,Ps):
    CMS_s = []
    for i in range(n_flujos):
        CMS_s.append(Ps[i]*t_swaps[i]*np.exp((i+1)*deltas[i]*sigmas2[i]*t_swaps[i]*bs[i]))    
    return CMS_s

CMS(n_flujos,t_swaps,sigmas2,bs,deltas,Ps)

########################## CVA ######################################

#Aqui vamos:
    #t <= T_0
q_T0_T = 0.2
payer=1   
LGD = 0.6
#CVA swaption
cva_swaption_t = LGD*swaption(t,t_0=t_0,delta=delta, t_swap=t_swap, sigma2=sigma2, n_pagos=n_pagos, N=N, P=[0.95,0.9,0.85,0.8,0.75], payer=payer)*q_T0_T
#CVA caplet y floorlet
cva_caplet_t = LGD*cap_floor(t,n_flujos=n_flujos,Ts=Ts,Ns=Ns,ks=ks,sigmas2=sigmas2,deltas=deltas,Ps=Ps,t_forwards=t_forwards,tp_ins=1)*q_T0_T
cva_floorlet_t=LGD*cap_floor(t,n_flujos=n_flujos,Ts=Ts,Ns=Ns,ks=ks,sigmas2=sigmas2,deltas=deltas,Ps=Ps,t_forwards=t_forwards,tp_ins=0)*q_T0_T

#CVA IRS
m = 10 #tamaño de la partición de default (del tiempo t a T_0)
q = [0.9 - 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]

#parámetros de los swaption
delta = 30/360
t_swap = 0.05
sigma2 = 0.8
k = 0.7
n_pagos = m 
N = 1000 #constante
t_0 =  2*delta #primera fecha de intercambio de tasas, en meses
t = [i*t_0/10 for i in range(m)] 
payer = 1

#exposure: valuar el swaption en cada periodo t de la partición del intervalo (t,T_0)
exposure = [swaption(j,t_0=t_0,delta=delta, t_swap=t_swap, sigma2=sigma2, n_pagos=n_pagos, N=N, P=[0.95,0.9,0.85,0.8,0.75], payer=1) for j in t]
cva_IRS_payer_t = sum([q[k] * exposure[k] * LGD for k in range(len(exposure))])

#CVA CMS
m = 10 #tamaño de la partición de default (del tiempo t a T_0, T_0 es la fecha de intercambio de la )
q = [0.9 - 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
t_swaps=[0.06 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
P_cms = [0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
P_cva = [0.07 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
sigmas2=[0.05 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
bs = [3 + np.random.randint(-4,4) for i in range(0,m)]
deltas = [30/360]*m


cva_CMS_t = LGD * sum([P_cva[k] * CMS(n_flujos=m,t_swaps=t_swaps,sigmas2=sigmas2,bs=bs,deltas=deltas,Ps=P_cms)[k] * q[k] for k in range(m)])


    