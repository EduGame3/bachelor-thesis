# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:38:52 2022

@author: pc
"""

import numpy as np
from scipy.stats import norm
import random

####################################### VALUACION #######################################

#-------------------------------------- SWAPTION --------------------------------------

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

delta  = 30/360 #igual para todos los periodos
t_swap = 0.05
sigma2 = 0.8
k = 0.7
n_pagos = 5 
N = 1000 #constante
t_0 =  2*delta #primera fecha de intercambio de tasas, en meses
t = delta
P = [0.95,0.9,0.85,0.8,0.75] #bonos cupón cero para traer a VP los flujos.

swaption(t,t_0=t_0,delta=delta, t_swap=t_swap, sigma2=sigma2, n_pagos=n_pagos, N=N, P=P, payer=0)

#-------------------------------------- CAPLET y FLOORLET --------------------------------------

def caplet_floorlet(t,T,N,k,sigma2,delta,P,t_forward,caplet):
    d1 = (np.log(t_forward / k) + 0.5*(T-t)*sigma2) / np.sqrt((T-t)*sigma2)
    d2 = d1 - 0.5*(T-t)*sigma2
    Nd1 = norm.cdf(d1 , loc = 0 , scale = 1 )
    Nd2 = norm.cdf(d2 , loc = 0 , scale = 1 )
    if caplet==1:
        return N*P*delta*(t_forward*Nd1-k*Nd2)
    else:
        return N*P*delta*(k*Nd2-t_forward*Nd1)

delta  = 30/360 
t_forward = 0.05
sigma2 = 0.8
k = 0.7 
N = 1000 
T =  2*delta
P = 0.7  #factor de descuento de t a T (de la fecha de valuacion al flujo)
t = delta
caplet = 1

caplet_floorlet(t,T=T,N=N,k=k,sigma2=sigma2,delta=delta,P=P,t_forward=t_forward,caplet=caplet)

#-------------------------------------- FRA --------------------------------------

def FRA(N,Pt,K,delta,t_forward,payer=1):
    if payer == 1:
        return N*Pt*delta*(t_forward-K)
    else:
        return N*Pt*delta*(K-t_forward)
    
N = 1000
Pt = 0.08
delta = 30/360
t_forward = 0.9
K = 0.7
payer = 1

FRA(N,P,K,delta,t_forward,payer=1)

#-------------------------------------- IRS --------------------------------------

def IRS(n_flujos,Ns,Ps,deltas,t_forwards,Ks,tipo):
    instrumentos = []
    for i in range(n_flujos):
        instrumentos.append(FRA(Ns[i],Ps[i],Ks[i],deltas[i],t_forwards[i],payer=tipo))
    return instrumentos

n_flujos = 20 #estamos pensando en pagos semestrales; duración 10 años
Ns = [1000]*n_flujos
Ps = [0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
deltas = [6*30/360]*n_flujos
t_forwards = [0.9 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos)]
Ks = [0.7]*n_flujos
tipo = [1]*n_flujos

default = random.uniform(0,deltas[0]*n_flujos)  #en años
flujos_restantes = []
for dato in np.cumsum(deltas):
    if dato > default:
        flujos_restantes.append(dato)
flujos_restantes
n_flujos_restantes = n_flujos - len(flujos_restantes)
Ns_n = Ns[n_flujos_restantes:]
deltas_n=deltas[n_flujos_restantes:]
Ks_n=Ks[n_flujos_restantes:]
Ps_n=[0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos_restantes)]
t_forwards_n=[0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,n_flujos_restantes)]

len(IRS(n_flujos,Ns,Ps,deltas,t_forwards,Ks,tipo))
IRS(n_flujos,Ns,Ps,deltas,t_forwards,Ks,tipo)

def expIRS(delta,n_flujos,default,Ns,Ps,deltas,t_forwards,Ks,tipo):
    flujos_restantes = []
    for dato in np.cumsum(deltas):
        if dato > default:
            flujos_restantes.append(dato)
    flujos_restantes = len(flujos_restantes)
    rebanada = n_flujos - flujos_restantes
    Ns_n = Ns[rebanada:]
    deltas_n=deltas[rebanada:]
    Ks_n=Ks[rebanada:]
    Ps_n=[0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,flujos_restantes)]
    t_forwards_n=[0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,flujos_restantes)]
    irs = IRS(n_flujos=flujos_restantes,Ns=Ns_n,Ps=Ps_n,deltas=deltas_n,t_forwards=t_forwards_n,Ks=Ks_n,tipo=tipo)
    return [irs[item] for item in range(len(irs)) if irs[item]>0]
   
default = random.uniform(0,deltas[0]*n_flujos)
default 

exposicion = expIRS(delta=deltas[0],n_flujos=n_flujos,default=default,Ns=Ns,Ps=Ps,deltas=deltas,t_forwards=t_forwards,Ks=Ks,tipo=1)
exposicion
len(exposicion)

a = [-2,-1,-0.1,-1,-2]    
[a[item] for item in range(len(a)) if a[item]>0]

#-------------------------------------- CMS --------------------------------------

def CMS(n_flujos,t_swaps,sigmas2,bs,deltas,Ps):
    CMS_s = []
    for i in range(n_flujos):
        CMS_s.append(Ps[i]*t_swaps[i]*np.exp((i+1)*deltas[i]*sigmas2[i]*t_swaps[i]*bs[i]))    
    return CMS_s

n_flujos = 5
t_swaps = [0.11]*n_flujos
sigmas2 = [0.04]*n_flujos
bs = [3]*n_flujos
deltas = [30/360]*n_flujos
Ps = [0.084651763,0.083699237,0.082200935,0.084811791,0.082386364]

CMS(n_flujos,t_swaps,sigmas2,bs,deltas,Ps)


####################################### CVA #######################################

#-------------------------------------- SWAPTION --------------------------------------

#t <= T_0
q_T0_T = 0.2
payer=1   
LGD = 0.6

cva_swaption_t = LGD*swaption(t,t_0=t_0,delta=delta, t_swap=t_swap, sigma2=sigma2, n_pagos=n_pagos, N=N, P=[0.95,0.9,0.85,0.8,0.75], payer=payer)*q_T0_T
cva_swaption_t

#-------------------------------------- CAPLET y FLOORLET --------------------------------------

cva_caplet_t = LGD*caplet_floorlet(t,T=T,N=N,k=k,sigma2=sigma2,delta=delta,P=P,t_forward=t_forward,caplet=1)*q_T0_T
cva_caplet_t
cva_floorlet_t=LGD*caplet_floorlet(t,T=T,N=N,k=k,sigma2=sigma2,delta=delta,P=P,t_forward=t_forward,caplet=0)*q_T0_T
cva_floorlet_t

#-------------------------------------- IRS --------------------------------------

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
exposure = [swaption(j,t_0=t_0,delta=delta, t_swap=t_swap, sigma2=sigma2, n_pagos=n_pagos, N=N, P=[0.95,0.9,0.85,0.8,0.75], payer=payer) for j in t]

cva_IRS_t = sum([q[k] * exposure[k] * LGD for k in range(len(exposure))])
cva_IRS_t

#-------------------------------------- CMS --------------------------------------

m = 10 #tamaño de la partición de default (del tiempo t a T_0, T_0 es la fecha de intercambio de la )
q = [0.9 - 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
t_swaps=[0.06 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
P_cms = [0.08 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
P_cva = [0.07 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
sigmas2=[0.05 + 0.05*np.random.normal(loc=0.0, scale=1.0) for i in range(0,m)]
bs = [3 + np.random.randint(-4,4) for i in range(0,m)]
deltas = [30/360]*m

cva_CMS_t = LGD * sum([P_cva[k] * CMS(n_flujos=m,t_swaps=t_swaps,sigmas2=sigmas2,bs=bs,deltas=deltas,Ps=P_cms)[k] * q[k] for k in range(m)])
cva_CMS_t
