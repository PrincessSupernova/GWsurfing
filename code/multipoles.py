#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:29:27 2022

@author: jschnitt
"""

import numpy as np
import matplotlib.pyplot as plt
Nt = 1001
Nth = 41
Nph = 30
m1 = 0.75
m2 = 0.25
M = m1+m2
R = 100
Torb = 2.*np.pi*np.sqrt(R**3./M)
Omg = 2.*np.pi/Torb
dt = Torb/(Nt)
tt = np.zeros(Nt)
for it in range(Nt):
  tt[it]=it*dt
X1 = np.zeros((Nt,3))
X2 = np.zeros((Nt,3))
V1 = np.zeros((Nt,3))
V2 = np.zeros((Nt,3))
Mij = np.zeros((Nt,3,3))
hij_thph = np.zeros((Nt,Nth,Nph,3,3))
htp_thph = np.zeros((Nt,Nth,Nph,2,2))
dhtp_thph = np.zeros((Nt,Nth,Nph,2,2))
hP_thph = np.zeros((Nth,Nph))
hX_thph = np.zeros((Nth,Nph))
PP_thph = np.zeros((Nth,Nph))
PX_thph = np.zeros((Nth,Nph))
Dij = np.zeros((3,3))
for i in range(3):
  Dij[i,i]=1.

dz = 2./(Nth-1)
cth = np.arange(-1,1,dz)
cth = np.append(cth,1.0)
th = np.arccos(cth)
sth = np.sin(th)
dph = 2.*np.pi/Nph
pp = np.arange(0,2.*np.pi,dph)
ni_thph = np.zeros((Nth,Nph,3))
Lthi_thph = np.zeros((Nth,Nph,3))
Lphi_thph = np.zeros((Nth,Nph,3))
Pij_thph = np.zeros((Nth,Nph,3,3))
Lamijkl_thph = np.zeros((Nth,Nph,3,3,3,3))

#Calculate the Lambda projection tensor over the sphere of (z,phi)
for ith in range(Nth):
  for iph in range(Nph):
    ni_thph[ith,iph,:]=[sth[ith]*np.cos(pp[iph]),sth[ith]*np.sin(pp[iph]),cth[ith]]
    Lthi_thph[ith,iph,:]=[cth[ith]*np.cos(pp[iph]),cth[ith]*np.sin(pp[iph]),-sth[ith]]
    Lphi_thph[ith,iph,:]=[-np.sin(pp[iph]),np.cos(pp[iph]),0]
    for i in range(3):
      for j in range(3):
        Pij_thph[ith,iph,i,j]=Dij[i,j]-ni_thph[ith,iph,i]*ni_thph[ith,iph,j]
    for i in range(3):
      for j in range(3):
        for k in range(3):
          for l in range(3):
            Lamijkl_thph[ith,iph,i,j,k,l]=Pij_thph[ith,iph,i,k]*Pij_thph[ith,iph,j,l]-0.5*Pij_thph[ith,iph,i,j]*Pij_thph[ith,iph,k,l]

#Calculate the mass quadrupole tensor and its derivatives            
dMij = Mij
ddMij = Mij
dddMij = Mij
dhij_thph = hij_thph
for it in range(Nt):
  ph = Omg*tt[it]
  X1[it,:]=R*m2/M*np.array([np.cos(ph),np.sin(ph),0])
  V1[it,:]=R*m2/M*Omg*np.array([-np.sin(ph),np.cos(ph),0])
  X2[it,:]=-R*m1/M*np.array([np.cos(ph),np.sin(ph),0])
  V2[it,:]=-R*m1/M*Omg*np.array([-np.sin(ph),np.cos(ph),0])
  for i in range(3):
    for j in range(3):
      Mij[it,i,j]=m1*X1[it,i]*X1[it,j]+m2*X2[it,i]*X2[it,j]

for i in range(3):
  for j in range(3):
    dMij[:,i,j]=np.gradient(Mij[:,i,j],tt,edge_order=2)
    ddMij[:,i,j]=np.gradient(dMij[:,i,j],tt,edge_order=2)
    dddMij[:,i,j]=np.gradient(ddMij[:,i,j],tt,edge_order=2)
#Calculate the wave tensor by contracting the projection tensor with the multipole expansion
for ith in range(Nth):
  for iph in range(Nph):
    for i in range(3):
      for j in range(3):
        for k in range(3):
          for l in range(3):
            hij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*ddMij[:,k,l]
            dhij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*dddMij[:,k,l]

#Now project the h_ij wave tensor onto the h_{\hat{theta} \hat{phi}} basis
for ith in range(Nth):
  for iph in range(Nph):
    for i in range(3):
      for j in range(3):
        htp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        dhtp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
        dhtp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
        dhtp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
        dhtp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
    PP_thph[ith,iph] = np.mean(htp_thph[:,ith,iph,0,0]**2)
    PX_thph[ith,iph] = np.mean(htp_thph[:,ith,iph,0,1]**2)


plt.plot(tt,dddMij[:,0,0])
plt.plot(tt,dddMij[:,0,1])
plt.plot(tt,dhij_thph[:,0,0,0,1])
#plt.plot(tt,hij_thph[:,20,0,1,1])
#plt.plot(tt,ddMij[:,0,1])
#plt.plot(tt,ddMij[:,2,2])
plt.show()

plt.plot(th,PP_thph[:,0])
plt.plot(th,PX_thph[:,0])
plt.show()
plt.plot(pp,PP_thph[10,:])
plt.plot(pp,PX_thph[10,:])
plt.show()

#Ptot_thph is the total radiated power as a function of theta and phi, averaged over a single orbit (should have no phi dependence)

Ptot_thph = PP_thph+PX_thph