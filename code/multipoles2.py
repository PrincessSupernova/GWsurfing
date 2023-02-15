#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:29:27 2022
latest version Feb 14 2023
includes mass octopole and current quadrupole terms
@author: jschnitt
"""

import numpy as np
import matplotlib.pyplot as plt
Nt = 202
Nth = 41
Nph = 30
m1 = 0.75
m2 = 0.25
M = m1+m2
R = 20
Torb = 2.*np.pi*np.sqrt(R**3./M)
Omg = 2.*np.pi/Torb
dt = 2.*Torb/(Nt)
tt = np.zeros(Nt)
for it in range(Nt):
  tt[it]=it*dt
X1 = np.zeros((Nt,3))
X2 = np.zeros((Nt,3))
V1 = np.zeros((Nt,3))
V2 = np.zeros((Nt,3))
Mij = np.zeros((Nt,3,3)) #mass quadrupole
Sijk = np.zeros((Nt,3,3,3)) #current quadrupole
Mijk = np.zeros((Nt,3,3,3)) #mass octopole
hij_thph = np.zeros((Nt,Nth,Nph,3,3))
hijS_thph = np.zeros((Nt,Nth,Nph,3,3)) #current quadrupole contribution to waveform
hijO_thph = np.zeros((Nt,Nth,Nph,3,3)) #octopole contribution to waveform
htp_thph = np.zeros((Nt,Nth,Nph,2,2))
htpS_thph = np.zeros((Nt,Nth,Nph,2,2))
htpO_thph = np.zeros((Nt,Nth,Nph,2,2))
dhtp_thph = np.zeros((Nt,Nth,Nph,2,2))
dhtpS_thph = np.zeros((Nt,Nth,Nph,2,2))
dhtpO_thph = np.zeros((Nt,Nth,Nph,2,2))
hP_thph = np.zeros((Nth,Nph))
hX_thph = np.zeros((Nth,Nph))
PP_thph = np.zeros((Nth,Nph))
PX_thph = np.zeros((Nth,Nph))
PPO_thph = np.zeros((Nth,Nph))
PXO_thph = np.zeros((Nth,Nph))
PPS_thph = np.zeros((Nth,Nph))
PXS_thph = np.zeros((Nth,Nph))
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
dMij = np.copy(Mij)      #this is the way to do it without sharing same pointer      
dMij = np.zeros((Nt,3,3))
ddMij = np.zeros((Nt,3,3))
dddMij = np.zeros((Nt,3,3))
tmp = np.zeros(Nt)
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
    tmp[:] = Mij[:,i,j]
    tmp=np.gradient(tmp,tt,edge_order=2)
    dMij[:,i,j]=tmp[:]
    tmp=np.gradient(tmp,tt,edge_order=2)
    ddMij[:,i,j]=tmp[:]
    tmp=np.gradient(tmp,tt,edge_order=2)
    dddMij[:,i,j]=tmp[:]

#Calculate the current quadrupole tensor and its derivatives            
dSijk = np.zeros((Nt,3,3,3))
ddSijk = np.zeros((Nt,3,3,3))
for it in range(Nt):
  for i in range(3):
    for j in range(3):
      for k in range(3):
        Sijk[it,i,j,k]=(m1*(X1[it,i]*X1[it,k]*V1[it,j]
                           +X1[it,j]*X1[it,k]*V1[it,i]
                           -2.*X1[it,i]*X1[it,j]*V1[it,k])
                        +m2*(X2[it,i]*X2[it,k]*V2[it,j]
                             +X2[it,j]*X2[it,k]*V2[it,i]
                             -2.*X2[it,i]*X2[it,j]*V2[it,k]))


for i in range(3):
  for j in range(3):
    for k in range(3):
      tmp[:]=Sijk[:,i,j,k]
      tmp=np.gradient(tmp,tt,edge_order=2)
      dSijk[:,i,j,k]=tmp[:]
      tmp=np.gradient(tmp,tt,edge_order=2)
      ddSijk[:,i,j,k]=tmp[:]

#Calculate the mass octopole tensor and its derivatives            
dMijk = np.zeros((Nt,3,3,3))
ddMijk = np.zeros((Nt,3,3,3))
dddMijk = np.zeros((Nt,3,3,3))
for it in range(Nt):
  for i in range(3):
    for j in range(3):
      for k in range(3):
        Mijk[it,i,j,k]=m1*X1[it,i]*X1[it,j]*X1[it,k]+m2*X2[it,i]*X2[it,j]*X2[it,k]

for i in range(3):
  for j in range(3):
    for k in range(3):
      tmp[:]=Mijk[:,i,j,k]
      tmp=np.gradient(tmp,tt,edge_order=2)
      dMijk[:,i,j,k]=tmp[:]
      tmp=np.gradient(tmp,tt,edge_order=2)
      ddMijk[:,i,j,k]=tmp[:]
      tmp=np.gradient(tmp,tt,edge_order=2)
      dddMijk[:,i,j,k]=tmp[:]

#Calculate the wave tensor by contracting the projection tensor with the multipole expansion
for ith in range(Nth):
  for iph in range(Nph):
    for i in range(3):
      for j in range(3):
        for k in range(3):
          for l in range(3):
            hij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*ddMij[:,k,l]
            dhij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*dddMij[:,k,l]
            for m in range(3):
              hijO_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*dddMijk[:,k,l,m]*ni_thph[ith,iph,m]/3.
              hijS_thph[:,ith,iph,i,j] += 2.*Lamijkl_thph[ith,iph,i,j,k,l]*ddSijk[:,k,l,m]*ni_thph[ith,iph,m]/3.
              

#Now project the h_ij wave tensor onto the h_{\hat{theta} \hat{phi}} basis
for ith in range(Nth):
  for iph in range(Nph):
    for i in range(3):
      for j in range(3):
        htp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
        htpO_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
        htpO_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
        htpO_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
        htpO_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
        htpS_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
        htpS_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
        htpS_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
        htpS_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
#        dhtp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#        dhtp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#        dhtp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#        dhtp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#calculate the power [hdot^2] as a function of theta, phi, averaging over an orbit
#by only including middle segment, reduce edge effects of np.gradient
    for i in range(2):
      for j in range(2):
        tmp[:] = htp_thph[:,ith,iph,i,j]
        dhtp_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=2)
        tmp[:] = htpO_thph[:,ith,iph,i,j]
        dhtpO_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=2)
        tmp[:] = htpS_thph[:,ith,iph,i,j]
        dhtpS_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=2)
    PP_thph[ith,iph] = np.mean(dhtp_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2)
    PX_thph[ith,iph] = np.mean(dhtp_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
    PPO_thph[ith,iph] = np.mean(dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2)
    PXO_thph[ith,iph] = np.mean(dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
    PPS_thph[ith,iph] = np.mean(dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2)
    PXS_thph[ith,iph] = np.mean(dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)


#plt.plot(tt,ddMij[:,0,0])
#plt.plot(tt,dddMijk[:,0,1,0])
#plt.plot(tt,dhij_thph[:,0,0,0,1])
plt.plot(tt,hij_thph[:,10,0,0,0])
plt.plot(tt,hijO_thph[:,10,0,0,0])
plt.plot(tt,hijS_thph[:,10,0,0,0])
#plt.plot(tt,ddMij[:,0,1])
#plt.plot(tt,ddMij[:,0,0])
plt.show()

plt.plot(tt,dhtp_thph[:,10,0,0,0])
plt.plot(tt,dhtp_thph[:,10,0,0,1])
plt.plot(tt,dhtpO_thph[:,10,0,0,0])
plt.plot(tt,dhtpO_thph[:,10,0,0,1])
plt.show()

#plot the power as a function of theta for the two polarizations
plt.plot(np.cos(th),PP_thph[:,0])
plt.plot(np.cos(th),PX_thph[:,0])
plt.show()
plt.plot(np.cos(th),PPO_thph[:,0])
plt.plot(np.cos(th),PXO_thph[:,0])
plt.show()
plt.plot(np.cos(th),PPS_thph[:,0])
plt.plot(np.cos(th),PXS_thph[:,0])
plt.show()

#analytic solution of P(theta) for quadrupole formula
#plt.plot(np.cos(th),4.*(1.+np.cos(th)**2)**2)
#plt.plot(np.cos(th),16.*np.cos(th)**2)
#plt.show()
