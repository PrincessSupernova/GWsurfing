#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:29:27 2022

@author: jschnitt
Updated April 17 2023 to include N binaries (or other systems)
"""

import numpy as np
import matplotlib.pyplot as plt
Nt = 101
Nt2 = Nt+20
Nth = 41
Nph = 30
Np = 8 #Number of particles
Nb = 4 #Number of binaries
#MP = [0.5,0.5]
#Mp = [0.5,0.5,0.5,0.5]
Mp = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
R = 20.0
#R_b = [R]
#R_b = [R,R]
R_b = [R,R,R,R]
Torb_b = np.zeros(Nb)
Omg_b = np.zeros(Nb)
for ib in range(Nb):
  Mtot = Mp[ib*2]+Mp[ib*2+1]
  Torb_b[ib] = 2.*np.pi*np.sqrt(R_b[ib]**3./Mtot)
  Omg_b[ib] = 2.*np.pi/Torb_b[ib]
M = 1.0
Torb = 2.*np.pi*np.sqrt(R**3./M)
Omg = 2.*np.pi/Torb
L = Torb/2. #distance between binaries
#z_b = [0]  #z-offset for each binary
#z_b = [-3*L/8,3*L/8]
#z_b = [-9*L/8,-3*L/8,3*L/8,9*L/8]
z_b = [0,1/4*L,2/4*L,3/4*L]
#phi_b = [np.pi/4.] #phase for each binary
#phi_b = [0.0,-np.pi/4.]
phi_b = [0*np.pi,-1/4*np.pi,-2/4*np.pi,-3/4*np.pi]

dt = Torb/(Nt-1)
tt = np.zeros(Nt)
for it in range(Nt):
  tt[it]=it*dt
tt2 = np.zeros(Nt2)
for it in range(Nt2):
  tt2[it]=(it-10)*dt

Xp = np.zeros((Nt2,Np,3))
Vp = np.zeros((Nt2,Np,3))
Mbij = np.zeros((Nt2,Nb,3,3)) #mass quadrupole
Sbijk = np.zeros((Nt2,Nb,3,3,3)) #current quadrupole
Mbijk = np.zeros((Nt2,Nb,3,3,3)) #mass octopole
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
Dij = np.zeros((3,3)) #Kronecker delta tensor
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

#Calculate the mass quadrupole tensor and its derivatives for each binary, as if it were centered on the origin
dMbij = np.zeros((Nt2,Nb,3,3))
ddMbij = np.zeros((Nt2,Nb,3,3))
tmp = np.zeros(Nt2)
for ib in range(Nb):
  for it in range(Nt2):
    ph = Omg_b[ib]*tt2[it]+phi_b[ib]
    Xp[it,ib*2+0,:]=R_b[ib]*Mp[ib*2+1]/M*np.array([np.cos(ph),np.sin(ph),0])
    Xp[it,ib*2+1,:]=-R_b[ib]*Mp[ib*2+0]/M*np.array([np.cos(ph),np.sin(ph),0])
    Vp[it,ib*2+0,:]=R_b[ib]*Mp[ib*2+1]/M*Omg_b[ib]*np.array([-np.sin(ph),np.cos(ph),0])
    Vp[it,ib*2+1,:]=-R_b[ib]*Mp[ib*2+0]/M*Omg_b[ib]*np.array([-np.sin(ph),np.cos(ph),0])

for ib in range(Nb):   
  for ip in range(2):
    for i in range(3):  
      for j in range(3):
        Mbij[:,ib,i,j]+=Mp[ib*2+ip]*Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,j]

for ib in range(Nb):
  for i in range(3):
    for j in range(3):
      tmp[:] = Mbij[:,ib,i,j]
      tmp=np.gradient(tmp,tt2,edge_order=2)
      dMbij[:,ib,i,j]=tmp[:]
      tmp=np.gradient(tmp,tt2,edge_order=2)
      ddMbij[:,ib,i,j]=tmp[:]

#Calculate the current quadrupole tensor and its derivatives            
#THESE CAN PROBABLY BE SPEEDED UP WITH TENSOR SYMMETRIES
dSbijk = np.zeros((Nt2,Nb,3,3,3))
ddSbijk = np.zeros((Nt2,Nb,3,3,3))
for ib in range(Nb):
  for ip in range(2):
    for i in range(3):
      for j in range(3):
        for k in range(3):
          Sbijk[:,ib,i,j,k]+=Mp[ib*2+ip]*(Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,k]*Vp[:,ib*2+ip,j]
                                          +Xp[:,ib*2+ip,j]*Xp[:,ib*2+ip,k]*Vp[:,ib*2+ip,i]
                                          -2.*Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,j]*Vp[:,ib*2+ip,k])

for ib in range(Nb):
  for i in range(3):
    for j in range(3):
      for k in range(3):
        tmp[:]=Sbijk[:,ib,i,j,k]
        tmp=np.gradient(tmp,tt2,edge_order=2)
        dSbijk[:,ib,i,j,k]=tmp[:]
        tmp=np.gradient(tmp,tt2,edge_order=2)
        ddSbijk[:,ib,i,j,k]=tmp[:]

#Calculate the mass octopole tensor and its derivatives            
dMbijk = np.zeros((Nt2,Nb,3,3,3))
ddMbijk = np.zeros((Nt2,Nb,3,3,3))
dddMbijk = np.zeros((Nt2,Nb,3,3,3))
for ib in range(Nb):
  for ip in range(2):
    for i in range(3):
      for j in range(3):
        for k in range(3):
          Mbijk[:,ib,i,j,k]+=Mp[ib*2+ip]*Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,j]*Xp[:,ib*2+ip,k]

for ib in range(Nb):
  for i in range(3):
    for j in range(3):
      for k in range(3):
        tmp[:]=Mbijk[:,ib,i,j,k]
        tmp=np.gradient(tmp,tt2,edge_order=2)
        dMbijk[:,ib,i,j,k]=tmp[:]
        tmp=np.gradient(tmp,tt2,edge_order=2)
        ddMbijk[:,ib,i,j,k]=tmp[:]
        tmp=np.gradient(tmp,tt2,edge_order=2)
        dddMbijk[:,ib,i,j,k]=tmp[:]

#cut off the "ghost cells" for each of the multipole tensors, so the total
#time covered is exactly one period
ddMbij = ddMbij[10:Nt+10]
ddSbijk = ddSbijk[10:Nt+10]
dddMbijk = dddMbijk[10:Nt+10]
tmp = np.zeros(Nt)

#Calculate the sum of the Nb multipole tensors, accounting for t_ret(theta)
#Let t_ret to the origin be 0
ddMij_th = np.zeros((Nt,Nth,3,3))
ddSijk_th = np.zeros((Nt,Nth,3,3,3))
dddMijk_th = np.zeros((Nt,Nth,3,3,3))
idex0 = np.array(range(Nt))
for ith in range(Nth):
  for ib in range(Nb):
    t_ret = z_b[ib]*cth[ith]
    t_ret = t_ret % Torb_b[ib]
    it_lo = int(t_ret/Torb*(Nt-1))
    if (it_lo == Nt-1):
      it_lo = 0
      t_ret = 0
    it_hi = it_lo+1
    w_lo = (tt[it_hi]-t_ret)/(tt[it_hi]-tt[it_lo])
    w_hi = 1.-w_lo
    print(ith,t_ret,it_lo,it_hi,w_lo,w_hi)
    #mod (Nt-1) should ensure that Mij_th[0]=Mij_th[Nt]
    idex_lo = (idex0+it_lo)%(Nt-1)
    idex_hi = (idex0+it_hi)%(Nt-1)
    ddMij_th[:,ith,:,:]+=w_lo*ddMbij[idex_lo,ib,:,:]+w_hi*ddMbij[idex_hi,ib,:,:]
    dddMijk_th[:,ith,:,:,:]+=w_lo*dddMbijk[idex_lo,ib,:,:,:]+w_hi*dddMbijk[idex_hi,ib,:,:,:]
    ddSijk_th[:,ith,:,:,:]+=w_lo*ddSbijk[idex_lo,ib,:,:,:]+w_hi*ddSbijk[idex_hi,ib,:,:,:]

#Calculate the wave tensor by contracting the projection tensor with the multipole expansion
for ith in range(Nth):
  for iph in range(Nph):
    for i in range(3):
      for j in range(3):
        for k in range(3):
          for l in range(3):
            hij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*ddMij_th[:,ith,k,l]
            for m in range(3):
              hijO_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*dddMijk_th[:,ith,k,l,m]*ni_thph[ith,iph,m]/3.
              hijS_thph[:,ith,iph,i,j] += 2.*Lamijkl_thph[ith,iph,i,j,k,l]*ddSijk_th[:,ith,k,l,m]*ni_thph[ith,iph,m]/3.
              

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

#calculate the power [hdot^2] as a function of theta, phi, averaging over an orbit
#by only including middle segment, reduce edge effects of np.gradient
    for i in range(2):
      for j in range(2):
        tmp[:] = htp_thph[:,ith,iph,i,j]
        dhtp_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=1)
        tmp[:] = htpO_thph[:,ith,iph,i,j]
        dhtpO_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=1)
        tmp[:] = htpS_thph[:,ith,iph,i,j]
        dhtpS_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=1)
    PP_thph[ith,iph] = np.mean(dhtp_thph[:,ith,iph,0,0]**2)
    PX_thph[ith,iph] = np.mean(dhtp_thph[:,ith,iph,0,1]**2)
    PPO_thph[ith,iph] = np.mean(dhtpO_thph[:,ith,iph,0,0]**2)
    PXO_thph[ith,iph] = np.mean(dhtpO_thph[:,ith,iph,0,1]**2)
    PPS_thph[ith,iph] = np.mean(dhtpS_thph[:,ith,iph,0,0]**2)
    PXS_thph[ith,iph] = np.mean(dhtpS_thph[:,ith,iph,0,1]**2)
print('Total power: ',np.sum(PP_thph)+np.sum(PX_thph))
plt.plot(tt,hij_thph[:,40,0,0,0])
plt.plot(tt,hij_thph[:,35,0,0,0])
plt.plot(tt,hij_thph[:,5,0,0,0])
plt.plot(tt,hij_thph[:,1,0,0,0])
#plt.plot(tt,hijO_thph[:,10,0,0,0])
#plt.plot(tt,hijS_thph[:,10,0,0,0])
#plt.plot(tt,ddMij[:,0,1])
#plt.plot(tt,ddMij[:,0,0])
plt.show()

plt.plot(tt,dhtp_thph[:,20,0,0,0])
plt.plot(tt,dhtp_thph[:,20,0,0,1])
#plt.plot(tt,dhtpO_thph[:,10,0,0,0])
#plt.plot(tt,dhtpO_thph[:,10,0,0,1])
plt.show()

#plot the power as a function of theta for the two polarizations
plt.plot(np.cos(th),PP_thph[:,0])
plt.plot(np.cos(th),PX_thph[:,0])
plt.show()
#plt.plot(np.cos(th),PPO_thph[:,0])
#plt.plot(np.cos(th),PXO_thph[:,0])
#plt.show()
#plt.plot(np.cos(th),PPS_thph[:,0])
#plt.plot(np.cos(th),PXS_thph[:,0])
#plt.show()

#analytic solution of P(theta) for quadrupole formula
#plt.plot(np.cos(th),4.*(1.+np.cos(th)**2)**2)
#plt.plot(np.cos(th),16.*np.cos(th)**2)
#plt.show()
