# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
from io import BytesIO

import dash
from dash import dcc,html
from dash.dependencies import Input, Output, State
import numpy as np
from numpy import pi,sqrt,sin,cos,arccos
import matplotlib.pyplot as plt

import reusable_components as rc  # see reusable_components.py


# ############ Create helper functions ############
def mpl_to_b64(fig, format="png", dpi=300, **kwargs):
    b_io = BytesIO()
    fig.savefig(b_io, format=format, bbox_inches="tight", dpi=dpi, **kwargs)
    b64_enc = base64.b64encode(b_io.getvalue()).decode("utf-8")
    return f"data:image/{format};base64," + b64_enc


def build_visualizations(m1, m2, a):
    """
    sg - add this
    """

    # number of time points
    Nt = 202
    # number of points in theta
    Nth = 41
    # number of points in phi
    Nph = 30
    # total mass of the binary
    M = m1+m2
    # orbital period from Kepler's law
    Torb = 2.*pi*sqrt(a**3./M)
    #orbital angular frequency
    Omega = 2.*pi/Torb
    # time resolution -- integrating over one cycle
    # dt = orbital period / number of time points
    dt = 2.*Torb/Nt
    # time array
    tt = np.linspace(0,Nt-1,Nt)*dt

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
        ph = Omega*tt[it]
        X1[it,:]=a*m2/M*np.array([np.cos(ph),np.sin(ph),0])
        V1[it,:]=a*m2/M*Omega*np.array([-np.sin(ph),np.cos(ph),0])
        X2[it,:]=-a*m1/M*np.array([np.cos(ph),np.sin(ph),0])
        V2[it,:]=-a*m1/M*Omega*np.array([-np.sin(ph),np.cos(ph),0])
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
#                    dhtp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
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







################################### sg from nov 2022 version of multipoles    
#    # cos \Omega t, sin\Omega t
#    cosOmega_t = np.cos(tt*Omega)
#    sinOmega_t = np.sin(tt*Omega)
#   
#    # prefactors
#    prefac1 = a*m2/M
#    prefac2 = -a*m1/M
#
#
#    ## positions
#    # mass 1
#    X1 = np.zeros((Nt,3))
#    X1[:,0] = prefac1*cosOmega_t
#    X1[:,1] = prefac1*sinOmega_t
#    # mass 2
#    X2 = np.zeros((Nt,3))
#    X2[:,0] = prefac2*cosOmega_t
#    X2[:,1] = prefac2*sinOmega_t
#    ## velocities -- SG: are these actually used?
#    # mass 1
#    V1 = np.zeros((Nt,3))
#    V1[:,0] = -prefac1*Omega*sinOmega_t
#    V1[:,1] = prefac1*Omega*cosOmega_t
#    # mass 2
#    V2 = np.zeros((Nt,3))
#    V2[:,0] = -prefac2*Omega*sinOmega_t
#    V2[:,1] = prefac2*Omega*cosOmega_t
#    ## quadrupole moment
#    Mij = np.zeros((Nt,3,3))
#    for i in range(3):
#        for j in range(3):
#            Mij[:,i,j]=m1*X1[:,i]*X1[:,j]+m2*X2[:,i]*X2[:,j]
#
#
#    Sijk = np.zeros((Nt,3,3,3)) #current quadrupole
#    Mijk = np.zeros((Nt,3,3,3)) #mass octopole
#
#
#
#    #Calculate the mass quadrupole derivatives            
#    dotMij = np.copyMij
#    ddotMij = Mij
#    dddotMij = Mij
#    for i in range(3):
#        for j in range(3):
#            dotMij[:,i,j]=np.gradient(Mij[:,i,j],tt,edge_order=2)
#            ddotMij[:,i,j]=np.gradient(dotMij[:,i,j],tt,edge_order=2)
#            dddotMij[:,i,j]=np.gradient(ddotMij[:,i,j],tt,edge_order=2)
#
#
#
#    ## h and hdot in cartesian basis
#    hij_thph = np.zeros((Nt,Nth,Nph,3,3))
#    hijS_thph = np.zeros((Nt,Nth,Nph,3,3)) #current quadrupole contribution to waveform
#    hijO_thph = np.zeros((Nt,Nth,Nph,3,3)) #octopole contribution to waveform
#    dhij_thph = hij_thph
#    ##  h and hdot in theta, phi space
#    htp_thph = np.zeros((Nt,Nth,Nph,2,2))
#    dhtp_thph = np.zeros((Nt,Nth,Nph,2,2))
#    hP_thph = np.zeros((Nth,Nph))
#    hX_thph = np.zeros((Nth,Nph))
#    PP_thph = np.zeros((Nth,Nph))
#    PX_thph = np.zeros((Nth,Nph))
#    P_thph = np.zeros((Nth,Nph))
#
#    # kronecker delta
#    Dij = np.identity(3)
#    # theta distributed uniformly in cos theta
#    costheta = np.linspace(-1,1,Nth)
#    th = arccos(costheta)
#    sintheta = sqrt(1. - costheta**2)
#    # phi distributed uniformly in sin theta
#    pp = np.linspace(0,2.*np.pi,Nph)
#    cosphi = cos(pp)
#    sinphi = sin(pp)
#   
#    ######SG test
#    hp_thetaphi = np.zeros((Nt,Nth,Nph))
#    hx_thetaphi = np.zeros((Nt,Nth,Nph))
#   
#    for it in range(0,Nt):
#        for jth in range(0,Nth):
#            for kph in range(0,Nph):
#                hp_thetaphi[it,jth,kph] = -2*(1. + costheta[jth]**2)*cos(2.*Omega*tt[it] - pp[kph])
#                hx_thetaphi[it,jth,kph] = -4*costheta[jth]*sin(2.*Omega*tt[it] - pp[kph])
#   
#    hpdot_thetaphi = np.zeros((Nt,Nth,Nph))
#    hxdot_thetaphi = np.zeros((Nt,Nth,Nph))
#
#    for jth in range(0,Nth):
#        for kph in range(0,Nph):
#            hpdot_thetaphi[:,jth,kph] = np.gradient(hp_thetaphi[:,jth,kph],tt,edge_order=2)
#            hxdot_thetaphi[:,jth,kph] = np.gradient(hx_thetaphi[:,jth,kph],tt,edge_order=2)
#   
#    ###P = np.mean(hpdot_thetaphi**2 + hxdot_thetaphi**2)
#    PP = np.mean(hpdot_thetaphi**2)
#    ###############
#   
#    ni_thph = np.zeros((Nth,Nph,3))
#    Lthi_thph = np.zeros((Nth,Nph,3))
#    Lphi_thph = np.zeros((Nth,Nph,3))
#    Pij_thph = np.zeros((Nth,Nph,3,3))
#    Lamijkl_thph = np.zeros((Nth,Nph,3,3,3,3))
#
#    #Calculate the Lambda projection tensor over the sphere of (z,phi)
#    for ith in range(Nth):
#        for iph in range(Nph):
#            ni_thph[ith,iph,:]=[sintheta[ith]*cosphi[iph],sintheta[ith]*sinphi[iph],costheta[ith]]
#            Lthi_thph[ith,iph,:]=[costheta[ith]*cosphi[iph],costheta[ith]*sinphi[iph],-sintheta[ith]]
#            Lphi_thph[ith,iph,:]=[-sinphi[iph],cosphi[iph],0]
#   
#   
#            for i in range(3):
#                for j in range(3):
#                    Pij_thph[ith,iph,i,j]=Dij[i,j]-ni_thph[ith,iph,i]*ni_thph[ith,iph,j]
#            for i in range(3):
#                for j in range(3):
#                    for k in range(3):
#                        for l in range(3):
#                            Lamijkl_thph[ith,iph,i,j,k,l]=Pij_thph[ith,iph,i,k]*Pij_thph[ith,iph,j,l]-0.5*Pij_thph[ith,iph,i,j]*Pij_thph[ith,iph,k,l]
#
#    for ith in range(Nth):
#        for iph in range(Nph):
#            for i in range(3):
#                for j in range(3):
#                    for k in range(3):
#                        for l in range(3):
#                            hij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*ddotMij[:,k,l]
#                            dhij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*dddotMij[:,k,l]
#
#    #Now project the h_ij wave tensor onto the h_{\hat{theta} \hat{phi}} basis
#    for ith in range(Nth):
#        for iph in range(Nph):
#            for i in range(3):
#                for j in range(3):
#                    htp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
#                    htp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
#                    htp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
#                    htp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#                    dhtp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*dhij_thph[:,ith,iph,i,j]
#            PP_thph[ith,iph] = np.mean(htp_thph[:,ith,iph,0,0]**2)
#            PX_thph[ith,iph] = np.mean(htp_thph[:,ith,iph,0,1]**2)
#            ## seg add
#            P_thph[ith,iph] = np.mean(htp_thph[:,ith,iph,0,0]**2+htp_thph[:,ith,iph,0,1]**2)



    ##### end of code from multipoles-orig.py
    
#    # Now make plots
#    fig, ax = plt.subplots()
#    #wfct.visualization.visualize_cut_plane(
#    #    fi.get_hor_plane(), ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
#    #)
#    ax.axhline(m1, color="k", ls="--", lw=1)
#    ax.axvline(a, color="k", ls="--", lw=1)
#    horiz_b64 = mpl_to_b64(fig)
#    plt.close(fig)

    # hij plots
    fig, ax = plt.subplots()
    ax.plot(tt,hij_thph[:,10,0,0,0],label=r"hij_thph")
    ax.plot(tt,hijO_thph[:,10,0,0,0],label=r"hij0_thph")
    ax.plot(tt,hijS_thph[:,10,0,0,0],label=r"hijS_thph")
    ax.legend()
    hij_plot = mpl_to_b64(fig)
    plt.close(fig)

    # dhtp plots
    fig, ax = plt.subplots() 
    ax.plot(tt,dhtp_thph[:,10,0,0,0],label=r"dhtp_thph[:,10,0,0,0]")
    ax.plot(tt,dhtp_thph[:,10,0,0,1],label=r"dhtp_thph[:,10,0,0,1]")
    ax.plot(tt,dhtpO_thph[:,10,0,0,0],label=r"dhtp0_thph[:,10,0,0,0]")
    ax.plot(tt,dhtpO_thph[:,10,0,0,1],label=r"dhtp0_thph[:,10,0,0,1]")
    ax.legend()
    dhtp_plot = mpl_to_b64(fig)
    plt.close(fig)

    # plot the power as a function of theta for the two polarizations
    fig, ax = plt.subplots()
    ax.plot(np.cos(th),PP_thph[:,0],label=r"PP_thph")
    ax.plot(np.cos(th),PX_thph[:,0],label=r"PX_thph")
    ax.legend()
    P_plot = mpl_to_b64(fig)
    plt.close(fig)

    # plot the power as a function of theta for the two polarizations
    fig, ax = plt.subplots()
    ax.plot(np.cos(th),PPO_thph[:,0],label=r"PPO_thph")
    ax.plot(np.cos(th),PXO_thph[:,0],label=r"PXO_thph")
    ax.legend()
    P0_plot = mpl_to_b64(fig)
    plt.close(fig)

    # plot the power as a function of theta for the two polarizations
    fig, ax = plt.subplots()
    ax.plot(np.cos(th),PPS_thph[:,0],label=r"PPS_thph")
    ax.plot(np.cos(th),PXS_thph[:,0],label=r"PXS_thph")
    ax.legend()
    PS_plot = mpl_to_b64(fig)
    plt.close(fig)

    return hij_plot, dhtp_plot, P_plot, P0_plot, PS_plot #horiz_b64, x_plane_b64, y_plane_b64


# ############ Initialize app ############
app = dash.Dash(__name__, external_stylesheets=[rc.MATERALIZE_CSS])#, prevent_initial_callbacks=True)
server = app.server


# ############ Build components and layouts ############
navbar = html.Nav(
    html.Div(
        className="nav-wrapper teal",
        children=[
            html.Img(
                src=app.get_asset_url("dash-logo.png"),
                style={"float": "right", "height": "100%", "padding-right": "15px"},
            ),
            html.A(
                "GW Surfing",
                className="brand-logo",
                href="https://plotly.com/dash/",
                style={"padding-left": "15px"},
            ),
        ],
    )
)

controls = [
    rc.CustomSlider(id="m1", min=0.1, max=1, value=0.75, label=r"Primary mass [$M_{\odot}$]"),
    rc.CustomSlider(id="m2", min=0.1, max=1, value=0.25, label=r"Secondary mass [$M_{\odot}$]"),
    rc.CustomSlider(id="a", min=10, max=200, value=20, label="Binary separation"),
    #rc.CustomSlider(id="y-loc", min=0, max=1, label="Number of binaries"),
]

left_section = rc.Card(
    rc.CardContent(
        [
            rc.CardTitle("hij plot"),
            html.Img(id="hij_plot", style={"width": "100%"}),
            rc.CardTitle("dhtp plot"),
            html.Img(id="dhtp_plot", style={"width": "100%"}),
            rc.CardTitle("P plot"),
            html.Img(id="P_plot", style={"width": "100%"}),
            rc.CardTitle("PO plot"),
            html.Img(id="PO_plot", style={"width": "100%"}),
            rc.CardTitle("PS plot"),
            html.Img(id="PS_plot", style={"width": "100%"}),
        ]
    )
)

right_section = rc.Card(
    rc.CardContent(
        [
            #rc.CardTitle("Horizontal Cut Plane"),
            #html.Img(id="no-gch-horizontal", style={"width": "100%"}),
            #rc.CardTitle("dhtp plot"),
            #html.Img(id="dhtp_plot", style={"width": "100%"}),
            #rc.CardTitle("Cross (Y-Normal) Cut Plane"),
            #html.Img(id="no-gch-y-normal", style={"width": "100%"}),
        ]
    )
)

app.layout = html.Div(
    style={"--slider_active": "teal"},
    # className="container",
    children=[
        navbar,
        html.Br(),
        rc.Row(
            rc.Col(
                rc.Card(rc.CardContent(rc.Row([rc.Col(c, width=3) for c in controls]))),
                width=12,
            )
        ),
        rc.Row(
            [
                rc.Col([html.H4("hij plot"), left_section], width=6),
                rc.Col([html.H4("to add: surface projection"), right_section], width=6),
            ]
        ),
    ],
)


@app.callback(
    Output("hij_plot", "src"),
    Output("dhtp_plot", "src"),
    Output("P_plot", "src"),
    Output("PO_plot", "src"),
    Output("PS_plot", "src"),
    Input("m1", "value"),
    Input("m2", "value"),
    Input("a", "value"),
)

def gch_update(m1, m2, a):
    return build_visualizations(m1, m2, a)


#@app.callback(
#    #Output("no-gch-horizontal", "src"),
#    Output("no-gch-x-normal", "src"),
#    #Output("no-gch-y-normal", "src"),
#    Input("m1", "value"),
#    Input("m2", "value"),
#    Input("a", "value"),
#)
#def no_gch_update(m1, m2, a):
#    return build_visualizations(m1, m2, a)


if __name__ == "__main__":
    app.run_server(debug=True, threaded=False)#, processes=2)