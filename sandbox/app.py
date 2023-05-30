import dash
from dash import html,dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def Header(name, app):
    title = html.H1(name)
    logo = html.Img(
        src=app.get_asset_url("dash-logo-new.png"), style={"float": "right", "height": 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")

    return dbc.Row([dbc.Col(title, width=8), dbc.Col(link, width=4)], align="center")


app = dash.Dash(__name__,
           external_stylesheets=[dbc.themes.VAPOR]
      )

app.layout = html.Div([
    # banner 
    html.Div(className="banner",
             children=[
               html.Div(className="container scalable",
                 children=[
                   html.Br(),
                   Header("GW Surfing", app),
                   html.Br()
             ])]),
    # body
    html.Div(id="body",className="container scalable",
             children=[
               html.Div(
                 children=[
                   dbc.Row([
                     # controls
                     dbc.Col(
                       html.Div([
                         html.H4("Controls"),
                         html.Br(),
                         html.P("Binary separation:"),
                         dcc.Slider(
                           id="slider-sma",
                           min=1.,
                           max=200.,
#                           step=20.,
                           value=20.,
                           tooltip={"placement": "bottom", "always_visible": True}),
                         html.Br(),
                         html.P("Number of binaries:"),
                         dcc.Slider(
                           id="slider-Nb",
                           min=2,
                           max=6,
                           value=4,
                           step=1,
                           tooltip={"placement": "bottom", "always_visible": True}),
                         html.Br(),
                         html.P("Distance between binaries [GW wavelengths]"),
                         dcc.Slider(
                           id="slider-db",
                           min=0.25,
                           max=1.,
                           step=0.25,
                           value=0.25,
                           tooltip={"placement": "bottom", "always_visible": True}),
                         html.Br(),
                         html.P("Initial phase lag between binaries [\u03c0]"),
                         dcc.Slider(
                           id="slider-pb",
                           min=0,
                           max=1.,
                           step=0.25,
                           value=0.25,
                           tooltip={"placement": "bottom", "always_visible": True}),
                         html.Br(),
                         html.P("Eccentricity: (Coming soon!)"),
                         dcc.Slider(
                           id="slider-ecc",
                           min=0,
                           max=1.,
                           #step=10.,
                           value=0.,
                           tooltip={"placement": "bottom", "always_visible": True}),
                         html.Br(),
                         html.P("Average power over one orbital period?"),
                         dcc.RadioItems(
                           options=[
                             {'label': 'Yes', 'value': True},
                             {'label': 'No', 'value': False}
                             ],
                           value=False,
                           id="radio-time-avg", 
                           inline=True),
                         html.Br(),
                       ]),width=4),
                     # graph
                     dbc.Col(
                       html.Div(id="div-graph",
                         children=dcc.Graph(
                                    id="graph",
                                    style={
                                      'height': 600
                                    }
                                  ),
                       )
                     ),
                   ])
             ])]),
])

@app.callback(
    Output("graph", "figure"),
    Input("slider-sma", "value"),
    Input("slider-Nb","value"),
    Input("slider-db", "value"),
    Input("slider-pb", "value"),
    Input("slider-ecc", "value"),
    Input("radio-time-avg", "value"),
)


def make_plots(sma,Nb,db,pb,ecc,timeavg):
    fig = produce_figures(sma,Nb,db,pb,ecc=ecc,timeavg=timeavg)
    return fig


def produce_figures(sma,Nb,db,pb,ecc=0.,timeavg=False):

    ## test the timeavg boolean works
    if timeavg:
        print("Time averaging")
    else:
        print("Not time averaging")
    ##### end timeavg boolean test

    # time elements
    Nt = 101
    Nt2 = Nt+20
    # grid points for theta, phi
    Nth = 41
    Nph = 30
    # number of particles
    Np = int(Nb*2) 
    # particle masses
    Mp = np.ones(Np)*0.5
    R = sma
    R_b = np.ones(Nb)*R
    Torb_b = np.zeros(Nb)
    Omg_b = np.zeros(Nb)
    for ib in range(Nb):
        Mtot = Mp[ib*2]+Mp[ib*2+1]
        Torb_b[ib] = 2.*np.pi*np.sqrt(R_b[ib]**3./Mtot)
        Omg_b[ib] = 2.*np.pi/Torb_b[ib]
    M = 1.0
    Torb = 2.*np.pi*np.sqrt(R**3./M)
    Omg = 2.*np.pi/Torb

    L = Torb/2. #GW wavelength

    z_b = np.linspace(1,Nb,Nb)*L*db
    #z_b = [0,1/4*L,2/4*L,3/4*L] # relative distance b/w binaries
    #z_b = [0,5/4*L,10/4*L,15/4*L] # relative distance b/w binaries

    phi_b = np.linspace(0,Nb-1,Nb)*(-np.pi*pb)
    #[0*np.pi,-1/4*np.pi,-2/4*np.pi,-3/4*np.pi]
 
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
    #Sbijk = np.zeros((Nt2,Nb,3,3,3)) #current quadrupole
    #Mbijk = np.zeros((Nt2,Nb,3,3,3)) #mass octopole
    hij_thph = np.zeros((Nt,Nth,Nph,3,3))
    #hijS_thph = np.zeros((Nt,Nth,Nph,3,3)) #current quadrupole contribution to waveform
    #hijO_thph = np.zeros((Nt,Nth,Nph,3,3)) #octopole contribution to waveform
    htp_thph = np.zeros((Nt,Nth,Nph,2,2))
    #htpS_thph = np.zeros((Nt,Nth,Nph,2,2))
    #htpO_thph = np.zeros((Nt,Nth,Nph,2,2))
    dhtp_thph = np.zeros((Nt,Nth,Nph,2,2))
    #dhtpS_thph = np.zeros((Nt,Nth,Nph,2,2))
    #dhtpO_thph = np.zeros((Nt,Nth,Nph,2,2))
    hP_thph = np.zeros((Nth,Nph))
    hX_thph = np.zeros((Nth,Nph))
    PP_thph = np.zeros((Nth,Nph))
    PX_thph = np.zeros((Nth,Nph))
    Ptot_thph = np.zeros((Nt,Nth,Nph))
    #Ptot_noavg_thph = np.zeros((Nt,Nth,Nph))
    #PPO_thph = np.zeros((Nth,Nph))
    #PXO_thph = np.zeros((Nth,Nph))
    #PtotO_thph = np.zeros((Nth,Nph))
    #PPS_thph = np.zeros((Nth,Nph))
    #PXS_thph = np.zeros((Nth,Nph))
    #PtotS_thph = np.zeros((Nth,Nph))
    
    Dij = np.zeros((3,3)) #Kronecker delta tensor
    for i in range(3):
        Dij[i,i]=1.

    theta = np.linspace(0,np.pi,Nth)
    phi = np.linspace(0,2.*np.pi,Nph)
   
    ni_thph = np.zeros((Nth,Nph,3))
    Lthi_thph = np.zeros((Nth,Nph,3))
    Lphi_thph = np.zeros((Nth,Nph,3))
    Pij_thph = np.zeros((Nth,Nph,3,3))
    Lamijkl_thph = np.zeros((Nth,Nph,3,3,3,3))

    #Calculate the Lambda projection tensor over the sphere of (z,phi)
    for ith in range(Nth):
        costheta = np.cos(theta[ith])
        sintheta = np.sin(theta[ith])
        for iph in range(Nph):
            cosphi = np.cos(phi[iph])
            sinphi = np.sin(phi[iph])

            ni_thph[ith,iph,:]=[sintheta*cosphi,sintheta*sinphi,costheta]
            Lthi_thph[ith,iph,:]=[costheta*cosphi,costheta*sinphi,-sintheta]
            Lphi_thph[ith,iph,:]=[-sinphi,cosphi,0]
 
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
            ### SG TO-DO -- do we need to change z component here to match z_b
            ph = Omg_b[ib]*tt2[it]+phi_b[ib]
            Xp[it,ib*2+0,:]=R_b[ib]*Mp[ib*2+1]/M*np.array([np.cos(ph),np.sin(ph),z_b[ib]])
            Xp[it,ib*2+1,:]=-R_b[ib]*Mp[ib*2+0]/M*np.array([np.cos(ph),np.sin(ph),-z_b[ib]])
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

#    #Calculate the current quadrupole tensor and its derivatives            
#    #THESE CAN PROBABLY BE SPED UP WITH TENSOR SYMMETRIES
#    dSbijk = np.zeros((Nt2,Nb,3,3,3))
#    ddSbijk = np.zeros((Nt2,Nb,3,3,3))
#    for ib in range(Nb):
#        for ip in range(2):
#            for i in range(3):
#                for j in range(3):
#                    for k in range(3):
#                        Sbijk[:,ib,i,j,k]+=Mp[ib*2+ip]*(Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,k]*Vp[:,ib*2+ip,j]
#                                                       +Xp[:,ib*2+ip,j]*Xp[:,ib*2+ip,k]*Vp[:,ib*2+ip,i]
#                                                       -2.*Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,j]*Vp[:,ib*2+ip,k])
#
#    for ib in range(Nb):
#        for i in range(3):
#            for j in range(3):
#                for k in range(3):
#                    tmp[:]=Sbijk[:,ib,i,j,k]
#                    tmp=np.gradient(tmp,tt2,edge_order=2)
#                    dSbijk[:,ib,i,j,k]=tmp[:]
#                    tmp=np.gradient(tmp,tt2,edge_order=2)
#                    ddSbijk[:,ib,i,j,k]=tmp[:]
#
#    #Calculate the mass octopole tensor and its derivatives            
#    dMbijk = np.zeros((Nt2,Nb,3,3,3))
#    ddMbijk = np.zeros((Nt2,Nb,3,3,3))
#    dddMbijk = np.zeros((Nt2,Nb,3,3,3))
#    for ib in range(Nb):
#        for ip in range(2):
#            for i in range(3):
#                for j in range(3):
#                    for k in range(3):
#                        Mbijk[:,ib,i,j,k]+=Mp[ib*2+ip]*Xp[:,ib*2+ip,i]*Xp[:,ib*2+ip,j]*Xp[:,ib*2+ip,k]
#
#    for ib in range(Nb):
#        for i in range(3):
#            for j in range(3):
#                for k in range(3):
#                    tmp[:]=Mbijk[:,ib,i,j,k]
#                    tmp=np.gradient(tmp,tt2,edge_order=2)
#                    dMbijk[:,ib,i,j,k]=tmp[:]
#                    tmp=np.gradient(tmp,tt2,edge_order=2)
#                    ddMbijk[:,ib,i,j,k]=tmp[:]
#                    tmp=np.gradient(tmp,tt2,edge_order=2)
#                    dddMbijk[:,ib,i,j,k]=tmp[:]

    #cut off the "ghost cells" for each of the multipole tensors, so the total
    #time covered is exactly one period
    ddMbij = ddMbij[10:Nt+10]
#    ddSbijk = ddSbijk[10:Nt+10]
#    dddMbijk = dddMbijk[10:Nt+10]
    tmp = np.zeros(Nt)

    #Calculate the sum of the Nb multipole tensors, accounting for t_ret(theta)
    #Let t_ret to the origin be 0
    ddMij_th = np.zeros((Nt,Nth,3,3))
#    ddSijk_th = np.zeros((Nt,Nth,3,3,3))
#    dddMijk_th = np.zeros((Nt,Nth,3,3,3))
    idex0 = np.array(range(Nt))
    for ith in range(Nth):
        costheta = np.cos(theta[ith])
        for ib in range(Nb):
            t_ret = z_b[ib]*costheta
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
#            dddMijk_th[:,ith,:,:,:]+=w_lo*dddMbijk[idex_lo,ib,:,:,:]+w_hi*dddMbijk[idex_hi,ib,:,:,:]
#            ddSijk_th[:,ith,:,:,:]+=w_lo*ddSbijk[idex_lo,ib,:,:,:]+w_hi*ddSbijk[idex_hi,ib,:,:,:]

    #Calculate the wave tensor by contracting the projection tensor with the multipole expansion
    for ith in range(Nth):
        for iph in range(Nph):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            hij_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*ddMij_th[:,ith,k,l]
#                            for m in range(3):
#                                hijO_thph[:,ith,iph,i,j] += Lamijkl_thph[ith,iph,i,j,k,l]*dddMijk_th[:,ith,k,l,m]*ni_thph[ith,iph,m]/3.
#                                hijS_thph[:,ith,iph,i,j] += 2.*Lamijkl_thph[ith,iph,i,j,k,l]*ddSijk_th[:,ith,k,l,m]*ni_thph[ith,iph,m]/3.
                 

    #Now project the h_ij wave tensor onto the h_{\hat{theta} \hat{phi}} basis
    for ith in range(Nth):
        for iph in range(Nph):
            for i in range(3):
                for j in range(3):
                    htp_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
                    htp_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
                    htp_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
                    htp_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hij_thph[:,ith,iph,i,j]
                    #htpO_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
                    #htpO_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
                    #htpO_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
                    #htpO_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijO_thph[:,ith,iph,i,j]
                    #htpS_thph[:,ith,iph,0,0] += Lthi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
                    #htpS_thph[:,ith,iph,1,0] += Lphi_thph[ith,iph,i]*Lthi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
                    #htpS_thph[:,ith,iph,0,1] += Lthi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
                    #htpS_thph[:,ith,iph,1,1] += Lphi_thph[ith,iph,i]*Lphi_thph[ith,iph,j]*hijS_thph[:,ith,iph,i,j]
           
    ##ccaallculate the power [hdot^2] as a function of theta, phi, averaging over an orbit
    ##bbyy  only including middle segment, reduce edge effects of np.gradient
            for i in range(2):
                for j in range(2):
                    tmp[:] = htp_thph[:,ith,iph,i,j]
                    dhtp_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=1)
                    #tmp[:] = htpO_thph[:,ith,iph,i,j]
                    #dhtpO_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=1)
                    #tmp[:] = htpS_thph[:,ith,iph,i,j]
                    #dhtpS_thph[:,ith,iph,i,j]=np.gradient(tmp,tt,edge_order=1)
            PP_thph[ith,iph] = np.mean(dhtp_thph[:,ith,iph,0,0]**2)
            PX_thph[ith,iph] = np.mean(dhtp_thph[:,ith,iph,0,1]**2)
            #PPO_thph[ith,iph] = np.mean(dhtpO_thph[:,ith,iph,0,0]**2)
            #PXO_thph[ith,iph] = np.mean(dhtpO_thph[:,ith,iph,0,1]**2)
            #PPS_thph[ith,iph] = np.mean(dhtpS_thph[:,ith,iph,0,0]**2)
            #PXS_thph[ith,iph] = np.mean(dhtpS_thph[:,ith,iph,0,1]**2)

            #### SG: time average the total power if needed
            if timeavg:
                Ptot_thph[:,ith,iph] = np.mean(dhtp_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2 + dhtp_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
                ## total power in octupole moment
                #PtotO_thph[ith,iph] = np.mean(dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2 + dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
                ## total power in current quadrupole moment
                #PtotS_thph[ith,iph] = np.mean(dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2 + dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)

            else: 
                # SG add -- total power in quadrupole moment, not time averaged
                Ptot_thph[:,ith,iph] = dhtp_thph[:,ith,iph,0,0]**2 + dhtp_thph[:,ith,iph,0,1]**2

    # Get polarization angle psi
    psiQ = np.zeros((Nt,Nth,Nph))
    #psiS = np.zeros((Nt,Nth,Nph))
    #psiO = np.zeros((Nt,Nth,Nph))

    for jth in range(0,Nth):
        for kph in range(0,Nph):
            psiQ[:,jth,kph] = np.arctan2(htp_thph[:,jth,kph,0,1],htp_thph[:,jth,kph,0,0])
            #psiS[:,jth,kph] = np.arctan2(htpS_thph[:,jth,kph,0,1],htpS_thph[:,jth,kph,0,0])
            #psiO[:,jth,kph] = np.arctan2(htpO_thph[:,jth,kph,0,1],htpO_thph[:,jth,kph,0,0])
    # Take sine of polarisation angle (then transpose) for plotting purposes
    sinPsi = np.sin(psiQ).T

   
    # Meshgrid angles for plot stuff
    Theta,Phi = np.meshgrid(theta,phi)
    # Standard cartesian coords from spherical polars
    X,Y,Z = np.sin(Theta)*np.cos(Phi),np.sin(Theta)*np.sin(Phi),np.cos(Theta)
    ## Amplitude from quadrupole power
    ampQ = Ptot_thph.T

    Xt = np.zeros((Nph,Nth,Nt))
    Yt = np.zeros((Nph,Nth,Nt))
    Zt = np.zeros((Nph,Nth,Nt))

    for it in range(Nt):
        Xt[:,:,it] = ampQ[:,:,it]*X[:,:]
        Yt[:,:,it] = ampQ[:,:,it]*Y[:,:]
        Zt[:,:,it] = ampQ[:,:,it]*Z[:,:]


    # Now plot!
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'rowspan': 2, 'colspan': 1, 'is_3d': True}, 
                                {'rowspan': 2, 'colspan': 1, 'is_3d': True}],
                               [None, None]],
                        subplot_titles=['Binary set-up',
                                        'Mass quadrupole'],
                        vertical_spacing=0.13, horizontal_spacing=0.05)

    ### left panel -- binary set-up
    fig.add_trace(go.Scatter3d(x=Xp[0,:,0],
                               y=Xp[0,:,1],
                               z=Xp[0,:,2],
                               mode='markers',
                               marker=dict(color='black',size=5),
                               showlegend=False
                               ),1,1)
    ### add orbital paths
    for ib in range(0,Nb):
        fig.add_trace(go.Scatter3d(x=Xp[:,2*ib,0],
                                   y=Xp[:,2*ib,1],
                                   z=Xp[:,2*ib,2],
                                   mode='lines',
                                   marker=dict(color='black',size=1),
                                   showlegend=False
                                   ),1,1)

    ### right panel -- mass quadrupole
    fig.add_trace(go.Surface(x=Xt[:,:,0],y=Yt[:,:,0],z=Zt[:,:,0],
                             surfacecolor=sinPsi[:,:,0], 
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'}),1, 2)
 
    ### new time index number -- only show movie over Torb to avoid repetition
    Ntt = int((Nt+1)/2)

    frames = [dict(
               name = k,
               data = [go.Scatter3d(x=Xp[k,:,0],
                                    y=Xp[k,:,1],
                                    z=Xp[k,:,2],
                                    mode='markers',
                                    marker=dict(color='black',size=5),
                                    showlegend=False),
                       go.Surface(x=Xt[:,:,k],y=Yt[:,:,k],z=Zt[:,:,k],
                                  surfacecolor=sinPsi[:,:,k], 
                                  colorscale='PiYG',
                                  colorbar={"title": 'sin(\u03a8)'})
                      ],
               traces = [0,Nb+1]
               ) for k in range(Ntt)]


    updatemenus = [dict(type='buttons',
                        buttons=[dict(label='Play',
                                      method='animate',
                                      args=[[f'{k}' for k in range(Ntt)], 
                                              dict(frame=dict(duration=20), 
                                                   transition=dict(duration=0),
                                                   easing='linear',
                                                   fromcurrent=True,
                                                   mode='immediate')]),
                                 dict(label='Pause',
                                      method='animate',
                                      args=[[None], 
                                              dict(frame=dict(duration=0), 
                                                   transition=dict(duration=0),
                                                   mode='immediate')])

                        ],
                        direction= 'left', 
                        pad=dict(r= 10, t=85), 
                        showactive =True, x= 0.1, y= 0, xanchor= 'right', yanchor= 'top')]

    sliders = [{'yanchor': 'top',
                'xanchor': 'left', 
                'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                'transition': {'duration': 30.0, 'easing': 'linear'},
                'pad': {'b': 10, 't': 50}, 
                'len': 0.9, 'x': 0.1, 'y': 0, 
                'steps': [{'args': [[k], {'frame': {'duration': 30.0, 'easing': 'linear'},
                                      'transition': {'duration': 0, 'easing': 'linear'}}], 
                           'label': k, 'method': 'animate'} for k in range(Ntt)]
               }]

    fig.update(frames=frames)

    fig.update_layout(title_text="Orbit-averaged GW power",
                      updatemenus=updatemenus,
                      sliders=sliders)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
