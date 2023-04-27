# Simpler gist here: https://gist.github.com/xhlulu/773fe238773ea69c8bc2b26560ab67d7
import random

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
                         #html.P("Total mass [Msun]:"),
                         #dcc.Slider(
                         #  id="slider-mtot",
                         #  min=1.,
                         #  max=5.,
                         #  #step=0.05,
                         #  value=1.),
                         html.P("Mass ratio:"),
                         dcc.Slider(
                           id="slider-q",
                           min=0.001,
                           max=1.,
                           value=0.75,
#                           step=0.05,
                           tooltip={"placement": "bottom", "always_visible": True}),
                         html.Br(),
                         html.P("Semi-major axis [1e-8 Msun]:"),
                         dcc.Slider(
                           id="slider-sma",
                           min=1.,
                           max=200.,
#                           step=20.,
                           value=20.,
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
    Input("slider-q","value"),
    Input("slider-sma", "value"),
    Input("slider-ecc", "value"),
)


def make_plots(q,R,ecc):
    fig = produce_figures(q,R,ecc=ecc)
    return fig


def produce_figures(q,R,Mtot=1.,ecc=0.,Nt=101,Nth=51,Nph=50):

    # orbit params
    m1 = Mtot/(q+1)
    m2 = Mtot-m1
    Torb = 2.*np.pi*np.sqrt(R**3./Mtot)
    Omega = 2.*np.pi/Torb

    # time series
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
    Mijk = np.zeros((Nt,3,3,3)) #mass octupole
    hij_thph = np.zeros((Nt,Nth,Nph,3,3))
    hijS_thph = np.zeros((Nt,Nth,Nph,3,3)) #current quadrupole contribution to waveform
    hijO_thph = np.zeros((Nt,Nth,Nph,3,3)) #octupole contribution to waveform
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
    Ptot_thph = np.zeros((Nth,Nph))
    PPO_thph = np.zeros((Nth,Nph))
    PXO_thph = np.zeros((Nth,Nph))
    PtotO_thph = np.zeros((Nth,Nph))
    PPS_thph = np.zeros((Nth,Nph))
    PXS_thph = np.zeros((Nth,Nph))
    PtotS_thph = np.zeros((Nth,Nph))
    Dij = np.zeros((3,3))
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

    #Calculate the mass quadrupole tensor and its derivatives
    dMij = np.copy(Mij)      #this is the way to do it without sharing same pointer      
    dMij = np.zeros((Nt,3,3))
    ddMij = np.zeros((Nt,3,3))
    dddMij = np.zeros((Nt,3,3))
    tmp = np.zeros(Nt)
    dhij_thph = hij_thph
   
    R_M = R/Mtot 
    RW_M = R_M*Omega

    for it in range(Nt):
        Wt = Omega*tt[it]
        cosWt = np.cos(Wt)
        sinWt = np.sin(Wt)
        X1[it,:]=m2*R_M*np.array([cosWt,sinWt,0])
        V1[it,:]=m2*RW_M*np.array([-sinWt,cosWt,0])
        X2[it,:]=-R_M*m1*np.array([cosWt,sinWt,0])
        V2[it,:]=-RW_M*m1*np.array([-sinWt,cosWt,0])
   
    for i in range(3):
        for j in range(3):
            Mij[:,i,j]=m1*X1[:,i]*X1[:,j]+m2*X2[:,i]*X2[:,j]

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
      
    #Calculate the mass octupole tensor and its derivatives            
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
            # SG add -- total power in quadrupole moment
            Ptot_thph[ith,iph] = np.mean(dhtp_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2 + dhtp_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
            PPO_thph[ith,iph] = np.mean(dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2)
            PXO_thph[ith,iph] = np.mean(dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
            # SG add -- total power in octupole moment
            PtotO_thph[ith,iph] = np.mean(dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2 + dhtpO_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
            PPS_thph[ith,iph] = np.mean(dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2)
            PXS_thph[ith,iph] = np.mean(dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)
            # SG add -- total power in current quadrupole moment
            PtotS_thph[ith,iph] = np.mean(dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,0]**2 + dhtpS_thph[(tt>=0.5*Torb)&(tt<1.5*Torb),ith,iph,0,1]**2)

    #### SG test -- get psi
    ######SG test
    psiQ = np.zeros((Nt,Nth,Nph))
    psiS = np.zeros((Nt,Nth,Nph))
    psiO = np.zeros((Nt,Nth,Nph))

    for jth in range(0,Nth):
        for kph in range(0,Nph):
            psiQ[:,jth,kph] = np.arctan2(htp_thph[:,jth,kph,0,1],htp_thph[:,jth,kph,0,0])
            psiS[:,jth,kph] = np.arctan2(htpS_thph[:,jth,kph,0,1],htpS_thph[:,jth,kph,0,0])
            psiO[:,jth,kph] = np.arctan2(htpO_thph[:,jth,kph,0,1],htpO_thph[:,jth,kph,0,0])
   
    ### end of multipoles2.py code -- now add in app stuff
    Theta,Phi = np.meshgrid(theta,phi)
   
    X,Y,Z = np.sin(Theta)*np.cos(Phi),np.sin(Theta)*np.sin(Phi),np.cos(Theta)
   
    ## mass quadrupole
    ampQ = Ptot_thph.T
   
    ## current quadrupole
    ampS = PtotS_thph.T
   
    ## mass octupole
    ampO = PtotO_thph.T


    fig = make_subplots(rows=6, cols=3,
                        specs=[[{'rowspan': 6, 'colspan': 2, 'is_3d': True}, 
                                None,
                                {'rowspan': 3, 'colspan': 1, 'is_3d': True}],
                               [None, None, None],
                               [None, None, None],
                               [None, None, 
                                {'rowspan': 3, 'colspan': 1, 'is_3d': True}],\
                               [None, None, None],
                               [None, None, None]],
                        subplot_titles=['Mass quadrupole',
                                        'Current quadrupole',
                                        'Mass octopole'],
                        vertical_spacing=0.13, horizontal_spacing=0.05)


#    camp = 45e-9
#    xb,yb,zb = camp*np.cos(phi),camp*np.sin(phi),np.zeros(Nph)

    

   
    ### left panel -- mass quadrupole
    fig.add_trace(go.Surface(x=X*ampQ, y=Y*ampQ, z=Z*ampQ,
                             surfacecolor=np.sin(psiQ[0,:,:]).T, 
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'}),1, 1)
   
    ### top right panel -- current quadrupole
    fig.add_trace(go.Surface(x=X*ampS, y=Y*ampS, z=Z*ampS,
                             surfacecolor=np.sin(psiS[0,:,:]).T,
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'}),1, 3)
    
    ### bottom right panel -- mass octopole
    fig.add_trace(go.Surface(x=X*ampO, y=Y*ampO, z=Z*ampO,
                             surfacecolor=np.sin(psiO[0,:,:]).T,
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'}),4, 3)


    ### new time index number -- only show movie over Torb to avoid repetition
    Ntt = int((Nt+1)/2)

    frames = [dict(
               name = k,
               data = [go.Surface(x=X*ampQ, y=Y*ampQ, z=Z*ampQ,
                             surfacecolor=np.sin(psiQ[k,:,:]).T, 
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'}),
                       go.Surface(x=X*ampS, y=Y*ampS, z=Z*ampS,
                             surfacecolor=np.sin(psiS[k,:,:]).T,
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'}),
                       go.Surface(x=X*ampO, y=Y*ampO, z=Z*ampO,
                             surfacecolor=np.sin(psiO[k,:,:]).T,
                             colorscale='PiYG',
                             colorbar={"title": 'sin(\u03a8)'})],
               traces = [0, 1, 2]
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
                           'label': k, 'method': 'animate'} for k in range(Ntt)
                    ]
 
                #'steps': [{'args': [[k], {'frame': {'duration': 30.0, 'easing': 'linear'},
                #                      'transition': {'duration': 0, 'easing': 'linear'}}], 
                #           'label': "%.2fT\u2092\u1d63"%(k/Ntt), 'method': 'animate'} for k in [0,0.2*Ntt,0.4*Ntt,0.6*Ntt,0.8*Ntt,Ntt]
                #    ]
               }]

    fig.update(frames=frames)

    fig.update_layout(title_text="GW power in different modes",
                      updatemenus=updatemenus,
                      sliders=sliders
    #                  scene = dict(xaxis = dict(nticks=4,range=[-8e-8,8e-8],),
    #                               yaxis = dict(nticks=4, range=[-8e-8,8e-8],),
    #                               zaxis = dict(nticks=4, range=[-2e-7,2e-7],),),
    #                  scene2 = dict(xaxis = dict(nticks=4,range=[-8e-8,8e-8],),
    #                              yaxis = dict(nticks=4, range=[-8e-8,8e-8],),
    #                              zaxis = dict(nticks=4, range=[-2e-7,2e-7],),),
    
    #                  scene3 = dict(xaxis = dict(nticks=4,range=[-8e-8,8e-8],),
    #                              yaxis = dict(nticks=4, range=[-8e-8,8e-8],),
    #                              zaxis = dict(nticks=4, range=[-2e-7,2e-7],),),
   
   
                     
                      )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
