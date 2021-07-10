from callInputBioscreen import callInputBioscreen
from callInputDomenico import callInputDomenico
from callInitializeC import callInitializeC
# from callContourPlot import callContourPlotXY, callContourPlotXZ, callContourPlotYZ
# from callPcolorPlot import callPcolorPlot
# from callLinePlot import callXlinePlot, callYlinePlot, callZlinePlot
from callContaminantTransport import callContaminantTransport

# import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
# import time
import re

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and page orientation
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center; color:black; font-size:5em'>\
            v Screen 1.0</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:black; font-size:2em'>\
            Screening tool for 3D groundwater contaminant transport</h1>", unsafe_allow_html=True)
st.sidebar.markdown('<h1><span style="color:black; font-size:1.2em;">\
        Control Panel </span></h1>', unsafe_allow_html=True)


def collect_integers(x): return [int(i)
                                 for i in re.split("[^0-9]", x) if i != ""]


def collect_float(x): return [float(i)
                              for i in re.split("[^0-9.-]", x) if i != ""]

# Get model input


def callGUIInputBioScreen():
    Option, Parameters, Domain, Observed, Modeled = callInputBioscreen()
    # Option, Parameters, Domain, Observed, Modeled = callInputDomenico()

    # Parameter dropdown
    pDropdown = st.sidebar.beta_expander("Parameters", expanded=False)
    Parameters['alphaX'] = pDropdown.slider(
        'alpha_X', 0.0, 500.0, Parameters['alphaX'], 1.0)
    Parameters['alphaY'] = pDropdown.slider(
        'alpha_Y', 0.0, 50.0, Parameters['alphaY'], 0.1)
    Parameters['alphaZ'] = pDropdown.slider(
        'alpha_Z', 0.0, 50.0, Parameters['alphaZ'], 0.1)
    Parameters['vx'] = pDropdown.slider(
        "v_x", 0.0, 200.0, Parameters['vx'], 0.5)
    Parameters['k'] = pDropdown.slider("k", 0.0, 20.0, Parameters['k'], 0.1)
    Parameters['ks'] = pDropdown.slider(
        "k_s", 0.0, 1.0, Parameters['ks'], 0.01)
    Parameters['R'] = pDropdown.slider("R", 0.0, 5.0, Parameters['R'], 0.01)

    # Domain dropdown
    dDropdown = st.sidebar.beta_expander("Domain", expanded=False)
    Domain['tN'] = dDropdown.number_input(
        "simulation time", value=Domain['tN'], step=0.1)
    Domain['x1'] = dDropdown.number_input(
        "domain x start", value=Domain['x1'], step=10.0)
    Domain['xN'] = dDropdown.number_input(
        "domain x end", value=Domain['xN'], step=10.0)
    Domain['y1'] = dDropdown.number_input(
        "domain y start", value=Domain['y1'], step=1.0)
    Domain['yN'] = dDropdown.number_input(
        "domain y end", value=Domain['yN'], step=1.0)
    Domain['z1'] = dDropdown.number_input(
        "domain z start", value=Domain['z1'], step=1.0)
    Domain['zN'] = dDropdown.number_input(
        "domain z end", value=Domain['zN'], step=1.0)

    # Boundary condition dropdown
    bDropdown = st.sidebar.beta_expander("Boundary", expanded=False)
    Parameters['n'] = bDropdown.number_input(
        "number of sources", value=len(Parameters['c0']), step=1)
    Option['boundary'] = bDropdown.selectbox("Boundary condition",
                                             ['Dirichlet', 'Cauchy'], index=0)
    c0 = bDropdown.text_input("c_0", value=str(
        ','.join(map(str, Parameters['c0']))))
    Parameters['c0'] = collect_float(c0)
    X1 = bDropdown.text_input("source x", value=str(
        ','.join(map(str, Parameters['X1']))))
    Parameters['X1'] = collect_float(X1)
    Y1 = bDropdown.text_input("source y start", value=str(
        ','.join(map(str, Parameters['Y1']))))
    Parameters['Y1'] = collect_float(Y1)
    Y2 = bDropdown.text_input("source y end", value=str(
        ','.join(map(str, Parameters['Y2']))))
    Parameters['Y2'] = collect_float(Y2)
    Z1 = bDropdown.text_input("source z start", value=str(
        ','.join(map(str, Parameters['Z1']))))
    Parameters['Z1'] = collect_float(Z1)
    Z2 = bDropdown.text_input("source z end", value=str(
        ','.join(map(str, Parameters['Z2']))))
    Parameters['Z2'] = collect_float(Z2)
    t1 = bDropdown.text_input("source t start", value=str(
        ','.join(map(str, Domain['t1']))))
    Domain['t1'] = collect_float(t1)
    tp = bDropdown.text_input("source pulse end", value=str(
        ','.join(map(str, Parameters['tp']))))
    Parameters['tp'] = collect_float(tp)

    Domain['deltaX'] = (Domain['xN']-Domain['x1'])/Domain['xSteps']
    Domain['x'] = np.arange(Domain['x1'], Domain['xN'], Domain['deltaX'])
    Domain['deltaY'] = (Domain['yN'] - Domain['y1'])/Domain['ySteps']
    if Domain['yN'] == Domain['y1']:
        Domain['y'] = np.array([Domain['y1']])
    elif Domain['yN'] > Domain['y1']:
        Domain['y'] = np.arange(Domain['y1'], Domain['yN'] + Domain['deltaY'],
                                Domain['deltaY'])
    Domain['deltaZ'] = (Domain['zN'] - Domain['z1'])/Domain['zSteps']
    if Domain['zN'] == Domain['z1']:
        Domain['z'] = np.array([Domain['z1']])
    elif Domain['zN'] > Domain['z1']:
        Domain['z'] = np.arange(Domain['z1'], Domain['zN'] + Domain['deltaZ'],
                                Domain['deltaZ'])
    return Option, Parameters, Domain, Observed, Modeled


Option, Parameters, Domain, Observed, Modeled = callGUIInputBioScreen()

# call check parameter validity?

cWexler, cDomenico, cNew = callInitializeC(Domain, Parameters, Option)


@st.cache(suppress_st_warning=True)
def callGUIContaminantTransport(Parameters, Option, Domain):
    sWexler, sDomenico, sNew = callContaminantTransport(
        Option, Parameters, Domain, cWexler, cDomenico, cNew)

    return sWexler, sDomenico, sNew


sWexler, sDomenico, sNew = callGUIContaminantTransport(
    Parameters, Option, Domain)


def callGUIPlot(Parameters, Option, Domain):
    # Plot results
    col1, col2 = st.beta_columns((1, 1))
    # Contour plot
    if len(Domain['y']) > 1:
        xyContour = col1.text_input(
            label='XY contour values', value='0.01,0.1,1,10')
        xyContour = collect_float(xyContour)
        if len(Domain['z']) > 1:
            xyContourZ = col1.slider("XY contour: z slice",
                                     float(Domain['z1']), float(Domain['zN']),
                                     float(Domain['z1']), float(Domain['deltaZ']))
            xyContourZIndex = np.searchsorted(Domain['z'], xyContourZ)
        else:
            xyContourZIndex = 0
        plt1 = plt.contourf(Domain['y'], Domain['x'], sNew[:, :,
                                                           xyContourZIndex], xyContour, cmap='jet')
        plt.clabel(plt1, colors='black')
        plt.xlabel('y [m]')
        plt.ylabel('x [m]')
        col1.pyplot()
    # Line plot
    if len(Domain['z']) > 1:
        zLineX = col2.slider("z line: x slice", float(Domain['x1']), float(
            Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
        zLineXIndex = np.searchsorted(Domain['x'], zLineX)
        if len(Domain['y']) > 1:
            zLineY = col2.slider("z line: y sice", float(Domain['y1']), float(
                Domain['yN']), float(Domain['y1']), float(Domain['deltaY']))
            zLineYIndex = np.searchsorted(Domain['y'], zLineY)
        else:
            zLineYIndex = 0
        plt.plot(sNew[zLineXIndex, zLineYIndex, :])
        col2.pyplot()

    col3, col4 = st.beta_columns((1, 1))
    # Contour plot
    if len(Domain['z']) > 1:
        xzContour = col3.text_input(
            label='XZ contour values', value='0.01,0.1,1,10')
        xzContour = collect_float(xzContour)
        if len(Domain['y']) > 1:
            xzContourY = col3.slider("XZ contour: y slice",
                                     float(Domain['y1']), float(Domain['yN']), float(Domain['y1']), float(Domain['deltaY']))
            xzContourYIndex = np.searchsorted(Domain['y'], xzContourY)
        else:
            xzContourYIndex = 0
        plt.contourf(Domain['z'], Domain['x'],
                     sNew[:, xzContourYIndex, :], xzContour, cmap='jet')
        col3.pyplot()
    # Line plot
    if len(Domain['y']) > 1:
        yLineX = col4.slider("y line: x slice", float(Domain['x1']), float(
            Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
        yLineXIndex = np.searchsorted(Domain['x'], yLineX)
        if len(Domain['z']) > 1:
            yLineZ = col4.slider("y line: z sice", float(Domain['z1']), float(
                Domain['zN']), float(Domain['z1']), max(1.0, float(Domain['deltaZ'])))
            yLineZIndex = np.searchsorted(Domain['z'], yLineZ)
        else:
            yLineZIndex = 0
        plt.plot(sNew[yLineXIndex, :, yLineZIndex])
        col4.pyplot()

    col5, col6 = st.beta_columns((1, 1))
    # Contour plot
    if (len(Domain['y']) > 1) and (len(Domain['z']) > 1):
        yzContour = col5.text_input(
            label='YZ contour values', value='0.01,0.1,1,10')
        yzContour = collect_float(yzContour)
        yzContourX = col5.slider("YZ contour: x slice",
                                 float(Domain['x1']), float(Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
        yzContourXIndex = np.searchsorted(Domain['x'], yzContourX)
        plt.contourf(Domain['z'], Domain['y'],
                     sNew[yzContourXIndex, :, :], yzContour, cmap='jet')
        col5.pyplot()
    # Line plot
    if len(Domain['y']) > 1:
        xLineY = col6.slider("x line: y slice", float(Domain['y1']), float(
            Domain['yN']), float(Domain['y1']), float(Domain['deltaY']))
        xLineYIndex = np.searchsorted(Domain['y'], xLineY)
    else:
        xLineYIndex = 0
    if len(Domain['z']) > 1:
        xLineZ = col6.slider("x line: z sice", float(Domain['z1']), float(
            Domain['zN']), float(Domain['z1']), float(Domain['deltaZ']))
        xLineZIndex = np.searchsorted(Domain['z'], xLineZ)
    else:
        xLineZIndex = 0
    plt.plot(sNew[:, xLineYIndex, xLineZIndex])
    col6.pyplot()


callGUIPlot(Parameters, Option, Domain)
