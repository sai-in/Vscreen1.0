# Streamlit_final.py [final]
"""
Created on 14th July 2021 
author @ sai charan

"""
from callInputBioscreen import callInputBioscreen
from callInputDomenico import callInputDomenico
from callInitializeC import callInitializeC
from callContaminantTransport import callContaminantTransport

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import plotly.graph_objects as go

import re
import utils
import base64
from io import BytesIO
import xlsxwriter

plt.rcParams['figure.dpi'] = 200
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and page orientation
st.set_page_config(layout="wide")
st.title("Screening tool for 3D groundwater contaminant transport")

st.sidebar.header("Control panel")


def collect_integers(x): return [int(i)
                                 for i in re.split("[^0-9]", x) if i != ""]


def collect_float(x): return [float(i)
                              for i in re.split("[^0-9.-]", x) if i != ""]

st.sidebar.header("Upload excel file as Input")
excel_file = st.sidebar.file_uploader(' ',type=['xlsx'])

st.subheader("**Please select checkbox to see the output**")
checkbox = st.checkbox('Plot',value=True)

# Get model input
def callGUIInputBioScreen():
    # Option, Parameters, Domain, Observed, Modeled = callInputBioscreen()
    Option, Parameters, Domain, Observed, Modeled = callInputDomenico()

    if excel_file is not None:
        df = pd.read_excel(excel_file,index_col=0)
        df.index.rename("Parameters",inplace=True) 
        df.columns = ['Values']

        dct = df.to_dict('dict')['Values']

        # Parameter dropdown
        pDropdown = st.sidebar.beta_expander("Parameters", expanded=False)

        Parameters['alphaX'] = pDropdown.slider('alpha_X', 0.0, 500.0, float(dct['alphaX']), 1.0)
        Parameters['alphaY'] = pDropdown.slider(
            'alpha_Y', 0.0, 50.0, float(dct['alphaY']), 0.1)
        Parameters['alphaZ'] = pDropdown.slider(
            'alpha_Z', 0.0, 50.0, float(dct['alphaZ']), 0.1)
        Parameters['vx'] = pDropdown.slider(
            "v_x", 0.0, 200.0, float(dct['vx']), 0.5)
        Parameters['k'] = pDropdown.slider("k", 0.0, 20.0, float(dct['k']), 0.1)
        Parameters['ks'] = pDropdown.slider(
            "k_s", 0.0, 1.0, float(dct['ks']), 0.01)
        Parameters['R'] = pDropdown.slider("R", 0.0, 5.0, float(dct['R']), 0.01)

        # Domain dropdown
        dDropdown = st.sidebar.beta_expander("Domain", expanded=False)
        Domain['tN'] = dDropdown.number_input(
            "simulation time", value=float(dct['tN']), step=0.1)
        Domain['x1'] = dDropdown.number_input(
            "domain x start", value= float(dct['x1']), step=10.0)
        Domain['xN'] = dDropdown.number_input(
            "domain x end", value=float(dct['xN']), step=10.0)
        Domain['y1'] = dDropdown.number_input(
            "domain y start", value=float(dct['y1']), step=1.0)
        Domain['yN'] = dDropdown.number_input(
            "domain y end", value=float(dct['yN']), step=1.0)
        Domain['z1'] = dDropdown.number_input(
            "domain z start", value=float(dct['z1']), step=1.0)
        Domain['zN'] = dDropdown.number_input(
            "domain z end", value=float(dct['zN']), step=1.0)

        # Boundary condition dropdown
        bDropdown = st.sidebar.beta_expander("Boundary", expanded=False)
        Parameters['n'] = bDropdown.number_input(
            "number of sources", value=len(list(map(float,dct['c0'].split(',')))), step=1)
        if dct['boundary'] == 'Dirichlet':
            index_ = 0
        elif dct['boundary'] == 'Cauchy':
            index_ = 1
        else:
            st.warning("Please enter correct Boundary")
        Option['boundary'] = bDropdown.selectbox("Boundary condition",
                                                 ['Dirichlet', 'Cauchy'], index=index_)
        c0 = bDropdown.text_input("c_0", value=dct['c0'])
        Parameters['c0'] = collect_float(c0)
        X1 = bDropdown.text_input("source x", value=str(''.join(map(str, dct['X1']))))
        Parameters['X1'] = collect_float(X1)
        Y1 = bDropdown.text_input("source y start", value=str(''.join(map(str, dct['Y1']))))
        Parameters['Y1'] = collect_float(Y1)
        Y2 = bDropdown.text_input("source y end", value=str(
            ''.join(map(str, dct['Y2']))))
        Parameters['Y2'] = collect_float(Y2)
        Z1 = bDropdown.text_input("source z start", value=str(
            ''.join(map(str, dct['Z1']))))
        Parameters['Z1'] = collect_float(Z1)
        Z2 = bDropdown.text_input("source z end", value=str(
            ''.join(map(str, dct['Z2']))))
        Parameters['Z2'] = collect_float(Z2)
        t1 = bDropdown.text_input("source t start", value=str(
            ''.join(map(str, dct['t1']))))
        Domain['t1'] = collect_float(t1)
        tp = bDropdown.text_input("source pulse end", value=str(
            ''.join(map(str, dct['tp']))))
        Parameters['tp'] = collect_float(tp)

    else:
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

    # call check parameter validity?
    if c0[-1] == ',':
        st.warning(
            "Initial source concentrations: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, c0.split(',')))) != Parameters['n']):
        st.warning(
            f"Please enter {Parameters['n']} values of Initial source concentrations")
        raise f"Please enter {Parameters['n']} values"

    if tp[-1] == ',':
        st.warning(
            "tp: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, tp.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of tp")
        raise f"Please enter {Parameters['n']} values"

    if t1[-1] == ',':
        st.warning(
            "t1: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, tp.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of t1")
        raise f"Please enter {Parameters['n']} values"

    if X1[-1] == ',':
        st.warning(
            "X1: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, X1.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of X1")
        raise f"Please enter {Parameters['n']} values"

    if Y1[-1] == ',':
        st.warning(
            "Y1: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, Y1.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of Y1")
        raise f"Please enter {Parameters['n']} values"

    if Y2[-1] == ',':
        st.warning(
            "Y2: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, Y2.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of Y2")
        raise f"Please enter {Parameters['n']} values"

    if Z1[-1] == ',':
        st.warning(
            "Z1: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, Z1.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of Z1")
        raise f"Please enter {Parameters['n']} values"

    if Z2[-1] == ',':
        st.warning(
            "Z2: Text input is ending with comma, Please remove comma after final value")
        raise 'Please remove the comma after the final value'
    elif (len(list(map(float, Z2.split(',')))) != Parameters['n']):
        st.warning(f"Please enter {Parameters['n']} values of Z2")
        raise f"Please enter {Parameters['n']} values"

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


@st.cache(suppress_st_warning=True)
def callGUIContaminantTransport(Parameters, Option, Domain):
    sWexler, sDomenico, sNew = callContaminantTransport(
        Option, Parameters, Domain, cWexler, cDomenico, cNew)

    return sWexler, sDomenico, sNew

if checkbox:
    cWexler, cDomenico, cNew = callInitializeC(Domain, Parameters, Option)
    sWexler, sDomenico, sNew = callGUIContaminantTransport(
        Parameters, Option, Domain)

if checkbox:
    # Plot results
    # Contour plots
    col1, col2, col3 = st.beta_columns((1, 1, 1))
    # XY contour
    col1.subheader("**X-Y contour plot**")
    if len(Domain['y']) > 1:
        xyContour = col1.text_input(label='XY contour values',
                                    value=str(','.join(map(str, Domain['contours']))))
        if xyContour == '':
            st.warning("xyContour: Text input is empty, Please enter any value")
            raise 'Please enter at least one value'
        elif xyContour[-1] == ',':
            st.warning(
                "xyContour: Text input is ending with comma, Please remove comma after final value")
            raise 'Please remove the comma after the final value'
        elif (list(map(float, xyContour.split(','))) != sorted(list(map(float, xyContour.split(','))))):
            st.warning("xyContour: Please enter the values in ascending order")
            raise 'Please enter the values in decreasing order'
        xyContour = collect_float(xyContour)
    if len(Domain['z']) > 1:
        xyContourZ = col1.slider("XY contour: z slice",
                                 float(Domain['z1']), float(Domain['zN']),
                                 float(Domain['z1']), float(Domain['deltaZ']))
        xyContourZIndex = np.searchsorted(Domain['z'], xyContourZ)
    else:
        xyContourZIndex = 0

    plt1 = plt.contourf(Domain['y'], Domain['x'], sNew[:, :, xyContourZIndex],
                        norm=LogNorm(), levels=xyContour, cmap='jet')
    # plt1 = plt.contour(Domain['y'], Domain['x'], sNew[:, :, xyContourZIndex],
    #                     levels=xyContour, cmap='jet')
    plt.clabel(plt1, colors='black')
    plt.xlabel('y [m]', fontsize=12, fontweight='bold')
    plt.ylabel('x [m]', fontsize=12, fontweight='bold')
    col1.pyplot()
    # fig1 = go.Figure(data = go.Contour(x = Domain['y'], y = Domain['x'],
    #                                       z=sNew[:,:,xyContourZIndex],
    #                                       showscale=False,
    #                                       contours=dict(coloring ='heatmap',
    #                                                     showlabels = True,
    #                                                     labelfont = dict(size = 12,color = 'black'))))
    # fig1.update_layout(xaxis_title='y [m]',
    #                    yaxis_title= 'x [m]',
    #                    font=dict(
    #                             family="Courier New, monospace",
    #                             size=15,
    #                             color="black"))
    # col1.plotly_chart(fig1,use_container_width=True)

    # XZ contour
    col2.subheader("**X-Z contour plot**")
    if len(Domain['z']) > 1:
        xzContour = col2.text_input(label='XZ contour values',
                                    value=str(','.join(map(str, Domain['contours']))))
        if xzContour == '':
            st.warning("xzContour: Text input is empty, Please enter any value")
            raise 'Please enter at least one value'
        elif xzContour[-1] == ',':
            st.warning(
                "xzContour: Text input is ending with comma, Please remove comma after final value")
            raise 'Please remove the comma after the final value'
        elif (list(map(float, xzContour.split(','))) != sorted(list(map(float, xzContour.split(','))))):
            st.warning("xzContour: Please enter the values in ascending order")
            raise 'Please enter the values in decreasing order'
        xzContour = collect_float(xzContour)
    if len(Domain['y']) > 1:
        xzContourY = col2.slider("XZ contour: y slice",
                                 float(Domain['y1']), float(Domain['yN']), float(Domain['y1']), float(Domain['deltaY']))
        xzContourYIndex = np.searchsorted(Domain['y'], xzContourY)
    else:
        xzContourYIndex = 0
    plt2 = plt.contourf(Domain['z'], Domain['x'], sNew[:, xzContourYIndex, :],
                        norm=LogNorm(), levels=xzContour, cmap='jet')
    plt.clabel(plt2, colors='black')
    plt.xlabel('z [m]', fontsize=12, fontweight='bold')
    plt.ylabel('x [m]', fontsize=12, fontweight='bold')
    col2.pyplot()
    # fig2 = go.Figure(data = go.Contour(x = Domain['z'], y = Domain['x'],
    #                                       z=sNew[:,xzContourYIndex,:],
    #                                       showscale=False,
    #                                       contours=dict(coloring ='heatmap',
    #                                                     showlabels = True,
    #                                                     labelfont = dict(size = 12,color = 'white'))))
    # fig2.update_layout(xaxis_title='z [m]',
    #                    yaxis_title= 'x [m]',
    #                    font=dict(
    #                             family="Courier New, monospace",
    #                             size=15,
    #                             color="black"))
    # col2.plotly_chart(fig2,use_container_width=True)


    # YZ contour
    col3.subheader("**Y-Z contour plot**")
    if (len(Domain['y']) > 1) and (len(Domain['z']) > 1):
        yzContour = col3.text_input(label='YZ contour values',
                                    value=str(','.join(map(str, Domain['contours']))))
        if yzContour == '':
            st.warning("yzContour: Text input is empty, Please enter any value")
            raise 'Please enter at least one value'
        elif yzContour[-1] == ',':
            st.warning(
                "yzContour: Text input is ending with comma, Please remove comma after final value")
            raise 'Please remove the comma after the final value'
        elif (list(map(float, yzContour.split(','))) != sorted(list(map(float, yzContour.split(','))))):
            st.warning("yzContour: Please enter the values in ascending order")
            raise 'Please enter the values in decreasing order'
        yzContour = collect_float(yzContour)
        yzContourX = col3.slider("YZ contour: x slice",
                                 float(Domain['x1']), float(Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
        yzContourXIndex = np.searchsorted(Domain['x'], yzContourX)

    plt3 = plt.contourf(Domain['z'], Domain['y'], sNew[yzContourXIndex, :, :],
                        norm=LogNorm(), levels=yzContour, cmap='jet')
    plt.clabel(plt3, colors='black')
    plt.xlabel('z [m]', fontsize=12, fontweight='bold')
    plt.ylabel('y [m]', fontsize=12, fontweight='bold')
    col3.pyplot()

    # fig3 = go.Figure(data = go.Contour(x = Domain['z'], y = Domain['y'],
    #                                       z=sNew[yzContourXIndex, :, :],
    #                                       showscale=False,
    #                                       contours=dict(coloring ='heatmap',
    #                                                     showlabels = True,
    #                                                     labelfont = dict(size = 12,color = 'white'))))
    # fig3.update_layout(xaxis_title='z [m]',
    #                    yaxis_title= 'y [m]',
    #                    font=dict(
    #                             family="Courier New, monospace",
    #                             size=15,
    #                             color="black"))
    # col3.plotly_chart(fig3,use_container_width=True)


    # Line plots
    col4, col5, col6 = st.beta_columns((1, 1, 1))
    # Z line
    col4.subheader("**Z line plot**")
    if len(Domain['z']) > 1:
        zLineX = col4.slider("z line: x slice", float(Domain['x1']), float(
            Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
    zLineXIndex = np.searchsorted(Domain['x'], zLineX)
    if len(Domain['y']) > 1:
        zLineY = col4.slider("z line: y sice", float(Domain['y1']), float(
            Domain['yN']), float(Domain['y1']), float(Domain['deltaY']))
        zLineYIndex = np.searchsorted(Domain['y'], zLineY)
    else:
        zLineYIndex = 0
    plt.plot(sNew[zLineXIndex, zLineYIndex, :])
    plt.xlabel('z [m]', fontsize=12, fontweight='bold')
    plt.ylabel(r'$C \ [mg L^{-1}]$', fontsize=12, fontweight='bold')
    col4.pyplot()

    # fig4 = go.Figure(data = go.Scatter(y=sNew[zLineXIndex,zLineYIndex, :],
    #                                             mode='lines+markers'))
    # fig4.update_layout(xaxis_title='z [m]',
    #                    yaxis_title= "C[mg/L]",
    #                    font=dict(
    #                             family="Courier New, monospace",
    #                             size=15,
    #                             color="RebeccaPurple"))
    # col4.plotly_chart(fig4,use_container_width=True)

    # Y line
    col5.subheader("**Y line plot**")
    if len(Domain['y']) > 1:
        yLineX = col5.slider("y line: x slice", float(Domain['x1']), float(
            Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
    yLineXIndex = np.searchsorted(Domain['x'], yLineX)
    if len(Domain['z']) > 1:
        yLineZ = col5.slider("y line: z sice", float(Domain['z1']), float(
            Domain['zN']), float(Domain['z1']), max(1.0, float(Domain['deltaZ'])))
        yLineZIndex = np.searchsorted(Domain['z'], yLineZ)
    else:
        yLineZIndex = 0
    plt.plot(sNew[yLineXIndex, :, yLineZIndex])
    plt.xlabel('y [m]', fontsize=12, fontweight='bold')
    plt.ylabel(r'$C \ [mg L^{-1}]$', fontsize=12, fontweight='bold')
    col5.pyplot()
    # fig5 = go.Figure(data = go.Scatter(y=sNew[yLineXIndex, :, yLineZIndex],
    #                                             mode='lines+markers'))
    # fig5.update_layout(xaxis_title='y [m]',
    #                    yaxis_title= "C[mg/L]",
    #                    font=dict(
    #                             family="Courier New, monospace",
    #                             size=15,
    #                             color="RebeccaPurple"))
    # col5.plotly_chart(fig5,use_container_width=True)


    # X line
    col6.subheader("**X line plot**")
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
    plt.xlabel('x [m]', fontsize=12, fontweight='bold')
    plt.ylabel(r'$C \ [mg L^{-1}]$', fontsize=12, fontweight='bold')
    col6.pyplot()

    # fig6 = go.Figure(data = go.Scatter(y=sNew[:, xLineYIndex, xLineZIndex],
    #                                             mode='lines+markers'))
    # fig6.update_layout(xaxis_title='x [m]',
    #                    yaxis_title= "C[mg/L]",
    #                    font=dict(
    #                             family="Courier New, monospace",
    #                             size=15,
    #                             color="RebeccaPurple"))
    # col6.plotly_chart(fig6,use_container_width=True)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Concentrations.xlsx">Download Excel file</a>'

if checkbox:
    col10,col11,col12 = st.beta_columns((1,1,1))
    with col10:
        st.subheader('**Concentrations along X-Y**')
        if len(Domain['z']) > 1:
            Z_index = st.slider('Z_index',float(Domain['z1']), float(
                Domain['zN']), float(Domain['z1']), float(Domain['deltaZ']))
            Zindex = np.searchsorted(Domain['z'], Z_index)
        else:
            Zindex = 0
        Z_array = np.array(sNew[:,:,Zindex])
        df_Z = pd.DataFrame(Z_array).T

        st.markdown(get_table_download_link(df_Z), unsafe_allow_html=True)

    with col11:
        st.subheader('**Concentrations along X-Z**')
        if len(Domain['y']) > 1:
            Y_index = st.slider('Y_index',float(Domain['y1']), float(
                Domain['yN']), float(Domain['y1']), float(Domain['deltaY']))
            Yindex = np.searchsorted(Domain['y'], Z_index)
        else:
            Yindex = 0
        Y_array = np.array(sNew[:,Yindex,:])
        df_Y = pd.DataFrame(Y_array).T

        st.markdown(get_table_download_link(df_Y), unsafe_allow_html=True)

    with col12:
        st.subheader('**Concentrations along Y-Z**')
        if len(Domain['x']) > 1:
            X_index = st.slider('X_index',float(Domain['x1']), float(
                Domain['xN']), float(Domain['x1']), float(Domain['deltaX']))
            Xindex = np.searchsorted(Domain['x'], Z_index)
        else:
            Xindex = 0
        X_array = np.array(sNew[Xindex,:,:])
        df_X = pd.DataFrame(X_array).T

        st.markdown(get_table_download_link(df_X), unsafe_allow_html=True)


dt = dict()
dt['boundary'] = Option['boundary']
dt['alphaX'] = Parameters['alphaX']
dt['alphaY'] = Parameters['alphaY']
dt['alphaZ'] = Parameters['alphaZ']
dt['vx']     = Parameters['vx']
dt['k']      = Parameters['k']
dt['R']      = Parameters['R']
dt['ks']     = Parameters['ks']
dt['c0']     = str(','.join(map(str, Parameters['c0'])))
dt['X1']     = str(','.join(map(str, Parameters['X1'])))
dt['Y1']     = str(','.join(map(str, Parameters['Y1'])))
dt['Y2']     = str(','.join(map(str, Parameters['Y2'])))
dt['Z1']     = str(','.join(map(str, Parameters['Z1'])))
dt['Z2']     = str(','.join(map(str, Parameters['Z2'])))
dt['tp']     = str(','.join(map(str, Parameters['tp'])))
dt['xSteps'] = Domain['xSteps']
dt['x1']     = Domain['x1']
dt['xN']     = Domain['xN']
dt['ySteps'] = Domain['ySteps']
dt['y1']     = Domain['y1']
dt['yN']     = Domain['yN']
dt['zSteps'] = Domain['zSteps']
dt['z1']     = Domain['z1']
dt['zN']     = Domain['zN']
dt['tN']     = Domain['tN']
dt['t1']     = str(','.join(map(str, Domain['t1'])))

df1 = pd.DataFrame({'Parameters':dt.keys(),'Values':dt.values()})
df1.set_index('Parameters',inplace=True)
#st.write(df1)
# def to_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, sheet_name='Sheet1')
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data

def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Your_File.xlsx">Download Excel file</a>'

st.sidebar.header('Download Input Parameter values')
st.sidebar.markdown(get_table_download_link(df1), unsafe_allow_html=True)

# col8 = st.beta_columns((1))
# st.header("Documentation")
# st.text("Insert documentation here")

# col9 = st.beta_columns((1))
# st.header("References")
# st.text("Insert references here")


