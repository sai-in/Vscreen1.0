# CallInput.py

import numpy as np


def callInputBioscreen():

    Option = {}
    Parameters = {}
    Domain = {}
    Observed = {}
    Modeled = {}

    Option['boundary'] = 'Dirichlet'
    Option['Wexler'] = 0
    Option['Domenico'] = 0
    Option['New'] = 1

    Parameters['alphaX'] = 13.3
    Parameters['alphaY'] = 1.3
    Parameters['alphaZ'] = 1.3

    Parameters['X1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Parameters['Y1'] = [0.0, 7.0, 37.0, -7.0, -37.0, -65.0]  # index 0, 3
    Parameters['Y2'] = [7.0, 37.0, 65.0, 0.0, -7.0, -37.0]  # index 0,3
    Parameters['Z1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Parameters['Z2'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Parameters['vx'] = 113.8             # Velocity in x direction [m day-1]
    Parameters['k'] = 4.6               # First order deacay constant [day-1]
    Parameters['R'] = 1.0                 # Retardation factor
    Parameters['c0'] = [13.68, 2.508, 0.0507, 13.68, 2.508, 0.0507]
    # Initial source concentration [mg L-1]
    Parameters['ks'] = 0.00              # Source decay
    Parameters['tp'] = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]

    Domain['xSteps'] = 20
    Domain['x1'] = 0.0
    Domain['xN'] = 300.0
    Domain['deltaX'] = (Domain['xN'] - Domain['x1'])/Domain['xSteps']
    Domain['x'] = np.arange(Domain['x1'], Domain['xN'] + Domain['deltaX'],
                            Domain['deltaX'])

    Domain['ySteps'] = 20
    Domain['y1'] = -100.0
    Domain['yN'] = 100.0
    Domain['deltaY'] = (Domain['yN'] - Domain['y1'])/Domain['ySteps']
    if Domain['yN'] == Domain['y1']:
        Domain['y'] = np.array([Domain['y1']])
    elif Domain['yN'] > Domain['y1']:
        Domain['y'] = np.arange(Domain['y1'], Domain['yN'] + Domain['deltaY'],
                                Domain['deltaY'])

    Domain['zSteps'] = 20
    Domain['z1'] = -100.0
    Domain['zN'] = 100.0  # I have made it to 1, to avoid zero divisibility error
    Domain['deltaZ'] = (Domain['zN'] - Domain['z1'])/Domain['zSteps']
    if Domain['zN'] == Domain['z1']:
        Domain['z'] = np.array([Domain['z1']])
    elif Domain['zN'] > Domain['z1']:
        Domain['z'] = np.arange(Domain['z1'], Domain['zN'] + Domain['deltaZ'],
                                Domain['deltaZ'])

    # Simulation start time [year]
    Domain['t1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Domain['tN'] = 6.0  # Simulation end time [year]
    
    Domain['contours'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    
    Observed['x'] = [0.0, 32.0, 64.0, 192.0, 288.0]
    Observed['y'] = [0.0, 0.0, 0.0, 0.0, 0.0]
    Observed['c'] = [12.0, 5.0, 1.0, 0.5, 0.001]

    Modeled['x'] = np.array([0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320])
    Modeled['y'] = np.array([-100, -50, 0, 50, 100])
    Modeled['z'] = np.array([[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                             [0.056, 0.096, 0.067, 0.035, 0.017, 0.008,
                                 0.003, 0.001, 0.001, 0.000, 0.000],
                             [13.544, 3.341, 1.059, 0.364, 0.130, 0.047,
                                 0.017, 0.006, 0.002, 0.001, 0.000],
                             [0.056, 0.096, 0.067, 0.035, 0.017, 0.008,
                                 0.003, 0.001, 0.001, 0.000, 0.000],
                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])

    return Option, Parameters, Domain, Observed, Modeled
