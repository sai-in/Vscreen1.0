# CallInput.py

import numpy as np


def callInputDomenico():

    Option = {}
    Parameters = {}
    Domain = {}
    Observed = {}
    Modeled = {}

    Option['Boundary'] = 'Dirichlet'
    Option['Wexler'] = 0
    Option['Domenico'] = 0
    Option['New'] = 1

    Parameters['alphaX'] = 42.58
    Parameters['alphaY'] = 8.4310
    Parameters['alphaZ'] = 0.00642

    Parameters['X1'] = [0.0]
    Parameters['Y1'] = [-120.0]
    Parameters['Y2'] = [120.0]
    Parameters['Z1'] = [-2.5]
    Parameters['Z2'] = [2.5]

    Parameters['vx'] = 0.2151         # Velocity in x direction [m day-1]
    Parameters['k'] = 0.0               # First order deacay constant [day-1]
    Parameters['R'] = 1.0              # Retardation factor
    Parameters['c0'] = [850.0]
    if Option['Boundary'] == 'Cauchy':
        Parameters['c0'] = [Parameters['c0']*Parameters['vx']]
    Parameters['ks'] = 0.00              # Source decay
    Parameters['tp'] = [5110.0]

    Domain['xSteps'] = 40
    Domain['x1'] = 0.0
    Domain['xN'] = 5000.0
    Domain['deltaX'] = (Domain['xN'] - Domain['x1'])/Domain['xSteps']
    Domain['x'] = np.arange(Domain['x1'], Domain['xN'] + Domain['deltaX'],
                            Domain['deltaX'])

    Domain['ySteps'] = 40.0
    Domain['y1'] = -1200.0
    Domain['yN'] = 1200.0
    Domain['deltaY'] = (Domain['yN'] - Domain['y1'])/Domain['ySteps']
    Domain['y'] = np.arange(Domain['y1'], Domain['yN'] + Domain['deltaY'],
                            Domain['deltaY'])

    Domain['zSteps'] = 40.0
    Domain['z1'] = -100.0
    Domain['zN'] = 100.0  # I have made it to 1, to avoid zero divisibility error
    Domain['deltaZ'] = (Domain['zN'] - Domain['z1'])/Domain['zSteps']
    Domain['z'] = np.arange(Domain['z1'], Domain['zN'] + Domain['deltaZ'],
                            Domain['deltaZ'])
    
    # if Domain['zN'] == Domain['z1']:
    #     Domain['z'] = [0.0]
    # elif Domain['zN'] > Domain['z1']:
    #     Domain['z'] = np.arange(
    #         Domain['z1'], Domain['zN'] + Domain['deltaZ'], Domain['deltaZ'])

    Domain['t1'] = [0.0] # Simulation start time [year]
    Domain['tN'] = 5100.0 # Simulation end time [year]
    
    Domain['contours'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    Observed = 0
    Modeled = 0
    
    return Option, Parameters, Domain, Observed, Modeled
