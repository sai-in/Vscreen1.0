
import numpy as np
from math import exp


def callInitializeC(Domain, Parameters, Option):

    # Initiaize concentration vector
    c = np.empty(((len(Parameters['c0']), len(Domain['x']),
                   len(Domain['y']), len(Domain['z'])))) # [mg L-1]

    # Source boundary condition
    c[:, 0, :, :] = 0

    for c0loop in range(len(Parameters['c0'])):
        if Parameters['tp'][c0loop] >= (Domain['tN']-Domain['t1'][c0loop]):
            xLoop = len(np.arange(Domain['x1'], Parameters['X1'][c0loop] +
                                  Domain['deltaX'], Domain['deltaX']))-1
            if len(Domain['y']) > 1:
                y1Loop = max(1, len(np.arange(Domain['y1'], Parameters['Y1'][c0loop] +
                                              Domain['deltaY'], Domain['deltaY'])))-1
                y2Loop = max(1, len(np.arange(Domain['y1'], Parameters['Y2'][c0loop] +
                                              Domain['deltaY'], Domain['deltaY'])))
            else:
                y1Loop,y2Loop = 0,1
                
            for yloop in range(y1Loop, y2Loop):
                if len(Domain['z']) > 1:
                    z1Loop = max(1, len(np.arange(Domain['z1'], Parameters['Z1'][c0loop] +
                                                  Domain['deltaZ'], Domain['deltaZ'])))-1
                    z2Loop = max(1, len(np.arange(Domain['z1'], Parameters['Z2'][c0loop] +
                                                  Domain['deltaZ'], Domain['deltaZ'])))
                else:
                    z1Loop,z2Loop = 0,1
                    
                for zloop in range(z1Loop, z2Loop):
                    c[c0loop, xLoop, yloop, zloop] = (Parameters['c0'][c0loop] *exp(-Parameters['ks']*(Domain['tN']-Domain['t1'][c0loop])))

    if Option['Wexler'] == 1:
        cWexler = c.copy()
    else:
        cWexler = 0
    if Option['Domenico'] == 1:
        cDomenico = c.copy()
    else:
        cDomenico = 0
    if Option['New'] == 1:
        cNew = c.copy()
    else:
        cNew = 0

    return cWexler, cDomenico, cNew
