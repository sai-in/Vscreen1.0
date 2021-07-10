import numpy as np
from callDomenico import callDomenico
from callWexler import callWexler


def callContaminantTransport(Option, Parameters, Domain, cWexler, cDomenico, cNew):

    # In numpy values gets passed by pass by reference
    # to prevent that we are using c.copy()
    # cWexler = c.copy()
    # cDomenico = c.copy()
    # cNew = c.copy()

    for ci in range(len(Parameters['c0'])):
        for xi in range(1, len(Domain['x'])):
            for yi in range(max(1, len(Domain['y']))):
                for zi in range(max(1, len(Domain['z']))):

                    tauDomenico = abs(
                        (Domain['x'][xi]-Parameters['X1'][ci])/Parameters['vx'])
                    n = Parameters['vx']*(Domain['tN']-Domain['t1']
                                          [ci]) / (Parameters['alphaX'])**0.25
                    tauNew = abs(((Domain['x'][xi]-Parameters['X1'][ci])/Parameters['vx']) /
                                 (1 + (Domain['x'][xi]/(Parameters['vx']*(Domain['tN']-Domain['t1'][ci])))**n)**(1/n))
                    if Option['Wexler'] == 1:
                        cWexler[ci, xi, yi, zi] = callWexler(Domain['x'][xi], Domain['y'][yi], Domain['z'][zi],
                                                             (Domain['tN']-Domain['t1']
                                                              [ci]), Parameters['vx'],
                                                             Parameters['alphaX'], Parameters['alphaY'], Parameters['alphaZ'],
                                                             Parameters['k'], Parameters['ks'], Parameters['R'], Parameters['X1'][ci],
                                                             Parameters['Y1'][ci], Parameters['Y2'][ci], Parameters['Z1'][ci],
                                                             Parameters['Z2'][ci], Parameters['c0'][ci], Parameters['tp'][ci], Option)
                    if Option["Domenico"] == 1:
                        cDomenico[ci, xi, yi, zi] = callDomenico(Domain['x'][xi], Domain['y'][yi], Domain['z'][zi],
                                                                 (Domain['tN']-Domain['t1']
                                                                 [ci]), Parameters['vx'],
                                                                 Parameters['alphaX'], Parameters['alphaY'], Parameters['alphaZ'],
                                                                 Parameters['k'], Parameters['ks'], Parameters['R'], Parameters['X1'][ci],
                                                                 Parameters['Y1'][ci], Parameters['Y2'][ci], Parameters['Z1'][ci],
                                                                 Parameters['Z2'][ci], Parameters['c0'][ci], Parameters['tp'][ci],
                                                                 tauDomenico, Option)
                    if Option['New'] == 1:
                        cNew[ci, xi, yi, zi] = callDomenico(Domain['x'][xi], Domain['y'][yi], Domain['z'][zi],
                                                            (Domain['tN']-Domain['t1']
                                                            [ci]), Parameters['vx'],
                                                            Parameters['alphaX'], Parameters['alphaY'], Parameters['alphaZ'],
                                                            Parameters['k'], Parameters['ks'], Parameters['R'], Parameters['X1'][ci],
                                                            Parameters['Y1'][ci], Parameters['Y2'][ci], Parameters['Z1'][ci],
                                                            Parameters['Z2'][ci], Parameters['c0'][ci], Parameters['tp'][ci],
                                                            tauNew, Option)

    sWexler = np.sum(cWexler, axis=0)
    sDomenico = np.sum(cDomenico, axis=0)
    sNew = np.sum(cNew, axis=0)

    # source boundary correction
    if Option['Wexler'] == 1:
        sWexler[0, :, :] = np.amax(cWexler[:, 0, :, :], axis=0)
    else:
        sWexler = 0
    if Option['Domenico'] == 1:
        sDomenico[0, :, :] = np.amax(cDomenico[:, 0, :, :], axis=0)
    else:
        sDomenico = 0
    if Option['New'] == 1:
        sNew[0, :, :] = np.amax(cNew[:, 0, :, :], axis=0)
    else:
        sNew = 0

    return sWexler, sDomenico, sNew
