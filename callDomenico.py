
# CallDomenico.py
from math import erfc, erf, exp, sqrt, pi


def callDomenico(x, y, z, t, v, alphaX, alphaY, alphaZ, k, ks, R, X1, Y1, Y2, Z1, Z2, c0, tp, tau, Option):

    v = v / R
    k = k / R

    if Option['boundary'] == 'Dirichlet':
        def phiX(t, x, v, alphaX, k, ks, X1):
            return (exp(-ks * t) *
                    (exp((x - X1) / (2.0 * alphaX) * (1.0 - (1 + 4.0 * (k - ks) * alphaX / v) ** 0.5)) *
                    erfc(((x - X1) - v * t * (1.0 + 4.0 * (k - ks) * alphaX / v) ** 0.5) /
                         (2.0 * (alphaX * v * t) ** 0.5)) +
                    exp((x - X1) / (2.0 * alphaX) * (1.0 + (1 + 4.0 * k * alphaX / v) ** 0.5)) *
                    erfc(((x - X1) + v * t * (1.0 + 4.0 * k * alphaX / v) ** 0.5) /
                         (2.0 * (alphaX * v * t) ** 0.5))))

    elif Option['boundary'] == 'Cauchy':
        if k == ks*R:
            def phiX(t, x, v, alphaX, k, ks, X1):
                return (1-exp(-(k-ks)*t)*(1.0-0.5*erfc((x-X1-v*t)/(2*(alphaX*v*t)**0.5)) -
                        (v*t/(pi*alphaX))**0.5 *
                        (exp(-(x-X1-v*t)**2/(4*alphaX*v*t))) +
                        0.5*(1.0+(x-X1)/alphaX+v*t/alphaX) *
                        exp((x-X1)/alphaX) *
                        erfc((x-X1+v*t)/(2*(alphaX*v*t)**0.5))))

        else:
            def phiX(t, x, v, alphaX, k, ks, X1):
                return (1.0/(1.0+(1.0+(4.0*alphaX*(k-ks))/v)**0.5) *
                        exp((x-X1)/(2.0*alphaX)*(1.0-(1.0+4.0*(k-ks)*alphaX/v)**0.5)) *
                        erfc((x-X1-v*t*(1.0+4.0*(k-ks)*alphaX/v)**0.5)/(2.0*(alphaX*v*t)**0.5)) +
                        1.0/(1.0-(1.0+(4.0*alphaX*(k-ks))/v)**0.5) *
                        exp((x-X1)/(2.0*alphaX)*(1.0+(1.0+4.0*(k-ks)*alphaX/v)**0.5)) *
                        erfc((x-X1+v*t*(1.0+4.0*(k-ks)*alphaX/v)**0.5)/(2.0*(alphaX*v*t)**0.5)) +
                        v/(2.0*alphaX*(k-ks)) *
                        exp((x-X1)/alphaX-k*t) *
                        erfc((x-X1+v*t)/(2.0*(alphaX*v*t)**0.5)))

    # alphaY
    if (alphaY == 0):  # Z dimession not present
        def fY(tau, v, alphaY, y, Y1, Y2):
            return (2)

    elif Y1 == Y2:
        def fY(tau, v, alphaY, y, Y1, Y2):
            return ((1 / sqrt(pi * v * alphaY * tau)) *
                    (exp(-(y - Y1) ** 2.0 / (4.0 * v * alphaY * tau))))

    else:
        def fY(tau, v, alphaY, y, Y1, Y2):
            return (erf((y - Y1) / (2.0 * (v * alphaY * tau) ** 0.5)) -
                    erf((y - Y2) / (2.0 * (v * alphaY * tau) ** 0.5)))

    # alphaZ
    if (alphaZ == 0):
        def fZ(tau, v, alphaZ, z, Z1, Z2):
            return (2)

    elif Z1 == Z2:
        def fZ(tau, v, alphaZ, z, Z1, Z2):
            return ((1 / sqrt(pi * v * alphaZ * tau)) *
                    (exp(-(z - Z1) ** 2.0 / (4.0 * v * alphaZ * tau))))

    else:
        def fZ(tau, v, alphaZ, z, Z1, Z2):
            return (erf((z - Z1) / (2.0 * (v * alphaZ * tau) ** 0.5)) -
                    erf((z - Z2) / (2.0 * (v * alphaZ * tau) ** 0.5)))

    # Compute solution

    if t > tp:
        domenico = ((c0 / 8.0)*exp(-ks*t) *
                    (phiX(t, x, v, alphaX, k, ks, X1) -
                     phiX(t-tp, x, v, alphaX, k, ks, X1)) *
                    fY(tau, v, alphaY, y, Y1, Y2) *
                    fZ(tau, v, alphaZ, z, Z1, Z2))
    else:
        domenico = ((c0 / 8.0)*exp(-ks*t) *
                    phiX(t, x, v, alphaX, k, ks, X1) *
                    fY(tau, v, alphaY, y, Y1, Y2) *
                    fZ(tau, v, alphaZ, z, Z1, Z2))

    return domenico
