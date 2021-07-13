
import scipy
from scipy import integrate
from math import erfc, erf, exp, sqrt, pi


def callWexler(x, y, z, t, v, alphaX, alphaY, alphaZ, k, ks, R, X1, Y1, Y2, Z1, Z2, c0, tp, Option):

    v = v / R
    k = k / R
    if Option['boundary'] == 'Dirichlet':
        def fX(tau, x, v, alphaX, k, ks, X1):
            return ((x - X1) / (pi * v * alphaX) ** 0.5 *
                    exp((x - X1) / (2.0 * alphaX)) *
                    exp(-v * tau / (4.0 * alphaX) - (x - X1) ** 2.0 /
                        (4.0 * v * alphaX * tau)-(k-ks)*tau) / tau ** 1.5)

    elif Option['boundary'] == 'Cauchy':
        def fX(tau, x, v, alphaX, k, ks, X1):
            return (v*tau/(pi*alphaX*v)**0.5 *
                    exp((x-X1)/(2.0*alphaX)) *
                    exp(-v*tau/(4.0*alphaX)-(x-X1) ** 2/(4.0*alphaX*v*tau)-(k-ks)*tau) *
                    tau**(-3.0/2.0)-v/(2.0*alphaX) *
                    exp((x-X1)/alphaX-(k-ks)*tau) *
                    erfc((x-X1+v*tau)/(2*(alphaX*v*tau)**0.5)))

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

    def fWexler(tau, x, v, alphaX, k, ks, X1, t, alphaY, y, Y1, Y2, alphaZ, z, Z1, Z2, c0):
        return (c0 / 8 * exp(-ks*t) *
                fX(tau, x, v, alphaX, k, ks, X1) *
                fY(tau, v, alphaY, y, Y1, Y2) *
                fZ(tau, v, alphaZ, z, Z1, Z2))

    # Calculation
    if t > tp:
        wexler = integrate.quad(lambda tau: fWexler(tau, x, v, alphaX, k, ks, X1, t, alphaY, y, Y1, Y2,
                                alphaZ, z, Z1, Z2, c0), t-tp, t)[0]    # Note tau should be a vector in the integral function
    else:
        wexler = integrate.quad(lambda tau: fWexler(
            tau, x, v, alphaX, k, ks, X1, t, alphaY, y, Y1, Y2, alphaZ, z, Z1, Z2, c0), 0, t)[0]

    return wexler
