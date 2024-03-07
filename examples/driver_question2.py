# GOPH420 - Inv & Param Estimation for Geophysicists
# Midterm 1 
# Questions 2 - Numerical Integration

import numpy as np
from midterm1.integration import (integrate_newton, integrate_gauss)

def main():
    # define/initialize variables
    v0 = 100
    c = 47.0
    g = 3.718
    m = 93.0
    vf = g * m / c

    # initialize equation 3 (velocity function)
    v = lambda t: vf + (v0 - vf) * np.exp(-(c / m) * t)
    
    # set integration limits (time in s)
    t0 = 0.0
    tf = 20.0

    # create list of data
    t = np.linspace(t0, tf, num = 100)
    vt = v(t)

    # set stepsizes of 1 and 2 s
    steps = [1, 2]

    # initialize lists for integrals for different stepsizes
    int_trap = []
    int_simp = []

    # integrate equation 3 (2.a and b)
    for step in steps:
        Itrap = integrate_newton(t[::step], vt[::step], alg= "trap")
        Isimp = integrate_newton(t[::step], vt[::step], alg= "simp")
        int_trap.append(Itrap)
        int_simp.append(Isimp)
    
    print(f"Integral Approximation using trapezoid rule with a stepsize of 1s: {int_trap[0]}")
    print(f"Integral Approximation using trapezoid rule with a stepsize of 2s: {int_trap[1]}\n")

    print(f"Integral Approximation using simpson's rule with a stepsize of 1s: {int_simp[0]}")
    print(f"Integral Approximation using simpson's rule with a stepsize of 2s: {int_simp[1]}\n")

    # Question 2.c) Integrate Results using 5-point Gauss-Legendre
    Igauss = integrate_gauss(v, lims = [0.0, 20.0], npts = 5)

    print(f"Integral Approximation using 5-point Gauss-Legendre Quadrature: {Igauss}")

    


if __name__ == "__main__":
    main()
