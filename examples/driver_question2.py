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
    t1 = np.linspace(t0, tf, num = 21)
    # print(f"t1: {t1}")
    t2 = np.linspace(t0, tf, num = 11)
    # print(f"t2: {t2}")
    vt1 = v(t1)
    vt2 = v(t2)

    # integrate equation 3 (2.a and b)
    Itrap1 = integrate_newton(t1, vt1, alg= "trap")
    Itrap2 = integrate_newton(t2, vt2, alg= "trap")
    Isimp1 = integrate_newton(t1, vt1, alg= "simp")
    Isimp2 = integrate_newton(t2, vt2, alg= "simp")

    
    print(f"Integral Approximation using trapezoid rule with a stepsize of 1s: {Itrap1}")
    print(f"Integral Approximation using trapezoid rule with a stepsize of 2s: {Itrap2}\n")

    print(f"Integral Approximation using simpson's rule with a stepsize of 1s: {Isimp1}")
    print(f"Integral Approximation using simpson's rule with a stepsize of 2s: {Isimp2}\n")

    # Question 2.c) Integrate Results using 5-point Gauss-Legendre
    Igauss = integrate_gauss(v, lims = [0.0, 20.0], npts = 5)

    print(f"Integral Approximation using 5-point Gauss-Legendre Quadrature: {Igauss}")

    


if __name__ == "__main__":
    main()
