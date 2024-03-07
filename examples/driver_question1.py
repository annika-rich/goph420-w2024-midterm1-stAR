# GOPH420 - Inv & Param Estimation for Geophysicists
# Midterm 1 
# Questions 1.b) Computing Relative Error
import numpy as np 

def main():
    """Driver Script to compute relative error in each variable with a measure of uncertainty,
        the total error in H, relative error in H, and the true value of Stefan-Boltzmann's Law
        for the given values.
    """
    sigma = 5.67e-8

    # i.) Relative Error in Radius, R
    R = 6.371e6
    del_R = 0.021e6
    error_R = del_R / R
    print(f"The relative error in the Earth's Radius (R): {error_R:0.5e}")

    # ii.) Relative error in emissivity, e
    e = 0.612
    del_e = 0.015
    error_e = del_e / e
    print(f"The relative error in average emissivity (e): {error_e:0.5e}")

    # iii.) Relative error in temperature T
    T = 285
    del_T = 5
    error_T = del_T / T
    print(f"The relative error in the average temperature (T): {error_T:0.5e}\n")

    # absolute values of partial derivatives of each variable
    partial_R = np.abs(8 * np.pi * R * e * sigma * T ** 4)
    partial_e = np.abs(4 * np.pi * R ** 2 * sigma * T ** 4)
    partial_T = np.abs(16 * np.pi * R ** 2 * e * sigma * T ** 3)

    # contribution of each variable to total error
    contr_R = partial_R * del_R
    contr_e = partial_e * del_e
    contr_T = partial_T * del_T
    # use these to calculate total error in H
    del_H = contr_R + contr_e + contr_T

    # iv.)
    # 0-th Order Taylor Series (this term does not account for uncertainty/floating point error)
    H_bar = 4 * np.pi * R ** 2 * e * sigma * T ** 4
    # The expected value of the heat energy H for the first order Taylor Series:
    print(f"The expected value of heat energy (W): {H_bar:0.5e}")

    # v.)  Th total error in H
    print(f"The total error in heat energy (H): {del_H:0.5e}\n")
    print(f"Error contribution from Earth's Radius (R): {contr_R:0.5e}")
    print(f"Error contribution from average emissivity (e): {contr_e:0.5e}")
    print(f"Error contribution from average Temperature (T): {contr_T:0.5e}\n")

    # vi. Relative error in H (use 1st order Tayler Series Sum as best approximation)
    rel_H = np.abs((del_H) / H_bar)
    print(f"The relative error in H: {rel_H:0.5e}")



if __name__ == "__main__":
    main()