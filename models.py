import numpy as np

def hougen_watson(parm, x):
    """
    Model (FT3 RDS-10): yhat = ((b1*b2*b3) * x1 * x2^0.5) / (1 + b2*x1 + b3*x2^0.5)^2
    :param parm: Model parameters [b1, b2, b3].
    :param x: Independent variables (N x 2 array): x[:, 0] is x1, x[:, 1] is x2.
    :return: Calculated rate (yhat).
    """
    b1, b2, b3 = parm[0], parm[1], parm[2]
    x1 = x[:, 0]
    x2 = x[:, 1]

    # Rate function calculation
    numerator = (b1 * b2 * b3) * x1 * np.sqrt(x2)
    denominator = (1 + b2 * x1 + b3 * np.sqrt(x2))**2
    yhat = numerator / denominator

    # Return scalar if a single point was passed
    if x.shape[0] == 1:
        return yhat[0]
    else:
        return yhat

def mars_van_krevelen(parm, x):
    """
    Model 1: MvK Redox (without water inhibition for parameter consistency)
    r = (k_r * k_o * P_MeOH * P_O2^0.5) / (k_r * P_MeOH + k_o * P_O2^0.5)
    
    Mapping to variables:
    P_MeOH ~ x1 (x[:, 0])
    P_O2   ~ x2 (x[:, 1])
    k_r    ~ parm[0]
    k_o    ~ parm[1]
    
    :param parm: Model parameters [k_r, k_o].
    :param x: Independent variables (N x 2 array): P_MeOH, P_O2.
    :return: Calculated rate (yhat).
    """
    # Assuming the first two parameters are k_r and k_o
    kr, ko = parm[0], parm[1]
    P_MeOH = x[:, 0]
    P_O2 = x[:, 1]

    # Rate function calculation
    numerator = kr * ko * P_MeOH * np.sqrt(P_O2)
    denominator = kr * P_MeOH + ko * np.sqrt(P_O2)
    
    # Handle division by zero for robust optimization
    yhat = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    # Return scalar if a single point was passed
    if x.shape[0] == 1:
        return yhat[0]
    else:
        return yhat