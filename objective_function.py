import numpy as np
# Note: models.py is imported in main.py, we only need the numpy here.

# The kinetic_model function will be passed from main.py
def objective_function(pari, xexp, rate_exp, kinetic_model):
    """
    Calculates the objective function (Root Mean Square Deviation, RMSD).
    f = sqrt(obj/vs), where obj = sum((1 - rate_calc / rate_exp)^2)

    :param pari: Current model parameters.
    :param xexp: Experimental independent variables.
    :param rate_exp: Experimental reaction rates.
    :param kinetic_model: The function corresponding to the kinetic model (e.g., hougen_watson or mars_van_krevelen).
    :return: fval (the objective value to minimize), ratec (the calculated rates).
    """
    
    number = xexp.shape[0]
    ratec = np.zeros(number)
    
    for ii in range(number):
        # Pass one row at a time, keeping it as a 2D array
        x0 = xexp[ii:ii+1, :] 
        
        # Calculate the rate using the passed kinetic model function
        ratec[ii] = kinetic_model(pari, x0)
        
    # Calculate the sum of squared relative deviations (obj)
    vs = len(rate_exp)
    relative_deviation = 1 - (ratec / rate_exp)
    obj = np.sum(relative_deviation**2)
    
    # Calculate the RMSD (fval)
    fval = np.sqrt(obj / vs)
    
    return fval, ratec

# Wrapper function for scipy.optimize.minimize
def obj_wrapper(pari, xexp, rate_exp, kinetic_model):
    fval, _ = objective_function(pari, xexp, rate_exp, kinetic_model)
    return fval