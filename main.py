import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import all necessary modules
from objective_function import objective_function, obj_wrapper
from models import hougen_watson, mars_van_krevelen
from dataset import load_data 

# --- MODEL SELECTION ---
# Set this variable to 'HW' or 'MVK' to choose which model to optimize.
# 'HW' (Hougen-Watson) uses 3 parameters.
# 'MVK' (Mars-van Krevelen) uses 2 parameters.
MODEL_TO_RUN = 'MVK' 
# -----------------------

# --- DATASET SELECTION ---
# Options: 'MATLAB' (Original 8 points), 'TABLE' (New 8 points), or 'ALL' (16 points)
DATASET_TO_USE = 'MATLAB' 
# -------------------------

def get_model_params(model_name):
    """Defines parameters, bounds, and the function based on the model name."""
    
    if model_name == 'HW':
        print(f"**Running Hougen-Watson Model (3 parameters: b1, b2, b3)**")
        # Initial guess and bounds for Hougen-Watson
        par0 = np.array([0.034616542, 4.354892129, 14.80747964])
        parlb = np.array([0.005, 5, 14.80747964])
        parub = np.array([0.05, 7, 20])
        kinetic_model = hougen_watson
    
    elif model_name == 'MVK':
        print(f"**Running Mars-van Krevelen Model (2 parameters: kr, ko)**")
        # Initial guess and bounds for MvK
        par0 = np.array([0.1, 15.0]) 
        
        # --- MODIFIED ---
        # Give the optimizer more freedom. 1e-6 is (0.000001)
        parlb = np.array([1e-6, 1e-6]) 
        parub = np.array([1000.0, 1000.0]) # Let it search up to 1000
        # --- END MODIFIED ---

        kinetic_model = mars_van_krevelen
        
    else:
        raise ValueError("Invalid MODEL_TO_RUN. Choose 'HW' or 'MVK'.")

    bounds = list(zip(parlb, parub))
    return par0, bounds, kinetic_model

def main():
    print(f"🚀 Starting Parameter Estimation. Selected Model: {MODEL_TO_RUN}")
    
    # --- 1. Load Data ---
    # Load data, including the mask if 'ALL' is selected
    xexp, rate_exp, temp_exp, matlab_mask = load_data(DATASET_TO_USE)
    
    if DATASET_TO_USE == 'ALL':
        num_points_matlab = np.sum(matlab_mask)
        num_points_table = np.sum(~matlab_mask)
        print(f"\nSuccessfully loaded {len(rate_exp)} total data points: "
              f"{num_points_matlab} (MATLAB) + {num_points_table} (TABLE).")
    else:
        print(f"\nSuccessfully loaded {len(rate_exp)} data points from dataset: '{DATASET_TO_USE}'.")
    
    # --- 2. Initial Setup ---
    par0, bounds, kinetic_model = get_model_params(MODEL_TO_RUN)

    # --- 3. Optimization ---
    print("\nStarting fmincon equivalent (minimize with SLSQP)...")
    
    result = minimize(
        obj_wrapper,
        par0,
        args=(xexp, rate_exp, kinetic_model),
        method='SLSQP',
        bounds=bounds,
        options={'disp': True, 'ftol': 1e-9, 'maxiter': 9000}
    )

    # --- 4. Results and Plotting ---
    pari = result.x
    fval = result.fun
    
    # Calculate the final ratec
    _, ratec = objective_function(pari, xexp, rate_exp, kinetic_model)

    print("\n" + "="*40)
    print("✅ Optimization Completed")
    print("="*40)
    print(f"Optimal Parameters (pari): {pari}")
    print(f"Final Objective Value (fval): {fval}")
    print("\nModel Rates (ratec):")
    print(ratec)
    
    # Plotting
    try:
        plt.figure(figsize=(7, 7))
        
        # Determine the min/max values for the perfect fit line
        min_val = min(min(ratec), min(rate_exp)) * 0.9
        max_val = max(max(ratec), max(rate_exp)) * 1.1

        if DATASET_TO_USE == 'ALL':
            # Split data using the mask for plotting with different colors/markers
            ratec_matlab = ratec[matlab_mask]
            rate_exp_matlab = rate_exp[matlab_mask]
            
            ratec_table = ratec[~matlab_mask]
            rate_exp_table = rate_exp[~matlab_mask]
            
            # Plot the two subsets with different colors
            plt.plot(ratec_matlab, rate_exp_matlab, 'bo', 
                     label=f'MATLAB Data ({len(ratec_matlab)} points)')
            plt.plot(ratec_table, rate_exp_table, 'gs', 
                     label=f'TABLE Data ({len(ratec_table)} points)')
        else:
            # Plot all points uniformly if not using 'ALL'
            plt.plot(ratec, rate_exp, 'bo', label=f'Data Points ({len(rate_exp)})')


        # Plot the perfect fit line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Fit Line')
        
        plt.xlabel('Calculated Rate (ratec)')
        plt.ylabel('Experimental Rate (rate)')
        plt.title(f'Parity Plot: {MODEL_TO_RUN} Model on {DATASET_TO_USE} Data')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    except ImportError:
        print("\nMatplotlib is not installed. Skipping plot.")

if __name__ == "__main__":
    main()
