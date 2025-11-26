import numpy as np

def load_data(selection='ALL'):
    """
    Loads experimental data based on the requested subset.
    
    :param selection: 'MATLAB' (Original 8 points), 'TABLE' (New 8 points), or 'ALL' (16 points).
    
    Returns:
        tuple: (xexp, rate_exp, temp_exp, dataset_mask) 
               dataset_mask is only returned for 'ALL' and is a boolean array 
               where True=MATLAB data, False=TABLE data.
    """
    
    # --- 1. Original MATLAB Data (8 points) ---
    temp_exp_matlab = np.full(8, 783.0) 
    xexp_matlab = np.array([
        [10.349, 7.878], [4.063, 8.220], [4.897, 10.354], [5.331, 3.599], 
        [8.332, 3.829], [6.401, 5.351], [3.305, 6.796], [6.411, 13.135]
    ])
    rate_exp_matlab = np.array([
        0.020476014, 0.009821433, 0.012133934, 0.012740631, 
        0.017745879, 0.014132819, 0.008593754, 0.01456072
    ])

    
    # --- 2. New Table Data (8 points) ---
    temp_exp_table = np.array([
        783, 783, 783, 783, 813, 843, 883, 933
    ])
    xexp_table = np.array([
        [73.1, 0.0994], [41.7, 0.0997], [29.1, 0.0996], [41.7, 0.0997], 
        [41.7, 0.0997], [41.7, 0.0997], [41.7, 0.0997], [41.7, 0.0997]
    ])
    rate_exp_table = np.array([
        0.0607, 0.0752, 0.0685, 0.0470, 0.0529, 0.0578, 0.0664, 0.0710
    ])
    
    # --- 3. Selection Logic ---
    if selection == 'MATLAB':
        # Return None for the mask when a single set is selected
        return xexp_matlab, rate_exp_matlab, temp_exp_matlab, None
    
    elif selection == 'TABLE':
        return xexp_table, rate_exp_table, temp_exp_table, None
    
    elif selection == 'ALL':
        xexp = np.concatenate((xexp_matlab, xexp_table), axis=0)
        rate_exp = np.concatenate((rate_exp_matlab, rate_exp_table))
        temp_exp = np.concatenate((temp_exp_matlab, temp_exp_table))
        
        # Create a boolean mask: True for MATLAB data, False for TABLE data
        matlab_mask = np.concatenate((np.full(len(xexp_matlab), True), np.full(len(xexp_table), False)))
        return xexp, rate_exp, temp_exp, matlab_mask
    
    else:
        raise ValueError("Invalid dataset selection. Use 'MATLAB', 'TABLE', or 'ALL'.")

if __name__ == '__main__':
    # Simple test to verify data loading
    x, r, t, m = load_data('ALL')
    print(f"Loaded {len(r)} total data points from ALL.")
    print(f"Matlab points: {np.sum(m)}, Table points: {np.sum(~m)}")
