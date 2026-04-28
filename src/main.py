import eval
import util
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --------------------------------
# DATA IMPORT
# --------------------------------

row1  = [18, 18, 19, 19.2, 15, 19.2, 16, 18.5, 19, 19, 17.5, 19, 19]
row2  = [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70]
row3  = [22, 45, 68, 40, 95, 56, 125, 56, 38, 25, 95, 60, 60]
row4  = [120, 120, 120, 120, 120, 120, 120, 120, 120, 150, 120, 120, 120]
row5  = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
row6  = [1, 1, 1, 2, 2, 2, 2, 2, 3, 1, 4, 4, 4]
row7  = [6.90E-06, 4.90E-06, 6.30E-06, 7.50E-06, 6.90E-06, 6.10E-06, 4.80E-06, 6.20E-06, 6.30E-06, 2.80E-05, 5.50E-06, 5.40E-06, 5.60E-06]
row8  = [9.1E-5, 7.10E-05, 6.5E-05, 7.10E-05, 7.10E-05, 7.00E-05, 7.00E-05, 7.50E-05, 8.00E-05, 8.00E-05, 8.00E-05, 8.00E-05, 8.00E-05]
row9  = [3.8, 3.6, 3.8, 3.8, 3.8, 3.8, 4, 4, 4, 4, 4, 4, 4]
row10 = [1, 2, 2, 3, 4, 4, 5, 5, 7, 8, 2, 2, 2]
row11 = [1300, 1100, 900, 750, 230, 2.3, 2.5, 3, 3, 3, 3, 3, 3]
row12 = [10, 2, 8, 3, 1, 3, 2, 5, 9, 9, 9, 9, 9]

# Separate into categorized (g), continuous (z), and response (y)
g_data = np.array([row1, row2, row3, row4, row5, row6, row9, row10])
z_data = np.array([row7, row8])
y_data = np.array(row11)

l_g = g_data.shape[0]  # Number of categorical variables
l_z = 3  # 2 continuous variables + 1 intercept
n = 1    # Response dimension
w = 1.0 / l_g  # Uniform weighting for mixing

se = True

# --------------------------------
# STRUCTURE ESTIMATION (optional)
# --------------------------------
if se:
    elimination_technique = ['global', 'forward', 'backward'] # backward is RECOMMENDED
    optimal_rows, best_ll = util.elimination(y_data, z_data, g_data, elimination_technique[2])
    optimal_g_data = g_data[optimal_rows, :]
else:
    optimal_g_data = g_data

# --------------------------------
# LOO (leave-one-out)
# --------------------------------

preds, rmse, mae, log_like = util.perform_loo_cv(
    y_data,
    z_data,
    optimal_g_data,
    mixing_method='forecast_mixing',
    solver_method='analytical',
    opt_prior_phi=False,
    verbose=False,
)

mean_preds, ols_preds = util.baseline_loo_cv(y_data, z_data)

eval.plot_loo_validation(y_data, preds, title_suffix="(Pooling)")
print("--------------------------------")
eval.plot_loo_validation(y_data, mean_preds, title_suffix="(Mean Baseline)")
eval.plot_loo_validation(y_data, ols_preds, title_suffix="(OLS Baseline)")

target_response = 450.0
z_current = np.array([1.0, 6.5e-06, 7.1e-05]) # Example current regressor

# --------------------------------
# OPTIMAL SETTINGS FINDER
# --------------------------------

# optimal_g, predicted_y = util.find_optimal_g(
#     target_response,
#     z_current,
#     trained_models,
#     l_g,
#     mixing_method='distribution_mixing'
# )

