##########################################
##  DEMO Version of the Nd CEF Analysis ##
##########################################

# 1) Import necessary libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField_utils as cef
from scipy.optimize import curve_fit
from lmfit.models import LinearModel, PseudoVoigtModel
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.optimize import shgo  # <-- SHGO import

mpl.rcParams['font.family'] = 'Arial'

#####################################################
## 1) Minimal Parameter Definitions & CF Analysis  ##
##    (Using literature values from L'Hôtel et al.)##
#####################################################

# Wybourne parameters in K (from L'Hôtel et al.)
B20_LH = -190
B40_LH = 5910
B43_LH = 110
B60_LH = 2810
B63_LH = 10
B66_LH = -890

# Convert K to meV, and apply Steven's factors (Demo: show partial calculation)
KtomeV = 1000/11600  # conversion factor
B20_LH = B20_LH * KtomeV * (-0.158/49.2)
B40_LH = B40_LH * KtomeV * (-0.0149/408.9)
B43_LH = B43_LH * KtomeV * (0.105/121.6)
B60_LH = B60_LH * KtomeV * (-0.00035/148.1)
B63_LH = B63_LH * KtomeV * (-0.0048/-98.0)
B66_LH = B66_LH * KtomeV * (-0.005/139.1)

# Put them in a dictionary
Bdictionary_LH = {
    'B20': B20_LH, 'B40': B40_LH, 'B43': B43_LH,
    'B60': B60_LH, 'B63': B63_LH, 'B66': B66_LH
}

# Create CFLevels object and diagonalize
NdCF1 = cef.CFLevels.Bdict('Nd3+', Bdict=Bdictionary_LH)
NdCF1.diagonalize()
print("Eigenvalues (demo):", np.around(NdCF1.eigenvalues, 2))

########################################################
## 2) Minimal Data Loading, Synthetic Example & Plot  ##
########################################################

# Generate some synthetic data (for demo purpose, here we are instead using simulated experimental data)
x_demo = np.linspace(0,120,200)
y_demo = np.exp(-(x_demo-30)**2/100) + 0.05*np.random.rand(len(x_demo))
z_demo = 0.1*np.ones_like(x_demo)

# Plot the synthetic data and mock CF peaks
plt.figure(figsize=(6,3))
plt.errorbar(x_demo, y_demo, z_demo, marker='.', color='green', ls='none',
             capsize=2, label='demo data')
plt.bar(NdCF1.eigenvalues,
        5*np.ones(len(NdCF1.eigenvalues)),
        color='r', width=1, label="demo CF peaks")
plt.xlim(0, 60)
plt.title("Demo: CF Levels & Synthetic Data")
plt.legend()
plt.show()

#########################################################
## 3) Background Subtraction & Comparison to Theory    ##
#########################################################

# Simple placeholder for background subtraction
def subtract_background(intensity, background_level=0.05):
    return intensity - background_level

# Apply it to the synthetic data
y_demo_sub = subtract_background(y_demo)

# Mock theoretical spectrum for comparison
theory_spectrum = np.exp(-(x_demo - 32)**2 / 150)  # purely illustrative

plt.figure(figsize=(6,3))
plt.plot(x_demo, y_demo_sub, 'g.-', label='data (background sub)')
plt.plot(x_demo, theory_spectrum, 'b-', label='mock theory')
plt.title("Demo Comparison: Data vs. Mock Theory")
plt.legend()
plt.show()

######################################################
## 4) Fit to Identify Peak Positions and Widths     ##
######################################################

# Example polynomial fit
f_sub = np.polyfit(x_demo, y_demo_sub, 4)  # polynomial order 4
p_sub = np.poly1d(f_sub)

plt.figure(figsize=(6,3))
plt.plot(x_demo, y_demo_sub, 'g.-', label='data (bg sub)')
plt.plot(x_demo, p_sub(x_demo), 'r-', label='polynomial fit')
plt.title("Demo Polynomial Fit (After Background Subtraction)")
plt.legend()
plt.show()

# Example Pseudo-Voigt fit for a single peak
pseudo_model = PseudoVoigtModel()
params = pseudo_model.guess(y_demo_sub, x=x_demo)
result = pseudo_model.fit(y_demo_sub, params, x=x_demo)
print("\nDemo Pseudo-Voigt fit results:")
print(result.fit_report())

#########################################################
## 5) Iterative Optimization of CF Parameters (SHGO)   ##
#########################################################

def objective_function(b20):
    """
    Example objective function that, for a given B20,
    re-diagonalizes the CF levels and computes chi-squared 
    vs. the data. In reality, we'd recast Bdict, recalc
    the theoretical spectrum, and compare with data.
    """
    # Copy the original dictionary for demonstration
    Bdict_temp = Bdictionary_LH.copy()
    Bdict_temp['B20'] = b20
    
    # Re-diagonalize with the new B20
    temp_CF = cef.CFLevels.Bdict('Nd3+', Bdict=Bdict_temp)
    temp_CF.diagonalize()
    
    # Generate a mock "theoretical" spectrum and compare to data
    # (purely synthetic for demonstration)
    theory_temp = np.exp(-(x_demo - 30 - b20)**2 / 100)
    
    # Simple chi-square calculation for demonstration
    chi_sq = np.sum(((y_demo_sub - theory_temp) / z_demo)**2)
    return chi_sq

# Use SHGO with a small range for the B20 parameter
res = shgo(
    func=objective_function,
    bounds=[(-0.5*abs(Bdictionary_LH['B20']), 0.5*abs(Bdictionary_LH['B20']))],
    n=20  # small number for demonstration
)
print("\nSHGO demo optimization result for B20:", res.x)
print("Minimum chi-square found:", res.fun)

###########################################################
## 6) Final Visualizations & Relation of B-parameters    ##
###########################################################

# Sample multiple B20 values around the optimum to visualize
b20_values = np.linspace(-0.3, 0.3, 30) * abs(Bdictionary_LH['B20'])
chisq_values = [objective_function(val) for val in b20_values]

plt.figure(figsize=(6,3))
plt.plot(b20_values, chisq_values, 'o-')
plt.axvline(res.x[0], color='r', linestyle='--', label='Optimal B20')
plt.title("Demo: B20 vs. Chi-Squared")
plt.xlabel("B20 (meV, scaled)")
plt.ylabel("Chi-Squared")
plt.legend()
plt.show()
