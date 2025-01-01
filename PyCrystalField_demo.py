##########################################
##  DEMO Version of the Nd CEF Analysis ##
##########################################

# 1) Import necessary libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
from lmfit.models import LinearModel, PseudoVoigtModel
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

#####################################################
## 1) Minimal Parameter Definitions & CF Analysis  ##
#####################################################

# Wybourne parameters in K (from L'hotel et al.)
B20_LH = -190
B40_LH = 5910
B43_LH = 110
B60_LH = 2810
B63_LH = 10
B66_LH = -890

# Convert K to meV, and apply Steven's factors (Demo: show partial calculation)
KtomeV = 1000/11600  # conversion factor
B20_LH = B20_LH*KtomeV*(-0.158/49.2)
B40_LH = B40_LH*KtomeV*(-0.0149/408.9)
B43_LH = B43_LH*KtomeV*(0.105/121.6)
B60_LH = B60_LH*KtomeV*(-0.00035/148.1)
B63_LH = B63_LH*KtomeV*(-0.0048/-98.0)
B66_LH = B66_LH*KtomeV*(-0.005/139.1)

# Put them in a dictionary
Bdictionary_LH = {
    'B20': B20_LH, 'B40': B40_LH, 'B43': B43_LH,
    'B60': B60_LH, 'B63': B63_LH, 'B66': B66_LH
}

# Create CFLevels object and diagonalize
NdCF1 = cef.CFLevels.Bdict('Nd3+', Bdict=Bdictionary_LH)
NdCF1.diagonalize()
print("Eigenvalues (demo):", np.around(NdCF1.eigenvalues, 2))

####################################################
## 2) Minimal Data Loading & Quick Plot (Demo)    ##
####################################################

# (In the real notebook, we loaded dt_cropped_x,y,z from a file.)
# For the DEMO, let's just generate some synthetic data:
x_demo = np.linspace(0,120,200)
y_demo = np.exp(-(x_demo-30)**2/100) + 0.05*np.random.rand(len(x_demo))
z_demo = 0.1*np.ones_like(x_demo)

plt.figure(figsize=(6,3))
plt.errorbar(x_demo, y_demo, z_demo, marker='.', color='green', ls='none', capsize=2, label='demo data')
plt.bar(NdCF1.eigenvalues, 5*np.ones(len(NdCF1.eigenvalues)), color='r', width=1, label="demo CF peaks")
plt.xlim(0, 60)
plt.title("Demo: CF Levels & Synthetic Data")
plt.legend()
plt.show()

####################################################
## 3) Minimal Fit Demo (Polynomial as Example)     ##
####################################################

from sklearn.linear_model import LinearRegression
import pandas as pd

# (In the real notebook, we used actual columns from data files.)
# Let's wrap synthetic data into a small dataframe
df_demo = pd.DataFrame({"Energy (meV)": x_demo,
                        "Intensity (a.u.)": y_demo,
                        "Intensity Uncertainty (a.u.)": z_demo})

lm = LinearRegression()
X_demo = df_demo[["Energy (meV)"]]
Y_demo = df_demo["Intensity (a.u.)"]
lm.fit(X_demo, Y_demo)

print("Demo linear fit slope:", lm.coef_)
print("Demo linear fit intercept:", lm.intercept_)

############################################################
## 4) Show a quick polynomial fit on the synthetic data   ##
############################################################

f = np.polyfit(x_demo, y_demo, 4)  # lower polynomial order for demo
p = np.poly1d(f)
print("Demo polyfit coefficients:\n", p)

plt.figure(figsize=(6,3))
plt.errorbar(x_demo, y_demo, z_demo, marker='.', ls='none', color='green', label='demo data')
plt.plot(x_demo, p(x_demo), 'r-', label='demo poly fit')
plt.title("Demo Polynomial Fit")
plt.legend()
plt.show()

##############################################
## 5) Demo End: Minimal Notebook Workflow   ##
##############################################

print("\nNotebook demo complete. In a real notebook, further sections would follow.")
