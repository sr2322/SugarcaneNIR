# -*- coding: utf-8 -*-
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import nan

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure

def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
    # Baseline correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()

    # Get the reference spectrum. If not given, estimate from the mean    
    if reference is None:    
        # Calculate mean
        matm = np.mean(input_data, axis=0)
    else:
        matm = reference

    # Define a new data matrix and populate it with the corrected data    
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(matm, input_data[i,:], 1, full=True)
        # Apply correction
        output_data[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 

    return (output_data, matm)


# Load data
url = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugarLit1.csv'
data = pd.read_csv(url, sep=',')
X = data.values[:,1:]
Xmsc = data.values[:,1:]

wl_str = data.columns[1:] 
wl = [float(numeric_string) for numeric_string in wl_str]


# Load reference data (to get total sugar values)
ref_url = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugarLit1_Refs.csv'
ref_data = pd.read_csv(ref_url, sep=',')
SampleTSraw = ref_data.values[:,1]
y = []
for s in SampleTSraw:
    test_s = str(s)
    if test_s[0].isnumeric():
        y.append(float(s))
    else:
        y.append(float(0.00))
TS_max = max(y)

# Apply correction
# X1 = msc(X)[0] # Take the first element of the output tuple
X2 = savgol_filter(X, 15, polyorder = 3, deriv=0)

# Plot second derivative
plt.figure(figsize=(8,4.5))
with plt.style.context(('ggplot')):
    plt.plot(wl, X2.T)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('D2 Absorbance')
    plt.show()
    
def optimise_pls_cv(X, y, n_comp, plot_components=True):

    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''

    mse = []
    component = np.arange(1, n_comp)

    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=9)

        mse.append(mean_squared_error(y, y_cv))

        comp = 100*(i+1)/n_comp
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)
            plt.xticks(range(0,20))

        plt.show()

    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin+1)

    # Fir to the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)

    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=9)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)

    # Plot regression and figures of merit
    rangey = max(y) - min(y)
    rangex = max(y_c) - min(y_c)

    # Fit a line to the CV vs response
    z = np.polyfit(y, y_c, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y, y_c, c='red', edgecolors='k')
        #Plot the best fit line
        ax.plot(y, np.polyval(z,y), c='blue', linewidth=1)
        #Plot the ideal 1:1 line
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(score_cv))
        plt.ylabel('Predicted $^{\circ}$Brix')
        plt.xlabel('Measured $^{\circ}$Brix')

        plt.show()

    return

optimise_pls_cv(X2,y, 20, plot_components=True)

# # COMMENT THIS IN/OUT ALTERNATELY WITH ABOVE BLOCK
# # Define the PLS regression object
# pls = PLSRegression(n_components=8)
# # Fit data
# pls.fit(X1, y)
# # Plot spectra
# plt.figure(figsize=(8,9))
# with plt.style.context(('ggplot')):
#     ax1 = plt.subplot(211)
#     for X1_i, TS_i in zip(X1,y):
#         if TS_i != 0:
#             ax1.plot(wl, X1_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
#     plt.ylabel('First derivative absorbance spectra')
#     ax2 = plt.subplot(212, sharex=ax1)
#     plt.plot(wl, np.abs(pls.coef_[:,0]))
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Absolute value of PLS coefficients')
#     plt.show()
