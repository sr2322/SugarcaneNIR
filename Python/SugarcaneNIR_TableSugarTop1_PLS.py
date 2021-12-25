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

def pls_variable_selection(X, y, max_comp):
    
    # Define MSE array to be populated
    mse = np.zeros((max_comp,X.shape[1]))
    # Loop over the number of PLS components
    for i in range(max_comp):
        
        # Regression with specified number of components, using full spectrum
        pls1 = PLSRegression(n_components=i+1)
        pls1.fit(X, y)
        
        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_ind = np.argsort(np.abs(pls1.coef_[:,0]))
        # Sort spectra accordingly 
        Xc = X[:,sorted_ind]
        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        for j in range(Xc.shape[1]-(i+1)):
            pls2 = PLSRegression(n_components=i+1)
            pls2.fit(Xc[:, j:], y)
            
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5)
            mse[i,j] = mean_squared_error(y, y_cv)
    
        comp = 100*(i+1)/(max_comp)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    # # Calculate and print the position of minimum in MSE
    mseminx,mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))
    print("Optimised number of PLS components: ", mseminx[0]+1)
    print("Wavelengths to be discarded ",mseminy[0])
    print('Optimised MSEP ', mse[mseminx,mseminy][0])
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.show()
    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=mseminx[0]+1)
    pls.fit(X, y)
        
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
    Xc = X[:,sorted_ind]
    return(Xc[:,mseminy[0]:],mseminx[0]+1,mseminy[0], sorted_ind)

def simple_pls_cv(X, y, n_comp):
    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y)
    y_c = pls.predict(X)
    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)    
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    # Plot regression 
    z = np.polyfit(y, y_cv, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y, y_cv, c='red', edgecolors='k')
        ax.plot(y, z[1]+z[0]*y, c='blue', linewidth=1)
        print("Equation: " + str(z[0]) + "s + " + str(z[1]))
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(score_cv))
        plt.ylabel('Predicted $^{\circ}$Brix')
        plt.xlabel('Actual $^{\circ}$Brix')
        
        plt.show()

# Load data
url = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugarTop1_nozeros.csv'
data = pd.read_csv(url, sep=',')
X = data.values[:,1:]
Xmsc = data.values[:,1:]

wl_str = data.columns[1:] 
wl = [float(numeric_string) for numeric_string in wl_str]


# Load reference data (to get total sugar values)
ref_url = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugarTop1_nozeros_Refs.csv'
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
X1 = savgol_filter(X, 55, polyorder = 3, deriv=0)

opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X1, y, 15)
simple_pls_cv(opt_Xc, y, ncomp)

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

# Get a boolean array according to the indices that are being discarded
wl = np.array(wl)
ix = np.in1d(wl.ravel(), wl[sorted_ind][:wav])
import matplotlib.collections as collections
# Plot spectra with superimpose selected bands
fig, ax = plt.subplots(figsize=(8,9))
with plt.style.context(('ggplot')):
    for X1_i, TS_i in zip(X1,y):
        if TS_i != 0:
            ax.plot(wl, X1_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    plt.ylabel('First derivative absorbance spectra')
    plt.xlabel('Wavelength (nm)')
    ax.set_facecolor('lightgray')
collection = collections.BrokenBarHCollection.span_where(
    wl, ymin=-1, ymax=1, where=ix == True, facecolor='black', alpha=0.5)
ax.add_collection(collection)
plt.show()

