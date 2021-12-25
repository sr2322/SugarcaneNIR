# -*- coding: utf-8 -*-
'''
Takes in a CSV (really semicolon separated) file containing spectral absorbance data for several samples
Applies scatter correction: Standard Normal Variate (SNV) or Multiplicative Scatter Correction (MSC). SNV better for NIR.
Applies filtering: Savitzky-Golay (Savgol) 
Performs linear regression (Absorbance vs Total Sugar Content) at each wavelength and calculates R^2
Displays raw spectra (top), corrected + filtered spectra (middle), and then R^2 values (bottom), all vs wavelength
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter
from scipy import stats

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

def snv(input_data):
    ''' Perform Standard Normal Variate correction'''
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):

        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])

    return output_data

# Load data
url = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/Chaix_Dataset/F750.CSV'
data = pd.read_csv(url, sep=';')
X = data.values[:,1:]
X_corrected = data.values[:,1:]

wl_str = data.columns[1:] 
wl = [float(numeric_string) for numeric_string in wl_str]

# Load reference data (to get total sugar values)
ref_url = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main//Chaix_Dataset/References.csv'
ref_data = pd.read_csv(ref_url, sep=';')
SampleTSraw = ref_data.values[:,1]
SampleTSes = []
for s in SampleTSraw:
    test_s = str(s)
    if test_s[0].isnumeric():
        SampleTSes.append(float(s))
    else:
        SampleTSes.append(float(0.00))
TS_max = max(SampleTSes)

# Apply correction
X_corrected = snv(X_corrected) # Take the first element of the output tuple
X_corrected = savgol_filter(X_corrected, 35, polyorder = 2, deriv = 0)

linregcoeffs_corrected = []
Xs_corrected = pd.DataFrame(data=X_corrected, index=SampleTSes, columns=wl_str)

for wl_str_i in wl_str:
    Column_Xs_corrected  = Xs_corrected[wl_str_i].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(SampleTSes,Column_Xs_corrected)
    linregcoeffs_corrected.append(round(r_value,2))
#linregcoeffs_msc = np.array(linregcoeffs_msc)


## Plot original and corrected spectra
plt.figure(figsize=(24,27))
with plt.style.context(('ggplot')):
    ax1 = plt.subplot(311)
    for X_i, TS_i in zip(X,SampleTSes):
        plt.plot(wl, X_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    # plt.plot(wl, X.T)
    plt.title('Original data')
    
    ax2 = plt.subplot(312)
    for X_corrected_i, TS_i in zip(X_corrected,SampleTSes):
        plt.plot(wl, X_corrected_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    # plt.plot(wl, X_corrected.T)
    plt.ylabel('Absorbance spectra')
    plt.title('SNV')
    
    ax2 = plt.subplot(313)
    plt.plot(wl,linregcoeffs_corrected)
    
    # plt.plot(wl, Xsnv.T)
    plt.xlabel('Wavelength (nm)')
    
    plt.show()


