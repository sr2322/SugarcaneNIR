# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

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
url = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/F750.CSV'
data = pd.read_csv(url, sep=';')
X = data.values[:,1:]
wl_str = data.columns[1:] 
wl = [float(numeric_string) for numeric_string in wl_str]

# Load reference data (to get total sugar values)
ref_url = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/References.csv'
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
Xmsc = msc(X)[0] # Take the first element of the output tuple
Xsnv = snv(X)

## Plot original and corrected spectra
plt.figure(figsize=(24,27))
with plt.style.context(('ggplot')):
    ax1 = plt.subplot(311)
    for X_i, TS_i in zip(X,SampleTSes):
        if TS_i != 0:
            plt.plot(wl, X_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    # plt.plot(wl, X.T)
    plt.title('Original data')
    
    ax2 = plt.subplot(312)
    for Xmsc_i, TS_i in zip(Xmsc,SampleTSes):
       if TS_i != 0:
            plt.plot(wl, Xmsc_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    # plt.plot(wl, Xmsc.T)
    plt.ylabel('Absorbance spectra')
    plt.title('MSC')
    
    # plt.plot(wl, Xsnv.T)
    plt.xlabel('Wavelength (nm)')
    plt.title('SNV')
    
    plt.show()

