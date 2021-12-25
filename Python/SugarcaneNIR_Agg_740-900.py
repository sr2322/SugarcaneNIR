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

# Load data for F750 -- 741-900nm
url_f750 = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/F750.CSV'
data_f750 = pd.read_csv(url_f750, sep=';')
X_f750 = data_f750.values[:,98:152]
wl_f750 = data_f750.columns[98:152]
wl_f750 = [float(numeric_string) for numeric_string in wl_f750]
print (wl_f750)
print ('\n')

# Load data for Scio -- 740-900nm
url_scio = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/Scio.CSV'
data_scio = pd.read_csv(url_scio, sep=';')
X_scio = data_scio.values[:,1:162]
wl_scio = data_scio.columns[1:162]
wl_scio = [float(numeric_string) for numeric_string in wl_scio]
print (wl_scio)
print ('\n')

wl = wl_f750 + wl_scio
wl.sort()
print(wl)

# Sort before applying correction!
# Apply correction
# Xmsc = msc(X)[0] # Take the first element of the output tuple
# Xsnv = snv(X)
