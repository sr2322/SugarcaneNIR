# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

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

# Apply correction - Xmsc and Xsnv are ndarrays
Xmsc = msc(X)[0] # Take the first element of the output tuple
Xsnv = snv(X)

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

Spearmans_msc = []
Xs_msc = pd.DataFrame(data=Xmsc, index=SampleTSes, columns=wl_str)

for wl_str_i in wl_str:
    Column_Xs_msc  = Xs_msc[wl_str_i].values
    spearman = scipy.stats.spearmanr(SampleTSes, Column_Xs_msc)[0]
    Spearmans_msc.append(round(spearman,2))
print (Spearmans_msc)

Spearmans_snv = []
Xs_snv = pd.DataFrame(data=Xsnv, index=SampleTSes, columns=wl_str)

for wl_str_i in wl_str:
    Column_Xs_snv  = Xs_snv[wl_str_i].values
    spearman = scipy.stats.spearmanr(SampleTSes, Column_Xs_snv)[0]
    Spearmans_snv.append(round(spearman,2))
print (Spearmans_snv)


## Plot original and corrected spectra
plt.figure(figsize=(24,27))
with plt.style.context(('ggplot')):
    ax1 = plt.subplot(311)
    plt.plot(wl,Spearmans_msc)
    plt.ylabel('Absorbance spectra')
    plt.title('MSC')
    
    ax2 = plt.subplot(312)
    plt.plot(wl,Spearmans_snv)
    # plt.plot(wl, Xsnv.T)
    plt.xlabel('Wavelength (nm)')
    plt.title('SNV')
    
    plt.show()
