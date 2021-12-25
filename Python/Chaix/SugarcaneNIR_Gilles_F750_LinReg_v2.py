# -*- coding: utf-8 -*-
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

# Load data
url = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/Scio.CSV'
data = pd.read_csv(url, sep=';')
X = data.values[:,1:]
Xmsc = data.values[:,1:]

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
Xmsc = msc(Xmsc)[0] # Take the first element of the output tuple
Xmsc = savgol_filter(Xmsc, 15, polyorder = 3, deriv = 0)

linregcoeffs_msc = []
Xs_msc = pd.DataFrame(data=Xmsc, index=SampleTSes, columns=wl_str)

for wl_str_i in wl_str:
    Column_Xs_msc  = Xs_msc[wl_str_i].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(SampleTSes,Column_Xs_msc)
    linregcoeffs_msc.append(round(r_value,2))
#linregcoeffs_msc = np.array(linregcoeffs_msc)
print(max(linregcoeffs_msc))
print(min(linregcoeffs_msc))

## Plot original and corrected spectra
plt.figure(figsize=(24,27))
with plt.style.context(('ggplot')):
    ax1 = plt.subplot(311)
    for X_i, TS_i in zip(X,SampleTSes):
        plt.plot(wl, X_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    # plt.plot(wl, X.T)
    plt.title('Original data')
    
    ax2 = plt.subplot(312)
    for Xmsc_i, TS_i in zip(Xmsc,SampleTSes):
        plt.plot(wl, Xmsc_i, lw=1, c=cm.hot( 1-(TS_i/TS_max)) )
    # plt.plot(wl, Xmsc.T)
    plt.ylabel('Absorbance spectra')
    plt.title('MSC')
    
    ax2 = plt.subplot(313)
    plt.plot(wl,linregcoeffs_msc)
    
    plt.axvline(x=730, color = "grey")
    plt.axvline(x=830, color = "grey")
    plt.axvline(x=915, color = "grey")
    plt.axvline(x=960, color = "grey")
    
    # plt.plot(wl, Xsnv.T)
    plt.xlabel('Wavelength (nm)')
    
    plt.show()


