# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
url = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugar1_nozeros.csv'
data = pd.read_csv(url, sep=',')
X = data.values[:,1:]
Xmsc = data.values[:,1:]

wl_str = data.columns[1:] 
wl = [float(numeric_string) for numeric_string in wl_str]

# Load reference data (to get total sugar values)
ref_url = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugar1_nozeros_Refs.csv'
ref_data = pd.read_csv(ref_url, sep=',')
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
Xmsc = savgol_filter(Xmsc, 45, polyorder = 3, deriv = 0)

linregcoeffs_msc = []
Xs_msc = pd.DataFrame(data=Xmsc, index=SampleTSes, columns=wl_str)

for wl_str_i in wl_str:
    Column_Xs_msc  = Xs_msc[wl_str_i].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(SampleTSes,Column_Xs_msc)
    linregcoeffs_msc.append(round(r_value,2))
print("StdCurves1 max is " + str(max(linregcoeffs_msc)) + "and min is " + str(min(linregcoeffs_msc)))

# Load data
url1 = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/F750.CSV'
data1 = pd.read_csv(url1, sep=';')
X1 = data1.values[:,1:]
Xmsc1 = data1.values[:,1:]

wl_str1 = data1.columns[1:] 
wl1 = [float(numeric_string) for numeric_string in wl_str1]

# Load reference data (to get total sugar values)
ref_url1 = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/References.csv'
ref_data1 = pd.read_csv(ref_url1, sep=';')
SampleTSraw1 = ref_data1.values[:,1]
SampleTSes1 = []
for s in SampleTSraw1:
    test_s = str(s)
    if test_s[0].isnumeric():
        SampleTSes1.append(float(s))
    else:
        SampleTSes1.append(float(0.00))

# Apply correction
Xmsc1 = msc(Xmsc1)[0] # Take the first element of the output tuple
# Xmsc = savgol_filter(Xmsc, 35, polyorder = 2, deriv = 0)

linregcoeffs_msc1 = []
Xs_msc1 = pd.DataFrame(data=Xmsc1, index=SampleTSes1, columns=wl_str1)

for wl_str_i in wl_str1:
    Column_Xs_msc1  = Xs_msc1[wl_str_i].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(SampleTSes1,Column_Xs_msc1)
    linregcoeffs_msc1.append(round(r_value,2))
print("F750 max is " + str(max(linregcoeffs_msc1)) + "and min is " + str(min(linregcoeffs_msc1)))

# Load data
url2 = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/Scio.CSV'
data2 = pd.read_csv(url2, sep=';')
X2 = data2.values[:,1:]
Xmsc2 = data2.values[:,1:]

wl_str2 = data2.columns[1:] 
wl2 = [float(numeric_string) for numeric_string in wl_str2]

# Load reference data (to get total sugar values)
ref_url2 = 'https://raw.githubusercontent.com/sr2322/SugarcaneNIR/main/References.csv'
ref_data2 = pd.read_csv(ref_url2, sep=';')
SampleTSraw2 = ref_data2.values[:,1]
SampleTSes2 = []
for s in SampleTSraw2:
    test_s = str(s)
    if test_s[0].isnumeric():
        SampleTSes2.append(float(s))
    else:
        SampleTSes2.append(float(0.00))

# Apply correction
Xmsc2 = msc(Xmsc2)[0] # Take the first element of the output tuple
# Xmsc = savgol_filter(Xmsc, 35, polyorder = 2, deriv = 0)

linregcoeffs_msc2 = []
Xs_msc2 = pd.DataFrame(data=Xmsc2, index=SampleTSes2, columns=wl_str2)

for wl_str_i in wl_str2:
    Column_Xs_msc2  = Xs_msc2[wl_str_i].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(SampleTSes2,Column_Xs_msc2)
    linregcoeffs_msc2.append(round(r_value,2))
print("Scio max is " + str(max(linregcoeffs_msc2)) + " and min is " + str(min(linregcoeffs_msc2)))

# Load data
url3 = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugarLit1.csv'
data3 = pd.read_csv(url3, sep=',')
X3 = data3.values[:,1:]
Xmsc3 = data3.values[:,1:]

wl_str3 = data3.columns[1:] 
wl3 = [float(numeric_string) for numeric_string in wl_str3]

# Load reference data (to get total sugar values)
ref_url3 = 'https://raw.githubusercontent.com/sr2322/mytests/main/TableSugarLit1_Refs.csv'
ref_data3 = pd.read_csv(ref_url3, sep=',')
SampleTSraw3 = ref_data3.values[:,1]
SampleTSes3 = []
for s in SampleTSraw3:
    test_s = str(s)
    if test_s[0].isnumeric():
        SampleTSes3.append(float(s))
    else:
        SampleTSes3.append(float(0.00))

# Apply correction
Xmsc3 = msc(Xmsc3)[0] # Take the first element of the output tuple
Xmsc3 = savgol_filter(Xmsc3, 45, polyorder = 3, deriv = 0)

linregcoeffs_msc3 = []
Xs_msc3 = pd.DataFrame(data=Xmsc3, index=SampleTSes3, columns=wl_str3)

for wl_str_i in wl_str3:
    Column_Xs_msc3  = Xs_msc3[wl_str_i].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(SampleTSes3,Column_Xs_msc3)
    linregcoeffs_msc3.append(round(r_value,2))
print("Bulby max is " + str(max(linregcoeffs_msc3)) + "and min is " + str(min(linregcoeffs_msc3)))


## Plot original and corrected spectra
fig, ax = plt.subplots(figsize=(12,8))
with plt.style.context(('ggplot')):
    plt.plot(wl,linregcoeffs_msc)
    plt.plot(wl1,linregcoeffs_msc1)
    plt.plot(wl2,linregcoeffs_msc2)
    plt.plot(wl3,linregcoeffs_msc3)
    plt.axvline(x=730, color = "grey")
    plt.axvline(x=830, color = "grey")
    plt.axvline(x=915, color = "grey")
    plt.axvline(x=959, color = "grey")
    plt.axvline(x=960, color = "grey")
    plt.axvline(x=961, color = "grey")
    
    # plt.plot(wl, Xsnv.T)
    plt.xlabel('Wavelength (nm)')
    
    plt.show()


