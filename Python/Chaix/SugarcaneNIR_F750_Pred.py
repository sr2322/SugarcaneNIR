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
    # for i in range(input_data.shape[0]):
    #     input_data[i,:] -= input_data[i,:].mean()

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

# Apply correction
Xmsc = snv(Xmsc)# Take the first element of the output tuple
Xmsc = savgol_filter(Xmsc,15, polyorder = 3, deriv = 0)

linregcoeffs_msc = []
Xs_msc = pd.DataFrame(data=Xmsc, index=SampleTSes, columns=wl_str)

# Xs_992  = Xs_msc["992.4"].values
# Xs_913  = Xs_msc["912.8"].values
# Xs_909  = Xs_msc["909.3"].values
Xs_740  = Xs_msc["740"].values
Xs_830  = Xs_msc["830"].values
Xs_910  = Xs_msc["910"].values
Xs_915  = Xs_msc["915"].values
Xs_960  = Xs_msc["960"].values
Xs_780  = Xs_msc["790"].values
Xs_840  = Xs_msc["840"].values
Xs_970  = Xs_msc["965"].values
Xs_1026  = Xs_msc["1025"].values

ratios = []
for i in range(len(Xs_970)):
    ratio = Xs_msc["786"].values[i] / Xs_msc["967"].values[i]
    #ratio = 122 + 1375*Xs_740[i] - 942*Xs_830[i] + 855*Xs_915[i] - 736*Xs_960[i]
    ratios.append(ratio)
np.array(ratios)
print (ratios)

    
# plt.plot(Xs_a,SampleTSes,'o')
# slope, intercept, r_value, p_value, std_err = stats.linregress(Xs_a,SampleTSes)
# plt.plot(Xs_a, slope*Xs_a + intercept)
# print (r_value)

# plt.plot(Xs_b,SampleTSes,'o')
# slope, intercept, r_value, p_value, std_err = stats.linregress(Xs_b,SampleTSes)
# plt.plot(Xs_b, slope*Xs_b + intercept)
# print (r_value)

plt.plot(ratios,SampleTSes,'o')
slope, intercept, r_value, p_value, std_err = stats.linregress(ratios,SampleTSes)
predicts = []
for r in ratios:
    predicts.append(r*slope+intercept)
    np.array(predicts)
    
plt.plot(ratios, predicts)
print(r_value)
print(str(slope)+"*r " + str(intercept)) #this is the equation to predict on SUGAR SOLUTION
# Goal is to see if we can just apply a constant correction factor to predict on SUGARCANE 

