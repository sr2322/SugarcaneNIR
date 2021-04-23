# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter
from scipy import stats

def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
    # # Baseline correction
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
SampleBrixraw = ref_data.values[:,1]
SampleBrixes = []
for s in SampleBrixraw:
    test_s = str(s)
    if test_s[0].isnumeric():
        SampleBrixes.append(float(s))
    else:
        SampleBrixes.append(float(0.00))
# Brix_max = max(SampleBrixes)        

# Apply corrections - does it make sense to run MSC? Can't run MSC on incoming sample data...
# Xmsc = msc(Xmsc)[0] # MSC
Xmsc = savgol_filter(Xmsc, 35, polyorder = 3, deriv = 0) # Savgol smoothing


R2_2D_list = []
Xs_msc = pd.DataFrame(data=Xmsc, index=SampleBrixes, columns=wl_str)
wl_numerators = wl_str
wl_denominators = wl_str

count = 0
size = len(wl_numerators)*len(wl_denominators)

# Comment this out
# append to R2_2D_list to make a list of lists. Will convert to np array later
for n in wl_numerators:
    R2_row = []
    for d in wl_denominators:
        numerators = Xs_msc[n].values
        denominators = Xs_msc[d].values
        ratios=[]
        for i in range(len(numerators)):
            ratio = numerators[i]/denominators[i]
            ratios.append(ratio)
        slope, intercept, r_value, p_value, std_err = stats.linregress(SampleBrixes, ratios) # y,x
        R2_row.append( round(r_value,2) )
        #count+=1
        #print(count)
    R2_2D_list.append(R2_row)
    
array = np.array(R2_2D_list)

# # Uncomment this and comment out above to see working heatmap template
# wl_denominators = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]
# wl_numerators = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#             "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

# array = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(array)

ax.set_xticks(np.arange(len(wl_numerators)))
ax.set_yticks(np.arange(len(wl_denominators)))
# ... and label them with the respective list entries
ax.set_xticklabels(wl_numerators)
ax.set_yticklabels(wl_denominators)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(wl_denominators)):
    for j in range(len(wl_numerators)):
        text = ax.text(j, i, array[i, j],
                       ha="center", va="center", color="w")

ax.set_title("array")
fig.tight_layout()
plt.show()

