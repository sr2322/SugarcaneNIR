Python scripts to analyze and display spectral data

Prediction models were all created using Partial Least-Squares Regression (PLS-R)

Note that there are two key preprocessing steps: scatter-correction and smoothing. For scatter correction, Standard Normal Variate correction (SNV) and Multiplicative Scatter Correction (MSC) result similar performance of the prediction model. However, MSC requires a "reference spectrum" (free of scattering effects ideally) that can only be approximated by taking the mean of all spectra in a dataset. Meanwhile SNV correction is done on each individual spectrum, and so does not require a reference spectrum. SNV is computationally less taxing, and more practical for prediction (i.e. preprocessing a single spectrum and then feeding it into the prediction model). So the final data analysis (see M.Eng. report) uses SNV correction. For smoothing, Savitzky-Golay ("Savgol") was used due its common use in NIR preprocessing.

