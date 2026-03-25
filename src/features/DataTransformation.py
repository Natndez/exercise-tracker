from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd

# Class removes the high frequency data (considered noise) from clean data
# Can only be applied when there arent missing values
class LowPassFilter:
    def low_pass_filter(
        self,
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        # Cutoff frequencies expressed as fraction of Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq
        
        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        return data_table
    
# Class for Prinicipal Component Analysis. We can only apply this when we don't have missing values
# We have to impote these first, beware......
class PrincipalComponentAnalysis:
    pca = []
    
    def __init__(self):
        self.pca = []
    
    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm
    
    # Perform the PCA on the selected columns and return the explained variance
    def determine_pc_explained_variance(self, data_table, cols):
        
        # Normalize
        dt_norm = self.normalize_dataset(data_table, cols)
        
        # Perform pca
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # Return explained variances
        return self.pca.explained_variance_ratio_
    
    # Apply a PCA given the number of selected components
    # Add new columns
    def apply_pca(self, data_table, cols, number_comp):
        
        # Normalize data
        dt_norm = self.normalize_dataset(data_table, cols)
        
        # Perform pca
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])
        
        # Transform old values
        new_values = self.pca.transform(dt_norm[cols])
        
        # Add new values
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]
        
        return data_table