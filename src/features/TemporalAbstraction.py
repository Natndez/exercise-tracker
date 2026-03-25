# Imports
import numpy as np
import scipy.stats as stats

# Class to abstract a history of numerical values we can use as an attribute
class NumericalAbstracion:
    
    # This function aggregates a list of values using the specified aggregation function
    # (e.g. mean, max, min, median, std)
    def aggregate_value(self, aggregation_function):
        # Compute and return
        if aggregation_function == 'mean':
            return np.mean
        elif aggregation_function == 'median':
            return np.median
        elif aggregation_function == 'max':
            return np.max
        elif aggregation_function == 'min':
            return np.min
        elif aggregation_function == 'std':
            return np.std
        else:
            return np.nan
    
    
    # Aggregate numerical columns specified given a window size and an aggregation function
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):
        
        # Create new columns for temporal data, pass over the dataset, and compute values
        for col in cols:
            data_table[
                col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )
            
        return data_table
        