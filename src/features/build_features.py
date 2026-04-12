import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstracion

# ====================================================
# Load Data
# ====================================================

df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenet.pkl')

predictor_columns = list(df.columns[:6])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ====================================================
# Dealing with missing values (Imputation)
# ====================================================

# Seeing that we have a solid amount of null values
df.info()

# Going to interpolate values for rows containing NaaN
for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info() # no longer seeing missing values

# ====================================================
# Calculating set duration
# ====================================================
df[df["set"] == 40]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

duration.seconds # Rounded to nearest second

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# ====================================================
# Butterworth lowpass filter
# ====================================================