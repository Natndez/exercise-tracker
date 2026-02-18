import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

# ====================================================
# Load data
# ====================================================
df = pd.read_pickle('../../data/interim/01_data_processed.pkl')

# Define outlier columns
outlier_cols = list(df.columns[:6])

# ====================================================
# Plotting outliers (Boxplots)
# ====================================================
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# Acceleromater Outliers Plot
df[["acc_x", "label"]].boxplot(by="label", figsize=(20, 10))
df[["acc_y", "label"]].boxplot(by="label", figsize=(20, 10))
df[["acc_z", "label"]].boxplot(by="label", figsize=(20, 10))

# Gyroscope Outliers Plot
df[["gyr_x", "label"]].boxplot(by="label", figsize=(20, 10))
df[["gyr_y", "label"]].boxplot(by="label", figsize=(20, 10))
df[["gyr_z", "label"]].boxplot(by="label", figsize=(20, 10))

# Show multiple columns at once (acceleromater)
df[outlier_cols[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))

# Show multiple columns at once (gyroscope)
df[outlier_cols[3:6] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))

# Function for plotting outliers in time
# Taken from https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

# ====================================================
# Interquartile Range (Basing on distribution)
# ====================================================