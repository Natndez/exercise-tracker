import pandas as pd # type: ignore
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
        ["Not an outlier " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

# ====================================================
# Interquartile Range (Distribution based)
# ====================================================

# IQR Function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column
col = "acc_x"
dataset = mark_outliers_iqr(df, col)

plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)

# Plot all columns
for col in outlier_cols:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)
    
# ====================================================
# Chauvenets Criteron (Distribution Based)
# ====================================================

# Check for normal distribution in our data
# (acceleromater)
df[outlier_cols[:3] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3, 3))

# (gyroscope)
df[outlier_cols[3:6] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3, 3))

# Both generally appear normally distributed

# Chauvenetes Function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Plot all columns
for col in outlier_cols:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)
    
# Seeing less outliers with chauvenet!


# ====================================================
# Local Outlier Function (Distance Based)
# ====================================================
