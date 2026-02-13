import pandas as pd #type: ignore
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# ====================================================
# Load data
# ====================================================

df = pd.read_pickle('../../data/interim/01_data_processed.pkl')

# ====================================================
# Plot single column
# ====================================================
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])

plt.plot(set_df["acc_y"].reset_index(drop=True))

# ====================================================
# Plot All Exercises
# ====================================================

# Iterate through different exercises
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.title(label)
    #plt.legend()
    plt.show()
    
# Iterate through different exercises (first 100 samples)
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.title(label)
    #plt.legend()
    plt.show()
    
# ====================================================
# Adjust plot settings
# ====================================================
