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

# Can change RC params in order to customize plots without too much repetition
# Uses mpl not plt
mpl.style.use("seaborn-deep")
mpl.rcParams["figure.figsize"] = (20, 5)

# ====================================================
# Compare Heavy vs Medium Sets (Squats)
# ====================================================
# Stacking queries to get our df
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index() # string match, need the quotes in this structure

# Groupby to get a plot based on category
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("Vertical Acceleration") 
ax.set_xlabel("Samples") 
plt.legend()

# ====================================================
# Compare Participants (Bench)
# ====================================================
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("Vertical Acceleration") 
ax.set_xlabel("Samples") 
plt.legend()

# ====================================================
# Plot Multiple Axis
# ====================================================