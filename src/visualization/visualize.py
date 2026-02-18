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
label = "squat"
participant = "A"

all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("Acceleration") 
ax.set_xlabel("Samples")
plt.legend()

# ====================================================
# Loop to plot all combinations per sensor
# ====================================================
labels = df["label"].unique()
participants = df["participant"].unique()

# Acceleromater data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
            )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("Acceleration") 
            ax.set_xlabel("Samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
            
# Gyroscope data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
            )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("Orientation") 
            ax.set_xlabel("Samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
            
# ====================================================
# Combining Plots into one figure
# ====================================================
# Using Row for this visualization
label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

# Plotting data for each sensor
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

# Plot styling
ax[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[0].set_ylabel("Acceleration")
ax[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("Samples")
ax[1].set_ylabel("Orientation")

# ====================================================
# Loop through all combinations and export
# ====================================================
labels = df["label"].unique()
participants = df["participant"].unique()

# Combined Dataframe
for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
            )
        if len(combined_plot_df) > 0:
            # Plotting data for each sensor
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            # Plot styling
            ax[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[0].set_ylabel("Acceleration")
            ax[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("Samples")
            ax[1].set_ylabel("Orientation")
            
            # Exporting figures to store them (always report figures)
            # Progreammatically name files (.title() just for caps)
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            
            plt.show()
            
            # Close figure to avoid memory issues in long loops like this
            plt.close(fig)
            
            