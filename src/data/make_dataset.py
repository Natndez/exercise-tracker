import pandas as pd # type: ignore
from glob import glob

# -----------------------------
# READ SINGLE CSV FILE
# -----------------------------
singe_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# -----------------------------
# LIST ALL DATA IN DATA/RAW/MetaMotion
# -----------------------------
files = glob("../../data/raw/MetaMotion/*.csv") # to get all of the csv files

# Should be 187 files
len(files)

# -----------------------------
# EXTRACT FEATURE FROM FILENAME
# -----------------------------
data_path = "../../data/raw/MetaMotion/"
f = files[0]

# Nameing structure ([participant]-[exercise(label)]-[category(heavy/light)]-[rpe])

f.split("-")[0].replace(data_path, "") # spliting and erasing path

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category

# -----------------------------
# READ ALL FILES
# -----------------------------

# Creating dataframes
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

# Will increment to make a set counter
acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(f)
    
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df]) # append new data to acc_df
        
    elif "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])
    
# -----------------------------
# WORKING WITH DATETIMES
# -----------------------------
acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# These columns are no longer needed
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]
# -----------------------------
# TURN INTO FUNCTION
# -----------------------------
files = glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    
    acc_set = 1
    gyr_set = 1
    
    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df]) 
            
        elif "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
    
    # Set datetime index
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
    
    # Remove unneeded columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files) 

# -----------------------------
# MERGING OUR DATASETS (easier to just have one large dataset)
# -----------------------------
# concatinating while avoiding duplicate values
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1) #axis=1 means we are merging column wise, not row wise

# Changing column labels
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# -----------------------------
# Resample Data (Converting Frequencies)
# -----------------------------

# Accelerometer: 12.500Hz => measures every 0.08 seconds
# Gyroscope: 25.000Hz => measures every 0.04 seconds

sampling = {
    'acc_x': "mean", 
    'acc_y': "mean", 
    'acc_z': "mean", 
    'gyr_x': "mean", 
    'gyr_y': "mean", 
    'gyr_z': "mean", 
    'label': "last",
    'category': "last", 
    'participant': "last", 
    'set': "last"
}

data_merged.columns

# Since the data is recorded over one week, we will split it by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled.info()

# We want set as an int not float64
data_resampled["set"] = data_resampled["set"].astype("int")

data_resampled.info()
# -----------------------------
# Export Dataset
# -----------------------------
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")