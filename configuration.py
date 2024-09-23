import torch
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import pickle
import pandas as pd

with open("./data/gait_data.pkl", "rb") as file:
    df=pd.read_pickle(file)

# Dataset Configuration
batch_size = 6            # Number of samples per batch
window_size = 200         # Size of the window for processing data
stride = int(window_size / 2)  # Stride size for moving the window
use_scaler = True         # Flag to determine whether to use a scaler
seed = 42                 # Seed for random number generation to ensure reproducibility

# GroupShuffleSplit configuration
group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

# Cross-validation setup with GroupKFold
n_splits = 5
group_kfold = GroupKFold(n_splits=n_splits)

# Check if a GPU is available for PyTorch
gpu_available = torch.cuda.is_available()  # Check if CUDA is available
gpu_count = torch.cuda.device_count()      # Get the number of available GPUs
gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"  # Get the name of the first GPU, if available

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special: temporal
positive_tolerance = 25  # norm = 25
rough_estimate_min_peak_distance = 40
