from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
import numpy as np
import torch

def remove_nan_positions(arr1, arr2):
    # Find positions of NaN in both arrays
    nan_positions_arr1 = np.isnan(arr1)
    nan_positions_arr2 = np.isnan(arr2)
    
    # Combine positions to find indices to remove
    nan_positions_combined = nan_positions_arr1 | nan_positions_arr2
    
    # Filter out the NaN positions from both arrays
    arr1_cleaned = arr1[~nan_positions_combined]
    arr2_cleaned = arr2[~nan_positions_combined]
    
    return arr1_cleaned, arr2_cleaned

def pad_and_stack(tensors):
    """
    Pad each tensor to the same length and stack them into a single tensor.
    
    Parameters:
    tensors (list of torch.Tensor): List of 2D tensors to be padded and stacked.

    Returns:
    torch.Tensor: Padded and stacked tensor.
    """
    # Find the maximum length along the second dimension
    max_len = max(tensor.size(1) for tensor in tensors)
    
    # Pad each tensor to have the same length
    padded_tensors = []
    for tensor in tensors:
        pad_size = max_len - tensor.size(1)
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        padded_tensors.append(padded_tensor)
    
    # Stack the padded tensors into a single tensor
    stacked_tensor = torch.stack(padded_tensors)
    
    return stacked_tensor

class CustomDataset(Dataset):
    """
    Custom dataset class for handling sequences and labels.
    """
    def __init__(self, sequences, labels):
        """
        Initialize the dataset with sequences and labels.
        
        Parameters:
        sequences (list of numpy arrays): List of sequences.
        labels (list of numpy arrays): List of corresponding labels.
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get the sequence and label at a specific index.
        
        Parameters:
        idx (int): Index of the item.
        
        Returns:
        tuple: (sequence, label) at the given index.
        """
        return self.sequences[idx], self.labels[idx]

def sliding_window(data, labels, window_size, stride):
    """
    Generate sequences and labels using a sliding window approach.
    
    Parameters:
    data (numpy array): Input data array.
    labels (numpy array): Input labels array.
    window_size (int): Size of the sliding window.
    stride (int): Stride of the sliding window.
    
    Returns:
    tuple: (sequences, labels_seq) where sequences are the extracted sequences
           and labels_seq are the corresponding labels.
    """
    sequences = []
    labels_seq = []
    
    # Slide the window over the data to generate sequences and labels
    for i in range(0, len(data) - window_size + 1, stride):
        sequences.append(data[i: i + window_size])
        labels_seq.append(labels[i: i + window_size])
    
    return sequences, labels_seq
