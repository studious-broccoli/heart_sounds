import torch
from torch.utils.data import Dataset
import os
from utils.dtw_tools import extract_mfcc_librosa


class HeartSoundDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mfcc = extract_mfcc_librosa(self.file_paths[idx])
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, 13, 300]
        label = 0 if self.labels[idx] == -1 else 1
        label = torch.tensor(label, dtype=torch.long)
        return {
            "mfccs": mfcc,  # shape: [1, 13, 300]
            "labels": label
        }
