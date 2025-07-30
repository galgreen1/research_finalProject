import os
import glob
import h5py
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

class RFMixtureDatasetBase(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError("Dataset root directory does not exsist.")
        self.files = glob.glob(os.path.join(self.root_dir, "*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = np.load(self.files[i], allow_pickle=True).item()
        mix_list = data["sample_mix"].tolist()
        s_list   = data["sample_soi"]  .tolist()
        sample_mix = torch.tensor(mix_list, dtype=torch.float32).transpose(0, 1)
        sample_soi   = torch.tensor(s_list,   dtype=torch.float32).transpose(0, 1)
        return {
            "sample_mix": sample_mix,
            "sample_soi"  : sample_soi,
            }

        # return {
        #     "sample_mix": torch.tensor(data["sample_mix"]).transpose(0, 1),
        #     "sample_soi": torch.tensor(data["sample_soi"]).transpose(0, 1),
        # }
    

def get_train_val_dataset(dataset: Dataset, train_fraction: float):
    # print(len(dataset))
    val_examples = int((1 - train_fraction) * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_examples, val_examples], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


if __name__ == "__main__":
    dataset = RFMixtureDatasetBase(
        root_dir="/home/dsi/galgreen/tmp/rfchallenge/npydataset/Dataset_OFDMQPSK_train_Mixture_M=50",
    )
