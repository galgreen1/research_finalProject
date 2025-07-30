import os, sys
import glob
import h5py
import numpy as np
from tqdm import tqdm
import argparse

main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
print(main_folder)

def preprocess_dataset(root_dir: str, save_dir: str) -> None:
    save_dir = os.path.join(save_dir, os.path.basename(root_dir))
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for folder in tqdm(glob.glob(os.path.join(root_dir, "*.h5"))):
        with h5py.File(folder, "r") as f:
            mixture = np.array(f.get("mixture"))
            soi = np.array(f.get("target"))
        for i in range(mixture.shape[0]):
            data = {
                "sample_mix": mixture[i, ...],
                "sample_soi": soi[i, ...],
            }
            np.save(os.path.join(save_dir, f"sample_{count}.npy"), data)
            count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-m', '--M_symbols', default=1, type=int)
    parser.add_argument('--soi_sig_type',default='OFDMQPSK', help='')
    args = parser.parse_args()

    soi_type = args.soi_sig_type
    M=args.M_symbols
    preprocess_dataset(root_dir=f'{main_folder}/dataset/Dataset_{soi_type}_train_Mixture_M={M}', 
                       save_dir=f'{main_folder}/npydataset/')
