
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import argparse
import numpy as np
from tqdm import tqdm
import torch

from src.torchwavenet import Wave
from omegaconf import OmegaConf
from src.config_torchwavenet import Config, parse_configs

# List of SNR values to test (converted to linear scale)
sigma_SNR = [math.sqrt(1/30), math.sqrt(1/20), math.sqrt(1/10)]


def calculate_ps_ratios_db():
   
    ps_db = np.linspace(0, 30, 24)
    ps = 10 ** (ps_db / 10)
    return ps, ps_db


def compute_P(ps_ratio):
    """
    Given ps_ratio = P_s/(1-P_s), return P_s.
    If ps_ratio==0, returns 0.
    """
    return 0.0 if ps_ratio == 0 else ps_ratio / (1 + ps_ratio)


def run_inference(foldername: str, soi_type: str,M):
    # Load config and model checkpoint
    base_cfg = OmegaConf.load("src/configs/wavenet.yml")
    cfg: Config = Config(**parse_configs(base_cfg, None))
    cfg.model_dir = f"torchmodels/dataset_{soi_type.lower()}_mixture_wavenet_M={M}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wave(cfg.model).to(device)
    checkpoint = torch.load(os.path.join(cfg.model_dir, "weights.pt"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Prepare the list of power ratios
    ps_ratios, _ = calculate_ps_ratios_db()

    # For each (P, SNR) combination, load the mixture, run the model, and save the estimate
    for p_ratio in ps_ratios:
        p = compute_P(p_ratio)
        for snr in sigma_SNR:
            # Load the complex-valued mixture (shape: [N, signal_length])
            fname = f"testmixture_{soi_type}_p{p:.5f}_SNR{1/(snr**2):.2f}.npy"
            path = os.path.join(foldername, fname)
            mixture = np.load(path)  # dtype: complex64 or complex128
            

            # Split real and imaginary parts and reshape for the model:
            # from [N, L] complex to [N, 2, L] float
            #real = np.real(mixture)
            #imag = np.imag(mixture)
            #inp = np.stack([real, imag], axis=1)  # [N, 2, L]
            #inp_tensor = torch.from_numpy(inp).float().to(device)
            inp = mixture.transpose(0, 2, 1)      # now channels=2, length=L
            inp_tensor = torch.from_numpy(inp).float().to(device)
            

            # Forward pass
            with torch.no_grad():
                output = model(inp_tensor)  # [N, 2, L]

            # Convert back to NumPy complex array [N, L]
            out_np = output.cpu().numpy()
            sig_est = out_np[:, 0, :] + 1j * out_np[:, 1, :]

            # Ensure output directory exists
            os.makedirs(foldername, exist_ok=True)

            #print(f'mix:{inp} out:{sig_est}')

            # Save the estimated SOI
            out_fname = f"Wavenet_estimated_soi_{soi_type}_p{p:.5f}_SNR{1/(snr**2):.2f}.npy"
            out_path = os.path.join(foldername, out_fname)
            np.save(out_path, sig_est)
            print(f"Saved {out_path}, shape {sig_est.shape} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WaveNet inference on RF mixtures")
    parser.add_argument("-m", "--M_symbols", type=int, default=1,
                        help="Number of mixture symbols (for folder naming)")
    parser.add_argument("-d", "--dataset", type=str, default="test",
                        help="Dataset identifier")
    parser.add_argument("--soi_sig_type", type=str, default="OFDMQPSK",
                        help="Signal-of-interest type")
    args = parser.parse_args()

    folder = os.path.join(
        "dataset",
        f"Dataset_{args.soi_sig_type}_{args.dataset}_Mixture_M={args.M_symbols}"
    )
    run_inference(folder, args.soi_sig_type,args.M_symbols)
