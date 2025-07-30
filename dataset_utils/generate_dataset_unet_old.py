import os, sys
import glob
import h5py
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rfcutils
import tensorflow as tf
import math



sigma_SNR = [math.sqrt(1/30),math.sqrt(1/20),math.sqrt(1/10)]  # SNR we will make
#sigma_SNR = [math.sqrt(1/10)]

def calculate_ps_ratios_db():
    ps_ratios_db = np.linspace(0, 30, 30)  # Extend range to 30 dB
    #ps_ratios_db = np.linspace(0, 1, 1)  # Extend range to 30 dB
    ps_ratios = 10 ** (ps_ratios_db / 10)
    ps_values = ps_ratios / (1 + ps_ratios)
    return ps_ratios, ps_ratios_db


# compute P
def compute_P(ps_ratio):
    if ps_ratio==0:
        return 0
    a=ps_ratio
    P=a/(a+1)

    #print(P)
    return P


def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda std,p_ratio,n, s_len: rfcutils.generate_ofdm_signal_sim(std,p_ratio,n, s_len//80)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi


def generate_dataset( soi_type, sig_len,n_examples, n_per_batch, verbosity,foldername):
    ps_ratios, ps_ratios_db = calculate_ps_ratios_db()
    generate_soi = get_soi_generation_fn(soi_type)

    n_batches = int(np.ceil(n_examples/n_per_batch))
    for idx in tqdm(range(n_batches), disable=not bool(verbosity)):
        for i in ps_ratios:
            for j in sigma_SNR:
                p=compute_P(i)
                sig_mixture, _, _, sig_target = generate_soi(j,i,n_per_batch, sig_len)
                # if i==0.5:
                #     print(f'mix:{sig_mixture}  target:{sig_target}')

                sig_mixture_comp = tf.stack((tf.math.real(sig_mixture), tf.math.imag(sig_mixture)), axis=-1)
                #print(f'shape:{sig_mixture_comp.shape}')

                #sig_mixture_comp = tf.squeeze(sig_mixture_comp, axis=1)
                #print(f'estimate:{sig_mixture_comp}')
                sig_target_comp = tf.stack((tf.math.real(sig_target), tf.math.imag(sig_target)), axis=-1)
                #print(f'target:{sig_target_comp}')
                #print(f'shape:{sig_target_comp.shape}')

                mixture_filename = f'{soi_type}_mixture_{idx:04}_p{p:.3f}_SNR{1/(j**2):.2f}.h5'
                if not os.path.exists(os.path.join(foldername)):
                    os.makedirs(os.path.join(foldername))
                output_path = os.path.join( foldername, mixture_filename)    
                with h5py.File(os.path.join(foldername, mixture_filename), 'w') as h5file0:
                    h5file0.create_dataset('mixture', data=sig_mixture_comp)
                    #print(f'mix:{sig_mixture_comp}')
                    h5file0.create_dataset('target', data=sig_target_comp)
                    #print(f'target:{sig_target_comp}')
                    h5file0.create_dataset('sig_type', data=f'{soi_type}_mixture')
                    print(f"✅ saved h5 file at: {output_path}")

                del sig_mixture_comp, sig_target_comp

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-l', '--sig_len', default=80, type=int)
    parser.add_argument('-m', '--M_symbols', default=1, type=int)
    parser.add_argument('-n', '--n_examples', default=2500, type=int, help='')
    parser.add_argument('-b', '--n_per_batch', default=1250, type=int, help='')
    parser.add_argument('-d', '--dataset', default='train', help='')
    parser.add_argument('-v', '--verbosity', default=1, help='')
    parser.add_argument('--soi_sig_type',default='OFDMQPSK', help='')
    args = parser.parse_args()

    soi_type = args.soi_sig_type
    dataset_type = args.dataset
    foldername = os.path.join('dataset', f'Dataset_{soi_type}_{dataset_type}_Mixture_M={args.M_symbols}')

    generate_dataset( soi_type, args.sig_len, args.n_examples, args.n_per_batch, args.verbosity,foldername)

