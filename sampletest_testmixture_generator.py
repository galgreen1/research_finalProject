



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
#sigma_SNR = [math.sqrt(1/80)]


def calculate_ps_ratios_db():
    ps_ratios_db = np.linspace(0, 30, 30)  # Extend range to 30 dB
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
        generate_soi = lambda std,m,p_ratio: rfcutils.create_sig(std,m,p_ratio)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi


def generate_dataset( soi_type, n_per_batch, verbosity,foldername,m):
    ps_ratios, ps_ratios_db = calculate_ps_ratios_db()
    generate_soi = get_soi_generation_fn(soi_type)

    
    for i in ps_ratios:
            for j in sigma_SNR:
                p=compute_P(i)
                sig_list = []
                bits_list = []

                for _ in range(n_per_batch):
                    s1, s2 = generate_soi(j, m ,i)
                    s1 = s1.reshape(-1)  # shape M,N - M*N
                    s2=s2.reshape(-1) # shape M,N - M*N
                    sig_list.append(s1)
                    bits_list.append(s2)

                sig1_batch = np.stack(sig_list, axis=0)
                
                sig_orig_batch = np.stack(bits_list, axis=0)

                sig_comp = np.stack([np.real(sig1_batch), np.imag(sig1_batch)], axis=-1)
                org_comp = np.stack([np.real(sig_orig_batch), np.imag(sig_orig_batch)], axis=-1)
                sig_comp = tf.convert_to_tensor(sig_comp, dtype=tf.float32)
                org_comp = tf.convert_to_tensor(org_comp, dtype=tf.float32)
                mixture_filename = f'{soi_type}_mixture_p{p:.5f}_SNR{1/(j**2):.2f}.h5'
                if not os.path.exists(os.path.join(foldername)):
                    os.makedirs(os.path.join(foldername))
                filename_mixture=os.path.join( foldername, f"testmixture_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy")    
                filename_target =  os.path.join( foldername, f"testsymbols_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy") 
                # if i==ps_ratios[0]:
                #     print(f'mix:{sig_comp}  orig:{org_comp}')   

                np.save(filename_mixture, sig_comp)
                np.save(filename_target, org_comp)
                #print(f'mixture:{sig_comp.shape} {sig_comp}')
                #print(f'orig:{org_comp.shape}  {org_comp}')
                print(f"✅ saved test files at: {foldername}")

                


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    #parser.add_argument('-l', '--sig_len', default=80, type=int)
    parser.add_argument('-m', '--M_symbols', default=1, type=int)
    parser.add_argument('-b', '--n_per_batch', default=6144, type=int, help='')
    parser.add_argument('-d', '--dataset', default='test', help='')
    parser.add_argument('-t', '--test_set', default='TestSet', help='')
    parser.add_argument('-v', '--verbosity', default=1, help='')
    parser.add_argument('--soi_sig_type',default='OFDMQPSK', help='')
    args = parser.parse_args()

    soi_type = args.soi_sig_type
    dataset_type = args.dataset
    test_set=args.test_set
    Ms=args.M_symbols
    foldername = os.path.join('dataset', f'Dataset_{soi_type}_{dataset_type}_{test_set}_M={Ms}')

    generate_dataset( soi_type, args.n_per_batch, args.verbosity,foldername,args.M_symbols)





