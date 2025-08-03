import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
import os, sys
import numpy as np
import random
import h5py
from tqdm import tqdm
import pickle
import math
import rfcutils
import argparse


from src import unet_model as unet

sigma_SNR = [math.sqrt(1/30),math.sqrt(1/20),math.sqrt(1/10)]  # SNR we will make
#sigma_SNR = [math.sqrt(1/30)]

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
    #P = round(P, 5)

    #print(P)
    return P


def run_inference(soi_type,folder_name,M,sig_len):
    

    gen = f"create_simulations_dataset_m_{M}"



    nn_model = unet.get_unet_model((sig_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0009)
    nn_model.load_weights(os.path.join('models', f'{gen.lower()}_unet_M_{M}', 'checkpoint'))
    #nn_model.load_weights('/home/dsi/galgreen/tmp/rfchallenge/models-save/create_simulations_dataset_unet_M_1/checkpoint')
    ps_ratios, _ = calculate_ps_ratios_db()
    for i in ps_ratios:         
        for j in sigma_SNR:
            #j = round(j, 5)
            p=compute_P(i)
            path=os.path.join( foldername, f"testmixture_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy")    
            mixture = np.load(path)
            
            #sig_comp = tf.stack((tf.math.real(mixture), tf.math.imag(mixture)), axis=-1)
            sig_comp = tf.convert_to_tensor(mixture, dtype=tf.float32) 
            #print(f'mix:{sig_comp.shape}')

            sig1_out = nn_model.predict(sig_comp, batch_size=64, verbose=1)
            sig1_est = tf.complex(sig1_out[:,:,0], sig1_out[:,:,1])
            sig1_est=sig1_est.numpy()
            #print(f'orig:{sig_comp}')
            #print(f'est:{sig1_out}')
            # if i==ps_ratios[5]:
            #     print(sig1_est[1][1],mixture[1][1])
            if not os.path.exists(os.path.join(foldername)):
                os.makedirs(os.path.join(foldername))
            #print(f'estimate soi:{sig1_est}')    
            #print(f'shape:{sig1_est.shape}')
            path=os.path.join( foldername, f"estimated_soi_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy")
            np.save(path, sig1_est)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-l', '--sig_len', default=64, type=int)
    parser.add_argument('-m', '--M_symbols', default=1, type=int)
    parser.add_argument('-d', '--dataset', default='test', help='')
    parser.add_argument('-t', '--test_set', default='TestSet', help='')
    parser.add_argument('--soi_sig_type',default='OFDMQPSK', help='')
    args = parser.parse_args()

    soi_type = args.soi_sig_type
    dataset_type = args.dataset
    test_set=args.test_set
    length=args.sig_len
    Ms=args.M_symbols
    length=length*Ms
    foldername = os.path.join('dataset', f'Dataset_{soi_type}_{dataset_type}_{test_set}_M={Ms}')

    run_inference(soi_type,foldername,Ms,length)
