import os, sys
import numpy as np
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse



# 4-QAM symbols 
qam_symbol = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
qam_symbol=qam_symbol/(math.sqrt(2))


# Normalize QAM symbols (we scale them to unit energy)
sigma_SNR = [math.sqrt(1/30),math.sqrt(1/20),math.sqrt(1/10)]  # SNR we will make
#sigma_SNR = [math.sqrt(1/80)]

# for a whole signal compute the closet 4QAM
def computeClosest(es_com):
    # Handle 1D input
    if es_com.ndim == 1:
        es_com = es_com[np.newaxis, :]
    qam_symbols =qam_symbol.reshape((1, 1, -1))
    closest_symbols = np.zeros_like(es_com, dtype=complex)
    #print(es_com)
    # Compute Euclidean distances for all elements in es_com to all 4-QAM symbols
    distances = np.abs(es_com[:, :, np.newaxis] - qam_symbols)  # Shape: (MxN, 4)

    # Find the index of the minimum distance for each element
    closest_symbol_indices = np.argmin(distances, axis=2)  # Shape: (MxN)

    # Map indices back to symbols
    closest_symbols = qam_symbols[0, 0, closest_symbol_indices]
    return closest_symbols.squeeze() 


def calculate_ps_ratios_db():
    ps_ratios_db = np.linspace(0, 30, 30)  # Extend range to 30 dB
    ps_ratios = 10 ** (ps_ratios_db / 10)
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

def convert_qam_to_bits(x):
    a = 1 / math.sqrt(2)
    qam_map = {
        complex(a, a): (0, 0),
        complex(-a, a): (0, 1),
        complex(-a, -a): (1, 1),
        complex(a, -a): (1, 0),
    }

    for key, val in qam_map.items():
        if np.isclose(x.real, key.real, atol=1e-5) and np.isclose(x.imag, key.imag, atol=1e-5):
            return val

    raise ValueError(f"⚠️ Symbol {x} not recognized as valid QAM")


# def convert_qam_to_bits(x):
#     print(f'x:{x}')
#     a=1/(math.sqrt(2))
#     print(f'x:{x}')
#     if x==complex(a,a):
#         return 0,0
#     if x==complex(-a,a):
#         return 0,1
#     if x==complex(-a,-a):
#         return 1,1
#     if x==complex(a,-a):
#         return 1,0



def run_demod_test(soi_type,testset_identifier,M,net):
    # Evaluation pipeline
    def compute_ber(sig_est, sig_soi):  # shape M,N
        assert sig_est.shape ==sig_soi.shape, 'Invalid SOI estimate shape'
        # map est to 4QAM 
        sig_est=computeClosest(sig_est)
        sig_est = np.atleast_2d(sig_est)
        sig_soi = np.atleast_2d(sig_soi)
     
        

        

        sum_error=0
        #print(f'sig est:{sig_est}')
        #print(f'sig:{sig_soi}')
        for i in range(sig_est.shape[0]):
            for j in range(sig_est.shape[1]):
                x11,x12=convert_qam_to_bits(sig_est[i][j])
                x21,x22=convert_qam_to_bits(sig_soi[i][j])
                #print(f'est:{sig_est[i][j]} and orig:{sig_soi[i][j]}')
                if x11!=x21:
                    sum_error+=1
                if x12!=x22:
                    sum_error+=1
        sum_error=sum_error/(sig_est.shape[0]*sig_est.shape[1]*2)     

        return sum_error           
        

        
    

    ps_ratios, ps_ratios_db = calculate_ps_ratios_db()
    save_error = np.zeros((len(sigma_SNR), len(ps_ratios)), dtype=np.float32)
    save_error_check = np.zeros((len(sigma_SNR), len(ps_ratios)), dtype=np.float32)
    for a in range(len(sigma_SNR)):

        for i in range(len(ps_ratios)):  
            j=sigma_SNR[a]
            
            p=compute_P(ps_ratios[i])
            
            sig=np.load(os.path.join( foldername, f"testsymbols_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy"))
            
            check= np.load(os.path.join( foldername, f"testmixture_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy"))
            
            real_check=check[:,:,0]
            imag=check[:,:,1]
            check=real_check+1j*imag
            assert ~np.isnan(check).any(), 'NaN or Inf in Signal load'
             

            real_part = sig[:, :,0]
            imag_part = sig[:,:, 1]
            sig=real_part+1j*imag_part
            assert ~np.isnan(sig).any(), 'NaN or Inf in Signal load'

            #print(f'sig: {sig.shape}')
            path=os.path.join( foldername, f"estimated_soi_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy")
            if net=='Wavenet':
                path=os.path.join( foldername, f"Wavenet_estimated_soi_{soi_type}_p{p:.5f}_SNR{1/(j**2):.2f}.npy")
                print("WAVENET network")
            sig_est=np.load(path)
            
            print(f'sig_est: {sig_est.shape}')
            assert ~np.isnan(sig_est).any(), 'NaN or Inf in Signal estimate load'
            
            #print(sig_est.shape)
            count_check=0
            count=0
            for num in range(sig.shape[0]):  # num of signals
             # Handle 1D input explicitly here:
                if sig.ndim == 1:
                    sig = sig[np.newaxis, :]
                    sig_est = sig_est[np.newaxis, :]
                    check = check[np.newaxis, :]
                count+=compute_ber(sig_est[num],sig[num])
                count_check+=compute_ber(check[num],sig[num])
                
            count=count/sig.shape[0]   # error for certain p and SNR  
            count_check=count_check/sig.shape[0]
            print(f'error for SNR={j} and p={p} is {count} and count check:{count_check}')
            #print(f'counter:{count}')
            save_error[a][i]=count 
            save_error_check[a][i]=count_check
    # now to print graph: 
    x1=save_error[0]
    a1=save_error_check[0]
    a2=save_error_check[1]
    a3=save_error_check[2]
    #print(f'x:{x1}')
    x2=save_error[1]
    x3=save_error[2]
    plt.figure(figsize=(10, 6))
    #plt.plot(ps_ratios_db, x1, label=f"Com: SNR=80dB M=1  ",color='blue')

    plt.plot(ps_ratios_db, x1, label=f"Com: SNR=30dB M={M} network={net} ",color='blue')
    plt.plot(ps_ratios_db, x2, label=f"Com: SNR=20dB M={M} network={net}", color='green')
    plt.plot(ps_ratios_db, x3,  label=f"Com: SNR=10dB M={M}  network={net}", color='red')

    plt.plot(ps_ratios_db, a1, label=f"Com: SNR=30dB M={M}  simulation",color='blue',linestyle='--')
    plt.plot(ps_ratios_db, a2, label=f"Com: SNR=20dB M={M} simulation", color='green',linestyle='--')
    plt.plot(ps_ratios_db, a3,  label=f"Com: SNR=10dB M={M}  simulation", color='red',linestyle='--')
    #print(f'x1:{x1}')

    plt.yscale("log")
    plt.xlim(0, 30)
    plt.ylim(1e-5, 1)
    plt.title("BER as a Function of $P_s/(1-P_s)")
    plt.xlabel("$P_s/(1-P_s)$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.legend()
    plt.show()
    # and save results  
    
    output_path = f"/home/dsi/galgreen/tmp/rfchallenge/graphs/ber_plot_M={M}_net={net}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    error_save_path = f"/home/dsi/galgreen/tmp/rfchallenge/graphs/save_error_M={M}_net={net}.npy"
    np.save(error_save_path, save_error)
    print(f"Saved BER matrix to: {error_save_path}")
    error_check_save_path = f"/home/dsi/galgreen/tmp/rfchallenge/graphs/save_error_check_M={M}_net={net}.npy"
    np.save(error_check_save_path, save_error_check)
    print(f"Saved BER matrix to: {error_check_save_path}")
    plt.close()
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check output')
    #parser.add_argument('-l', '--sig_len', default=3200, type=int)
    parser.add_argument('-m', '--M_symbols', default=1, type=int)
    parser.add_argument('-d', '--dataset', default='test', help='')
    parser.add_argument('-t', '--test_set', default='TestSet', help='')
    parser.add_argument('--soi_sig_type',default='OFDMQPSK', help='')
    parser.add_argument('--network',default='unet', help='')
    args = parser.parse_args()

    soi_type = args.soi_sig_type
    dataset_type = args.dataset
    test_set=args.test_set
    Ms=args.M_symbols
    net=args.network
    foldername = os.path.join('dataset', f'Dataset_{soi_type}_{dataset_type}_{test_set}_M={Ms}')
  
    run_demod_test(soi_type,foldername,Ms,net)
    
