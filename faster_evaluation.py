import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import os
import math
import argparse

import numpy as np

import matplotlib.pyplot as plt

# 4-QAM constellation
QAM_SYMBOLS = tf.constant(
    [-1-1j, -1+1j,  1-1j,  1+1j],
    dtype=tf.complex64
)
BITS_TABLE = tf.constant([
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 0],
], dtype=tf.int32)


def compute_closest_indices(es):
    diff = tf.abs(es[..., tf.newaxis] - QAM_SYMBOLS[tf.newaxis, tf.newaxis, :])
    return tf.argmin(diff, axis=-1)


def compute_ber_tensorflow(mixture, symbols):
    #mix = tf.complex(mixture[:,:,0], mixture[:,:,1])
    #soi = tf.complex(symbols[:,:,0], symbols[:,:,1])
    soi=symbols
    idx_mix = compute_closest_indices(mixture)
    idx_soi = compute_closest_indices(soi)
    #idx_soi=soi
    bits_mix = tf.gather(BITS_TABLE, idx_mix)
    bits_soi = tf.gather(BITS_TABLE, idx_soi)
    bit_errors = tf.cast(tf.not_equal(bits_mix, bits_soi), tf.int32)
    total_errors = tf.reduce_sum(bit_errors)
    M, N = tf.shape(idx_mix)[0], tf.shape(idx_mix)[1]
    total_bits = 2 * M * N
    ber = tf.cast(total_errors, tf.float32) / tf.cast(total_bits, tf.float32)
    return ber.numpy()


def run_demod_test(soi_type, dataset, M, net):
    print("TF devices:", tf.config.list_physical_devices('GPU'))

    sigma_SNR = [math.sqrt(1/10), math.sqrt(1/20), math.sqrt(1/30)]
    #sigma_SNR = [math.sqrt(1/10)]
    ps_db = np.linspace(0, 30, 24)
    ps_ratios = 10 ** (ps_db / 10)

    folder = os.path.join('dataset', f"Dataset_{soi_type}_{dataset}_Mixture_M={M}")
    out_dir = os.path.join('graphs', f"BER_{soi_type}_{dataset}_M={M}_net={net}")
    os.makedirs(out_dir, exist_ok=True)

    ber_mix_mat = np.zeros((len(sigma_SNR), len(ps_ratios)), dtype=np.float32)
    ber_est_mat = np.zeros_like(ber_mix_mat)

    for a, sigma in enumerate(sigma_SNR):
        #snr_db = 10*math.log10(1/(sigma**2))
        snr_db=1/sigma**2
        for i, ps in enumerate(ps_ratios):
            p = ps / (ps + 1)
            fn_sym = f"testsymbols_{soi_type}_p{p:.5f}_SNR{snr_db:.2f}.npy"
            fn_mix = f"testmixture_{soi_type}_p{p:.5f}_SNR{snr_db:.2f}.npy"
            
            fn_est = (f"Wavenet_estimated_soi_{soi_type}_p{p:.5f}_SNR{snr_db:.2f}.npy"
                      if net.lower()=='wavenet'
                      else f"estimated_soi_{soi_type}_p{p:.5f}_SNR{snr_db:.2f}.npy")

            sym = np.load(os.path.join(folder, fn_sym))
            real=sym[:,:,0]
            imag=sym[:,:,1]
            sym=real+1j*imag
            mix = np.load(os.path.join(folder, fn_mix))
            #print(f'mix shape{mix.shape}')
            real_mix=mix[:,:,0]
            imag=mix[:,:,1]
            mix=real_mix+1j*imag
            est = np.load(os.path.join(folder, fn_est))

            ber_mix = compute_ber_tensorflow(mix, sym)
            ber_est = compute_ber_tensorflow(est, sym)

            ber_mix_mat[a,i] = ber_mix
            ber_est_mat[a,i] = ber_est

            print(f"SNR={snr_db:.1f}dB p={p:.5f} → mix={ber_mix:.3e} est={ber_est:.3e}")

    # שמירת המטריצות
    np.save(os.path.join(out_dir, "ber_mix.npy"), ber_mix_mat)
    np.save(os.path.join(out_dir, "ber_est.npy"), ber_est_mat)

    # ציור ושמירה של הגרף
    plt.figure(figsize=(8,5))
    for a, sigma in enumerate(sigma_SNR):
        snr_db = 10*math.log10(1/(sigma**2))
        plt.semilogy(ps_db, ber_mix_mat[a], '--', label=f"Simulation M={M} SNR={snr_db:.0f}dB ")
        plt.semilogy(ps_db, ber_est_mat[a], '-',  label=f"Network={net} M={M} SNR={snr_db:.0f}dB")

    plt.xlabel(r"$P_s/(1-P_s)$ (dB)")
    plt.ylabel("BER")
    plt.title(r"BER vs. $P_s/(1-P_s)$")
    plt.grid(True, which="both")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.ylim(1e-6,1)
    plt.xlim(0,30)
    plt.tight_layout()

    graph_path = os.path.join(out_dir, "ber_plot.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Saved plot to {graph_path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--soi_sig_type', default='OFDMQPSK')
    p.add_argument('-d','--dataset', default='test')
    p.add_argument('-m','--M_symbols', type=int, default=50)
    p.add_argument('--network', default='unet')
    args = p.parse_args()

    run_demod_test(
        soi_type = args.soi_sig_type,
        dataset  = args.dataset,
        M        = args.M_symbols,
        net      = args.network,
    )
