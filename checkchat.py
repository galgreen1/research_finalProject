import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import os
import math
import argparse
import numpy as np

import matplotlib.pyplot as plt

# Define 4-QAM symbols and bit mapping according to user function
a = 1 / math.sqrt(2)
qam_symbols = tf.constant([
    complex(a, a),  # symbol (0)
    complex(-a, a), # symbol (1)
    complex(-a, -a),# symbol (2)
    complex(a, -a)  # symbol (3)
], dtype=tf.complex64)
torch_qam_bits = tf.constant([
    [0, 0],  # (a, a)
    [0, 1],  # (-a, a)
    [1, 1],  # (-a, -a)
    [1, 0],  # (a, -a)
], dtype=tf.int32)

# SNR values for normalization
sigma_SNR = [math.sqrt(1/30), math.sqrt(1/20), math.sqrt(1/10)]

@tf.function
def compute_symbol_indices(es_com):
    """
    Find index 0-3 of closest 4-QAM symbol for each sample.
    """
    es = tf.convert_to_tensor(es_com, dtype=tf.complex64)
    es_exp = tf.expand_dims(es, axis=-1)  # (..., 1)
    qam_exp = tf.reshape(qam_symbols, [*([1] * len(es.shape)), 4])  # (..., 4)
    dists = tf.abs(es_exp - qam_exp)  # (..., 4)
    return tf.argmin(dists, axis=-1)  # (...)

@tf.function
def compute_ber_tf(sig_est, sig_soi):
    """
    Compute BER using user's bit mapping.
    """
    idx_est = compute_symbol_indices(sig_est)
    idx_soi = compute_symbol_indices(sig_soi)
    est_bits = tf.gather(torch_qam_bits, idx_est)  # (..., 2)
    soi_bits = tf.gather(torch_qam_bits, idx_soi)  # (..., 2)
    errors = tf.not_equal(est_bits, soi_bits)
    total_err = tf.reduce_sum(tf.cast(errors, tf.float32))
    shape=tf.shape(sig_est)
    #tf.print("shape sig_est:", tf.shape(sig_est), summarize=-1)
    nbits = tf.cast( 2*shape[0]*shape[1], tf.float32)
    return total_err / nbits


def calculate_ps_ratios_db():
    ps_ratios_db = np.linspace(0, 30, 30)
    return 10 ** (ps_ratios_db / 10), ps_ratios_db


def compute_P(ps_ratio):
    return 0.0 if ps_ratio == 0 else ps_ratio / (ps_ratio + 1)


def to_tf_complex(arr):
    """
    Convert a NumPy array of shape (..., 2) or complex dtype to tf.complex64.
    """
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return tf.cast(arr, tf.complex64)
    # assume last dim is [real, imag]
    return tf.complex(arr[..., 0], arr[..., 1])


def run_demod_test(soi_type, foldername, M, net):
    ps_ratios, ps_ratios_db = calculate_ps_ratios_db()
    save_error = np.zeros((len(sigma_SNR), len(ps_ratios)), dtype=np.float32)
    save_error_check = np.zeros((len(sigma_SNR), len(ps_ratios)), dtype=np.float32)

    for a, snr_val in enumerate(sigma_SNR):
        for i, ps in enumerate(ps_ratios):
            p = compute_P(ps)
            # Load and convert to tf tensors
            sig_np = np.load(os.path.join(
                foldername,
                f"testsymbols_{soi_type}_p{p:.5f}_SNR{1/(snr_val**2):.2f}.npy"
            ))
            check_np = np.load(os.path.join(
                foldername,
                f"testmixture_{soi_type}_p{p:.5f}_SNR{1/(snr_val**2):.2f}.npy"
            ))
            sig_tf = to_tf_complex(sig_np)
            check_tf = to_tf_complex(check_np)

            # Load estimated
            est_file = f"estimated_soi_{soi_type}_p{p:.5f}_SNR{1/(snr_val**2):.2f}.npy"
            if net == 'Wavenet':
                est_file = f"Wavenet_estimated_soi_{soi_type}_p{p:.5f}_SNR{1/(snr_val**2):.2f}.npy"
            est_np = np.load(os.path.join(foldername, est_file))
            est_tf = to_tf_complex(est_np)

            # Compute BER on GPU
            ber_est = compute_ber_tf(est_tf, sig_tf).numpy()
            ber_sim = compute_ber_tf(check_tf, sig_tf).numpy()
            save_error[a, i] = ber_est
            save_error_check[a, i] = ber_sim
            print(f"error SNR={snr_val}, p={p:.5f}: est={ber_est}, sim={ber_sim}")

    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    labels = [30, 20, 10]
    for idx, label in enumerate(labels):
        color = colors[idx]
        plt.plot(ps_ratios_db, save_error[idx], label=f"Network SNR={label}dB M={M} net={net}", color=color)
        plt.plot(ps_ratios_db, save_error_check[idx], '--', label=f"Simulation SNR={label}dB M={M}", color=color)
    plt.yscale('log'); plt.xlim(0, 30); plt.ylim(1e-5, 1)
    plt.title("BER vs Ps ratio"); plt.xlabel("Ps/(1-Ps) (dB)"); plt.ylabel("BER")
    plt.grid(True); plt.legend()
    os.makedirs(f"/home/dsi/galgreen/tmp/rfchallenge/graphs1", exist_ok=True)
    plt.savefig(
        f"/home/dsi/galgreen/tmp/rfchallenge/graphs1/ber_plot_M={M}_net={net}.png",
        dpi=300
    )
    np.save(
        f"/home/dsi/galgreen/tmp/rfchallenge/graphs1/save_error_M={M}_net={net}.npy",
        save_error
    )
    np.save(
        f"/home/dsi/galgreen/tmp/rfchallenge/graphs/save_error_check_M={M}_net={net}.npy",
        save_error_check
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demodulation with TF')
    parser.add_argument('-m', '--M_symbols', default=1, type=int)
    parser.add_argument('-d', '--dataset', default='test')
    parser.add_argument('-t', '--test_set', default='TestSet')
    parser.add_argument('--soi_sig_type', default='OFDMQPSK')
    parser.add_argument('--network', default='unet')
    args = parser.parse_args()
    folder = os.path.join(
        'dataset',
        f"Dataset_{args.soi_sig_type}_{args.dataset}_{args.test_set}_M={args.M_symbols}"
    )
    run_demod_test(
        args.soi_sig_type, folder, args.M_symbols, args.network
    )

