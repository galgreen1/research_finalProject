import numpy as np
import tensorflow as tf
from scipy.signal import filtfilt, butter
import math
import random

# General Parameters
NFFT = 64
CP_LEN = 16
B = 10e6  # Bandwidth
A = 2 * np.sqrt(2)  # Amplitude
T = NFFT / B
deltaf = B / NFFT
pi = math.pi
taps = 14
NUM_BITS_PER_SYMBOL = 2

# Quantization

def quan(S, l=4):
    S = np.squeeze(S)
    min_val, max_val = np.min(S), np.max(S)
    if min_val == max_val:
        return S
    levels = np.linspace(min_val, max_val, l + 1)
    quantization_levels = (levels[:-1] + levels[1:]) / 2
    S_quantized = np.array([quantization_levels[np.argmin(np.abs(s - quantization_levels))] for s in S], dtype=complex)
    return S_quantized

# Power split helper

def compute_P(ps_ratio):
    return ps_ratio / (1 + ps_ratio) if ps_ratio != 0 else 0

# Sensing signal (FMCW like)

def create_fmcw():
    s = [A * math.cos(pi * (k**2) / NFFT) for k in range(1, NFFT // 2)]
    s_full = [A] + s + [math.cos(pi * (NFFT / 2)**2 / NFFT)] + s[::-1]
    s_transform = np.fft.fft(s_full)
    return s_transform.real

# QAM symbols

def create_qam_symbols(M=1):
    bits = np.random.randint(0, 2, size=(M, NFFT * NUM_BITS_PER_SYMBOL))
    qam = np.zeros((M, NFFT), dtype=complex)
    for i in range(M):
        for j in range(0, NFFT * 2, 2):
            b1, b2 = bits[i, j], bits[i, j + 1]
            if b1 == 0 and b2 == 0:
                qam[i, j // 2] = complex(1, 1)
            elif b1 == 0 and b2 == 1:
                qam[i, j // 2] = complex(-1, 1)
            elif b1 == 1 and b2 == 1:
                qam[i, j // 2] = complex(-1, -1)
            elif b1 == 1 and b2 == 0:
                qam[i, j // 2] = complex(1, -1)
    return qam / np.sqrt(2), bits

# Interpolation

def interpolate_signal(x_d, L1, target_length):
    x_up = np.zeros(len(x_d) * L1, dtype=complex)
    x_up[::L1] = x_d
    fc_LPF = 1 / (2 * L1)
    b, a = butter(N=4, Wn=fc_LPF, btype='low')
    output_signal = filtfilt(b, a, x_up)
    if len(output_signal) > target_length:
        return output_signal[:target_length]
    else:
        return np.pad(output_signal, (0, target_length - len(output_signal)), mode='constant')

# Mix sensing and communication

def combine_sensing_communication(ps_ratio, sensing, com, M, l):
    p = math.sqrt(compute_P(ps_ratio))
    p_c = math.sqrt(1 - compute_P(ps_ratio))
    sensing = quan(sensing, l)
    sensing = np.tile(sensing, (M, 1))
    return p * sensing + p_c * com

# Add CP and IFFT

def add_cp_and_ifft(signal):
    cp = signal[:, -CP_LEN:]
    with_cp = np.concatenate([cp, signal], axis=1)
    return np.fft.ifft(with_cp, axis=1)

# Add noise and channel

def add_channel_and_noise(sig, sigma):
    sig_f = np.fft.fft(sig, axis=1)
    gauss = np.random.normal(0, sigma, sig.shape[1])
    gauss_f = np.fft.fft(gauss)
    h = np.random.normal(0, 1, NFFT) + 1j * np.random.normal(0, 1, NFFT)
    h_interp = interpolate_signal(h, 6, NFFT + CP_LEN)
    h_freq = np.fft.fft(h_interp)
    noise = gauss_f / h_freq
    return sig_f + noise

# Estimate sensing signal

def estimate_sense(ps_ratio, mix, M, l):
    p = math.sqrt(compute_P(ps_ratio))
    est = np.sum(mix, axis=0) / (M * p)
    return quan(est, l)

# Project back to QAM constellation

def compute_closest_levels(es, levels):
    return np.array([levels[np.argmin(np.abs(val - levels))] for val in es], dtype=complex)

# Estimate communication signal

def estimate_com(ps_ratio, signal, estimate_sen, M):
    p = math.sqrt(compute_P(ps_ratio))
    p_c = math.sqrt(1 - compute_P(ps_ratio))
    return (signal - p * estimate_sen) / p_c

# Full pipeline

def generate_isac_signal(sigma=1, M=1, ps_ratio=0.5, l=4):
    sensing = create_fmcw()
    com, bits = create_qam_symbols(M)
    mix = combine_sensing_communication(ps_ratio, sensing, com, M, l)
    with_cp = add_cp_and_ifft(mix)
    noisy = add_channel_and_noise(with_cp, sigma)
    # symbols = np.fft.ifft(noisy, axis=1)[:, CP_LEN:]
    symbols = noisy[:, CP_LEN:]
    es_sense = estimate_sense(ps_ratio, symbols, M, l)
    levels = np.unique(quan(sensing, l))
    es_sense_closest = compute_closest_levels(es_sense, levels)
    es_comm = estimate_com(ps_ratio, symbols, es_sense_closest, M)
    return es_comm, com, bits

# Example usage
if __name__ == "__main__":
    est_comm, true_comm, bits = generate_isac_signal(sigma=1, M=1, ps_ratio=0.5, l=4)
    print("Estimated communication symbol:", est_comm)
    print("True communication symbol:", true_comm)