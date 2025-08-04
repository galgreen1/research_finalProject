import os
import math
import numpy as np
import matplotlib.pyplot as plt


out_dir = "graphs/BER_OFDMQPSK_test_M=50_net=unet"  


ber_mix = np.load(os.path.join(out_dir, "ber_mix.npy"))   # shape = (num_snr, num_ps)
ber_est = np.load(os.path.join(out_dir, "ber_est.npy"))   # same shape


sigma_SNR = [math.sqrt(1/10),math.sqrt(1/20),math.sqrt(1/30)]
ps_db = np.linspace(0, 30, ber_mix.shape[1])  


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(8,5))
for a, sigma in enumerate(sigma_SNR):
    snr_db = 1/(sigma**2)
    c = colors[a % len(colors)]
   
    plt.semilogy(ps_db, ber_mix[a],
                 linestyle='--', color=c,
                 label=f"Simulation M=50 SNR={snr_db:.0f}dB")
  
    plt.semilogy(ps_db, ber_est[a],
                 linestyle='-', color=c,
                 label=f"Network=unet M=50 SNR={snr_db:.0f}dB")

plt.xlabel(r"$P_s/(1-P_s)$ (dB)")
plt.ylabel("BER")
plt.title("BER vs. $P_s/(1-P_s)$")
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.legend(
    bbox_to_anchor=(1.02, 1),   
    loc='upper left',           
    fontsize='x-small',
    frameon=False
)

plt.xlim(0,30)
plt.ylim(1e-6,1)
plt.tight_layout()

out_dir = "graphs/final_results"  
plt.savefig(os.path.join(out_dir, "ber_plot_M=50.png"), dpi=300)
plt.show()
