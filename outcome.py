import numpy as np
import matplotlib.pyplot as plt
import os

# Load BER matrices
save_error_M1 = np.load("/home/dsi/galgreen/tmp/rfchallenge/graphs/save_error_M=1.npy")
save_error_M50 = np.load("/home/dsi/galgreen/tmp/rfchallenge/graphs/save_error_M=50.npy")
data_txt = np.loadtxt('/home/dsi/galgreen/tmp/rfchallenge/outputs/outputs1.txt', skiprows=1)
ps_db = data_txt[:, 0]
ber_30_M50 = data_txt[:, 1]
ber_20_M50 = data_txt[:, 2]
ber_10_M50 = data_txt[:, 3]
ber_10_M1 = data_txt[:, 4]

# Generate x-axis: ps_ratios_db from 0 to 30 dB with 30 points
ps_ratios_db = np.linspace(0, 30, save_error_M1.shape[1])

# Prepare the plot
plt.figure(figsize=(10, 6))

# Plot each SNR line for M=50
plt.plot(ps_ratios_db, save_error_M50[0], label="SNR=30dB, M=50 network", color='blue', linestyle='-')
plt.plot(ps_ratios_db, save_error_M50[1], label="SNR=20dB, M=50 network", color='green', linestyle='-')
plt.plot(ps_ratios_db, save_error_M50[2], label="SNR=10dB, M=50 network", color='red', linestyle='-')

# Plot corresponding SNR lines for M=1
#plt.plot(ps_ratios_db, save_error_M1[0], label="SNR=30dB, M=1", color='blue', linestyle='--')
#plt.plot(ps_ratios_db, save_error_M1[1], label="SNR=20dB, M=1", color='green', linestyle='--')
plt.plot(ps_ratios_db, save_error_M1[2], label="SNR=10dB, M=1 network", color='purple', linestyle='-')

plt.plot(ps_db, ber_30_M50, '--', label='SNR=30dB, M=50 simulation', color='blue')
plt.plot(ps_db, ber_20_M50, '--', label='SNR=20dB, M=50 simulation', color='green')
plt.plot(ps_db, ber_10_M50, '--', label='SNR=10dB, M=50 simulation', color='red')
plt.plot(ps_db, ber_10_M1, '--', label='SNR=10dB, M=1 simulation', color='purple')

# Log scale for y-axis
plt.yscale("log")
plt.xlim(0, 30)
plt.ylim(1e-5, 1)

# Titles and labels
plt.title("BER Comparison")
plt.xlabel("$P_s / (1 - P_s)$ [dB]")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both")
plt.legend()

# Save the plot
output_path = "/home/dsi/galgreen/tmp/rfchallenge/graphs/ber_plot.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved plot to: {output_path}")

plt.show()
