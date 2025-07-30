#!/usr/bin/env python3
import numpy as np

# נתיב לקובץ שהכיל את המטריצה (3×30)
error_save_path = "/home/dsi/galgreen/tmp/rfchallenge/graphs/save_error_M=1.npy"

# טענת המטריצה
save_error = np.load(error_save_path)

# שורה 2 היא SNR=10 dB
ber_snr10 = save_error[2]

# הדפסה של כל הוקטור
print("BER for SNR=10 dB:", ber_snr10)

