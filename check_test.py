import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import filtfilt, butter
from tqdm import tqdm



# Global variables
N = 64  # Number of subcarriers
B = 10 * 10**6  # Bandwidth and sampling frequency [Hz]
T = N / B  # Symbol duration [s]
deltaf = B / N  # Carrier frequency spacing [Hz]
# M = 50  # Number of symbols collected to estimate the sensing signal
#L = 4  # Quantization level for sensing
pi = math.pi
A = 1  # Signal amplitude
f = 0  # Baseband frequency
fc = f  # Carrier frequency
CP = 16  # Cyclic prefix length
taps = 14  # Number of taps in the Rayleigh fading channel


# compute P
def compute_P(ps_ratio):
    if ps_ratio==0:
        return 0
    a=ps_ratio
    P=a/(a+1)

    #print(P)
    return P


# discrete sensing signal
def create_fmcw():
    # variables:
    # B: band width of sensing signal , N:sample number , N=BT-> T=N/B
    s = [0.0]*N
    for k in range(1,N//2):
        # s[k] = A*math.cos(2*pi*k*f/B + pi*(k**2)/N) and f=0
        s[k]=A*math.cos(pi*(k**2)/N)
        s[N-k]=s[k]
    s[0]=A*math.cos(0)
    s[N//2]=math.cos(pi*((N/2)**2)/N)
    #s[0]=0
    #s[N//2]=0

    s_transform = np.fft.fft(s)  # FFT

    # graph check
    # real_part = np.abs(s_transform.real)
    # imag_part = np.abs(s_transform.imag)
    # plt.figure(figsize=(10, 6))
    # plt.plot(real_part, label="Real Part (Magnitude)", color="blue")
    # plt.plot(imag_part, label="Imaginary Part (Magnitude)", color="orange")
    # plt.title("Magnitude of Real and Imaginary Parts of s_k")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Magnitude")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # there are imaginary parts that small from 10^-6 so we will ignore them!!
    return s_transform.real


def create_communication_symbol(M):
    # 64 4-QAM symbols
    # 00:1+i  01:-1+i  10:-1-i  11:1-i
    bits_num=N*2
    com=np.zeros(bits_num)
    com_mat=np.zeros((M,N),dtype=complex)
    for i in range(M):  # for every line
        for j in range(bits_num):
            com[j] = random.randint(0, 1)
        # print(average_power(com))
        for j in range(0, bits_num-1, 2):
            if com[j] == 0 and com[j + 1] == 0:
                com_mat[i][j // 2] = complex(1, 1)
            elif com[j] == 0 and com[j + 1] == 1:
                com_mat[i][j // 2] = complex(-1, 1)
            elif com[j] == 1 and com[j + 1] == 1:
                com_mat[i][j // 2] = complex(-1, -1)
            elif com[j] == 1 and com[j + 1] == 0:
                com_mat[i][j // 2] = complex(1, -1)
    #print(com_mat)            

    return com_mat  # shape M,N


def quan(S,l):
    min=S[0]
    max=S[0]
    for i in range(len(S)):
        if S[i]<min:
            min=S[i]
        elif S[i]>max:
            max=S[i]
    # same level for all
    if min==max:
        return S
    levels = np.linspace(min, max, l + 1)  # L parts
    # print(levels)
    quantization_levels = (levels[:-1] + levels[1:]) / 2
    # print(quantization_levels)

    S_quantized = np.zeros(len(S),dtype=complex)
    for i in range(len(S)):
        distances = np.abs(S[i] - quantization_levels)  #
        S_quantized[i] = quantization_levels[np.argmin(distances)]  #
    return S_quantized


def combine_sensing_communication(ps_ratio, S, com,l,M):
    # compute ps
    p1=compute_P(ps_ratio)
    p=math.sqrt(p1)
    p_c=math.sqrt(1-p1)
    # S=create_fmcw()
    sense=quan(S,l)
    sense = np.tile(sense, (M, 1))
    # print(sense)
    # com=create_communication_symbol()
    combined=p*sense+p_c*com
    # combined=np.zeros((N,N),dtype=complex)
    # for i in range(N):
    #     for j in range(N):
    #         combined[i][j]=p*sense[j]+p_c*com[i][j]
    return combined


def add_CP(ps_ratio,S,com,l,M):
    sig=combine_sensing_communication(ps_ratio,S,com,l,M)
    #print(f"sig[1]: {sig[1]} ")
    added=N+CP
    addedMat=np.zeros((M,added),dtype=complex)
    for i in range(M):
        for j in range(added):
            if j<CP:
                addedMat[i][j]=sig[i][N-CP+j]
            else:
                addedMat[i][j]=sig[i][j-CP]
    # print(f"line 1 before:{sig[1]} and after: {addedMat[1]}")
    #print(f"sigCP[1]:{addedMat[1]}")
    addedMat = np.fft.ifft(addedMat, axis=1)  #OFDM
    return addedMat


# add gauss noise
def add_noise(ps_ratio,sigma,S,com,l,M):
    sig=add_CP(ps_ratio,S,com,l,M)  # length=N+CP
    sig=np.fft.fft(sig,axis=1)  #FFT
    gaussian_noise_time = np.random.normal(0, sigma, N+CP)
    gaussian_noise_freq = np.fft.fft(gaussian_noise_time)
    # channel_response = rayleigh.rvs(scale=1, size=taps)
    #try to do that by myself
    sigma_ry=20*2/(4-math.pi)
    h_real = np.random.normal(0, sigma_ry / np.sqrt(2), N)  # Real part
    h_imag = np.random.normal(0, sigma_ry / np.sqrt(2), N)  # Imaginary part

    # Combine real and imaginary parts to form complex coefficients
    h = h_real + 1j * h_imag  # fix to add zeros in time

    channel_response=h
    channel_response_fixed=interpolate_signal(channel_response,6,len(gaussian_noise_time))

    channel_response_freq = np.fft.fft(channel_response_fixed)
    noise=gaussian_noise_freq/channel_response_freq
    noise = np.tile(noise, (M, 1))
    # divide in H of the channel
    signal=sig+noise
    # signal=sig+gaussian_noise_freq
    return signal


def interpolate_signal(x_d, L1, target_length):
    # Add L-1 zeros between every 2 samples

    x_up = np.zeros(len(x_d) * L1, dtype=complex)
    x_up[::L1] = x_d

    # Create LPF
    fc_LPF = 1 / (2 * L1)  # Normalized cutoff frequency
    b, a = butter(N=4, Wn=fc_LPF, btype='low', analog=False)  # Order 4
    output_signal = filtfilt(b, a, x_up)  # Zero-phase filtering
    #print(len(output_signal))

    # Adjust length
    if len(output_signal) > target_length:
        output_signal = output_signal[:target_length]
    elif len(output_signal) < target_length:
        output_signal = np.pad(output_signal, (0, target_length - len(output_signal)), mode='constant')

    return output_signal


def computeCloset(es_com):
    qam_symbols = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
    qam_symbols = qam_symbols.reshape((1, 1, -1))
    closest_symbols = np.zeros_like(es_com, dtype=complex)
    # Compute Euclidean distances for all elements in es_com to all 4-QAM symbols
    distances = np.abs(es_com[:, :, np.newaxis] - qam_symbols)  # Shape: (MxN, 4)

    # Find the index of the minimum distance for each element
    closest_symbol_indices = np.argmin(distances, axis=2)  # Shape: (MxN)

    # Map indices back to symbols
    closest_symbols = qam_symbols[0, 0, closest_symbol_indices]
    return closest_symbols


def computeCloset_sense(es_sen,S):
    unique_levels = np.unique(S)
    S_mapped = np.zeros(len(es_sen), dtype=complex)
    for i in range(len(es_sen)):
        distances = np.abs(es_sen[i] - unique_levels)
        S_mapped[i] = unique_levels[np.argmin(distances)]
    return S_mapped


def estimate_sense(ps_ratio,signal1,m,l):
    # compute ps
    p1=compute_P(ps_ratio)
    p=math.sqrt(p1)
    if p==0:
        return 0
    sigmay=np.zeros(N, dtype=complex)
    for i in range(m):
        sigmay+=signal1[i]
    x=sigmay/(m*p)
    # return x
    return quan(x,l)


def estimate_com(ps_rati,signal_co,estimate_sen,M):
    p1 = compute_P(ps_rati)
    p = math.sqrt(p1)
    p_c=math.sqrt(1-p1)
    #print(len(signal))
    comm=np.zeros((M,N),dtype=complex)
    for i in range(M):
        #print(len(signal_co[i]))
        #print(f"signal[i]:{signal[i]}\n p:{p}\n esti sense:{estimate_sense}\n")
        comm[i] = (signal_co[i] - p * estimate_sen) / p_c
        #print(f"comm[i]:{comm[i]}")
    return comm


def calculate_ps_ratios_db():
    ps_ratios_db = np.linspace(0, 30, 30)  # Extend range to 30 dB
    ps_ratios = 10 ** (ps_ratios_db / 10)
    ps_values = ps_ratios / (1 + ps_ratios)
    return ps_ratios, ps_ratios_db


def convert_qam_to_bits(x):
    if x==complex(1,1):
        return 0,0
    if x==complex(-1,1):
        return 0,1
    if x==complex(-1,-1):
        return 1,1
    if x==complex(1,-1):
        return 1,0


def simulation(sigma,m,l):
    ps_ratios, ps_ratios_db = calculate_ps_ratios_db()
    ber_com_results = []

    ber_sen_results = []
    times=8192
    with tqdm(total=245760, desc="Total Progress") as pbar:
        for ps_ratio in ps_ratios:
            # ps_ratio=math.pow(10,-1)
            ber_com = 0
            ber_sen = 0
            # do times each value
            for a in range(times):
                S = create_fmcw()
                com = create_communication_symbol(m)
                #print(f'com:{com.shape}')
                signal_sim = add_noise(ps_ratio, sigma, S, com,l,m)
                # signal_sim =combine_sensing_communication(ps_ratio,S,com)
                # signal_sim=add_CP(ps_ratio,S, com)
                # print(signal_sim)
                # return to 64
                addedMat = np.zeros((m, N), dtype=complex)
                for i in range(m):
                    for j in range(N):
                        if j < N - CP:
                            addedMat[i][j] = signal_sim[i][CP + j]
                        else:
                            addedMat[i][j] = (signal_sim[i][j + CP] + signal_sim[i][j - N + CP]) / 2

                          

                # es_sense = estimate_sense(ps_ratio, addedMat)
                # es_com = estimate_com(ps_ratio, addedMat, es_sense)
                es_sense = estimate_sense(ps_ratio, addedMat,m,l)

                es_sense = computeCloset_sense(es_sense, quan(S,l))
                es_com = estimate_com(ps_ratio, addedMat, es_sense,m)
                #es_com = estimate_com(ps_ratio, addedMat, quan(S,l))
                closest_symbols = computeCloset(es_com)

                # print(closest_symbols)
                # print(com)
                check_com = np.zeros_like(com, dtype=float)
                for i in range(m):
                    for j in range(N):
                        a1,a2=convert_qam_to_bits(closest_symbols[i][j])
                        b1,b2=convert_qam_to_bits(com[i][j])
                        x1=a1-b1
                        x2=a2-b2
                        #x1, x2 = convert_qam_to_bits(closest_symbols[i][j]) - convert_qam_to_bits(com[i][j])
                        if x1 != 0 and x2 != 0:
                            check_com[i][j] = 2
                            # print("2:")
                            # print(convert_qam_to_bits(closest_symbols[i][j]))
                            # print(convert_qam_to_bits(com[i][j]))

                        else:
                            if x1 != 0 or x2 != 0:
                                check_com[i][j] = 1
                                # print("1:")
                                # print(convert_qam_to_bits(closest_symbols[i][j]))
                                # print(convert_qam_to_bits(com[i][j]))
                            else:
                                check_com[i][j] = 0
                                # print("0:")
                                # print(convert_qam_to_bits(closest_symbols[i][j]))
                                # print(convert_qam_to_bits(com[i][j]))


                # print(check_com)
                #check_sense = es_sense - quan(S,l)
                # print(check_com)
                # print(check_sense)
                sum_sen = 0
                sum_com = 0
                # for i in range(N):
                #     if np.abs(check_sense[i]) != 0:  # in case of little mistakes
                #         sum_sen = sum_sen + 1
                for i in range(m):
                    # if np.abs(check_sense[i]) > 1e-3:  # in case of little mistakes
                    #if np.abs(check_sense[i]) != 0:  # in case of little mistakes
                        #sum_sen = sum_sen + 1
                    for j in range(N):
                        sum_com=sum_com+check_com[i][j]
                        # if np.abs(check_com[i][j]) != 0:
                        #     # print(f"orig:{com[i][j]} and {closest_symbols[i][j]}")
                        #     sum_com = sum_com + 1
                # ber_sen_results.append(sum_sen / N)
                # ber_com_results.append(sum_com / N ** 2)
                # print(sum_com / N ** 2)
                # print(f"sum com:{sum_com} so {(sum_com / ((2*N) ** 2))}")
                ber_com = ber_com + (sum_com / ((2*N*m)))
                #ber_sen = ber_sen + (sum_sen / N)
                pbar.update(1)
                # print(f"ps ratio:{ps_ratio} bercom:{sum_com / N ** 2}")
                # print(f"error in sense:{sum_sen / N}")
                # print(f"error in com:{sum_com / N ** 2}")
            ber_com_results.append(ber_com / times)
            #ber_sen_results.append(ber_sen / times)

    #return ber_com_results,ber_sen_results
    return ber_com_results


def graph1():
    l=4
    # sigma = math.sqrt(1 / 30)
    # x1= simulation(sigma, 50,l)
    # sigma = math.sqrt(1 / 20)
    # a1= simulation(sigma, 50,l)
    # sigma = math.sqrt(1 / 10)
    # z1= simulation(sigma, 50,l)
    sigma = math.sqrt(1 / 10)
    c1= simulation(sigma, 1,l)
    ps_ratios, ps_ratios_db = calculate_ps_ratios_db()

    # plot
    plt.figure(figsize=(10, 6))

    # plt.plot(ps_ratios_db, x1, 'b-o', label=f"Com: SNR=30dB M=50  L={l}", color='blue')
    # plt.plot(ps_ratios_db, a1, 'b-o', label=f"Com: SNR=20dB M=50  L={l}", color='green')
    # plt.plot(ps_ratios_db, z1, 'b-o', label=f"Com: SNR=10dB M=50  L={l}", color='red')
    plt.plot(ps_ratios_db, c1, 'b-o', label=f"Com: SNR=10dB M=1  L={l}", color='yellow')
    # plt.plot(ps_ratios_db, ber_sen_results, 'r-s', label="Sen: SNR=30dB")

    plt.yscale("log")
    plt.xlim(0, 30)
    plt.ylim(1e-5, 1)
    plt.title("BER as a Function of $P_s/(1-P_s)$", fontsize=16)
    plt.xlabel("$P_s/(1-P_s)$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.legend(loc='best')
    output_path = "outputs/ber_plot1_check.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

  
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    graph1()
    #graph2()
    



# See PyCharm help at https://www.jetbrains.com/help/pycharm/