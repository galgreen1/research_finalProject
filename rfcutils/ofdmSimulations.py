import sionna as sn
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter

NFFT = 64
CP_LEN = 16
OFDM_LEN = NFFT + CP_LEN
CODERATE = 1
n_streams_per_tx = 1
# Global variables
B = 10 * 10**6  # Bandwidth and sampling frequency [Hz]
T = NFFT / B  # Symbol duration [s]
deltaf = B / NFFT  # Carrier frequency spacing [Hz]
pi = math.pi
A = 2  # Signal amplitude
f = 0  # Baseband frequency
fc = f  # Carrier frequency
taps = 14  # Number of taps in the Rayleigh fading channel

# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()

# 4-QAM constellation
NUM_BITS_PER_SYMBOL = 2
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=False) # The constellation is set to be NOT trainable
stream_manager = sn.mimo.StreamManagement(np.array([[1]]), 1)

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)



def quan(S,l=4):
    S=np.squeeze(S)
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



# create sensing signal length N
def create_fmcw():
    # variables:
    # B: band width of sensing signal , N:sample number , N=BT-> T=N/B
    s = [0.0]*NFFT
    for k in range(1,NFFT//2):
        # s[k] = A*math.cos(2*pi*k*f/B + pi*(k**2)/N) and f=0
        s[k]=A*math.cos(pi*(k**2)/NFFT)
        s[NFFT-k]=s[k]
    s[0]=A*math.cos(0)
    s[NFFT//2]=math.cos(pi*((NFFT/2)**2)/NFFT)
    #s[0]=0
    #s[N//2]=0
    #print(f'fmcw:{s}')


    s_transform = np.fft.fft(s)  # FFT
    s_transform=s_transform.real
    #print(f'sense:{s_transform}')

    return s_transform
    # cp = s_transform[-CP_LEN:]
    # with_cp = np.concatenate([cp, s_transform])
    # return tf.convert_to_tensor(with_cp, dtype=tf.float32)   # convert to tf


# compute P
def compute_P(ps_ratio):
    if ps_ratio==0:
        return 0
    a=ps_ratio
    P=a/(a+1)

    #print(P)
    return P

def create_rayleigh_channel(taps=14, std=1):
    h = tf.complex(
        tf.random.normal([taps], mean=0.0, stddev=std/np.sqrt(2)),
        tf.random.normal([taps], mean=0.0, stddev=std/np.sqrt(2))
    )
    return h


def interpolate_signal(x_d, L1, target_length):
    # Add L-1 zeros between every 2 samples
    x_d=x_d.numpy()

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
    return tf.convert_to_tensor(output_signal, dtype=tf.complex64)



def computeCloset_sense(es_sen,S):
    S=np.squeeze(S)
    unique_levels = np.unique(S)
    S_mapped = np.zeros(len(es_sen), dtype=complex)
    for i in range(len(es_sen)):
        distances = np.abs(es_sen[i] - unique_levels)
        S_mapped[i] = unique_levels[np.argmin(distances)]
    return S_mapped


def estimate_sense(ps_ratio,signal1,m,l,batch_size):
    # compute ps
    #print("signal1 shape:", signal1.shape)

    p1=compute_P(ps_ratio)
    p=math.sqrt(p1)
    if p==0:
        return 0
    sigmay=np.zeros_like(signal1, dtype=complex)
    result = tf.reduce_sum(signal1, axis=[0, 1])
    x=result/(m*batch_size*p)
    # return x
    return quan(x,l)


def estimate_com(ps_rati,signal_co,estimate_sen,M,batch_size):
    sen_expanded = tf.reshape(estimate_sen, (1, 1, NFFT))           
    sen_tiled = tf.tile(sen_expanded, [batch_size, M, 1])     
    sen_tiled = tf.cast(sen_tiled, tf.complex64)
    p1 = compute_P(ps_rati)
    p = math.sqrt(p1)
    p_c=math.sqrt(1-p1)
    #print(len(signal))
    #print(f'signal co:{signal_co} sen :{sen_tiled}  p_c:{p_c}  sum:{signal_co-p*sen_tiled}')
    comm=(signal_co-p*sen_tiled)/p_c
    #print(f'xomm:{comm}')
    return comm    



# # The encoder maps information bits to coded bits
# encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)


def get_resource_grid(num_ofdm_symbols):
    RESOURCE_GRID = sn.ofdm.ResourceGrid( num_ofdm_symbols=num_ofdm_symbols,
                                          fft_size=NFFT,
                                          subcarrier_spacing=10e6/NFFT,
                                          num_tx=1,
                                          num_streams_per_tx=n_streams_per_tx,
                                          num_guard_carriers=(0,0),
                                          dc_null=False,
                                          cyclic_prefix_length=CP_LEN,
                                          pilot_pattern=None,
                                          pilot_ofdm_symbol_indices=[])
    return RESOURCE_GRID


def generate_ofdm_signal_sim(std_gauss,ps_ratio,batch_size, num_ofdm_symbols, ebno_db=None,l=4):
    RESOURCE_GRID = get_resource_grid(num_ofdm_symbols)

    # Number of coded bits in a resource grid
    n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL)
    # Number of information bits in a resource groud
    k = int(n*CODERATE)

    bits = binary_source([batch_size, 1, n_streams_per_tx, k])
    return modulate_ofdm_signal_sim(std_gauss,ps_ratio,bits, RESOURCE_GRID, ebno_db,l)


def ofdm_demod(sig, RESOURCE_GRID, no=1e-4):
    rg_demapper = sn.ofdm.ResourceGridDemapper(RESOURCE_GRID, stream_manager)
    ofdm_demod_block = sn.ofdm.OFDMDemodulator(NFFT, 0, CP_LEN)

    x_ofdm_demod = ofdm_demod_block(sig)
    x_demod = rg_demapper(tf.reshape(x_ofdm_demod, (sig.shape[0],1,1,-1,NFFT)))
    llr = demapper([x_demod,no])
    llr = tf.squeeze(llr, axis=[1,2])
    return tf.cast(llr > 0, tf.float32), x_ofdm_demod


def modulate_ofdm_signal_sim(std_gauss,ps_ratio, info_bits, RESOURCE_GRID, ebno_db=None,l=4):
    # codewords = encoder(info_bits) # using uncoded bits for now
    codewords = info_bits
    rg_mapper = sn.ofdm.ResourceGridMapper(RESOURCE_GRID)
    ofdm_mod = sn.ofdm.OFDMModulator(RESOURCE_GRID.cyclic_prefix_length)



    x = mapper(codewords)
    QAM_s=x

    

    # qam_with_cp = []
    M = RESOURCE_GRID.num_ofdm_symbols
    QAM_s = tf.squeeze(QAM_s, axis=[1, 2])
    #print(f'QAM:{QAM_s}')
    # symbols_per_ofdm_symbol = NFFT
    # print("symbol num:")
    # print(symbols_per_ofdm_symbol)

    # for i in range(M):
    #     start_idx = i * symbols_per_ofdm_symbol
    #     end_idx = (i + 1) * symbols_per_ofdm_symbol
    #     ofdm_symbol = QAM_s[:, start_idx:end_idx]  
    #     cp = QAM_s[:,-CP_LEN:]
    #     ofdm_symbol_with_cp = tf.concat([cp, ofdm_symbol], axis=1)
    #     qam_with_cp.append(ofdm_symbol_with_cp)

    # qam_with_cp = tf.stack(qam_with_cp, axis=0)
    # target_reshaped = tf.reshape(qam_with_cp, (1, -1))


    # print("target size:")
    # print(target_reshaped.shape)
    




    # add sense to every ofdm symbol
    
    batch_size = x.shape[0]
    x=tf.reshape(x,(batch_size,1,1,M,NFFT))

    sensing = create_fmcw()  # shape: [NFFT], real\ plt.figure(figsize=(10, 4))
#     plt.figure(figsize=(10, 6))
#     plt.plot(sensing)
#     plt.title("FFT of FMCW-like Signal (Real Part)")
#     plt.xlabel("Frequency Bin")
#     plt.ylabel("Amplitude")
#     plt.grid(True)
#     plt.tight_layout()

#     output_path = "/home/dsi/galgreen/tmp/rfchallenge/rfcutils/fmcw_fft.png"
#     plt.savefig(output_path)

    #print(f'sensing:{sensing}')
    sense=quan(sensing,l)
    sensing=sense
    #print(f'sense quan:{sense}')
    #sense = np.tile(sense, (M, 1))
    sensing=sensing.real
    
   


    sensing = tf.complex(sensing, tf.zeros_like(sensing))  # shape: [NFFT], complex
    sensing = tf.reshape(sensing, (1, 1, 1,1, NFFT))  # broadcastable
    sensing = tf.tile(sensing, (batch_size,1,1,M,1))  # shape: like x
    sensing = tf.cast(sensing, dtype=tf.complex64)
    #print(f'sensing:{sensing}')
    #print(sensing.shape)
    p1 = compute_P(ps_ratio)
    p = math.sqrt(p1)
    p_c = math.sqrt(1 - p1)
    #print(f'sensing:{sensing}')
    x= p_c * x+ p * sensing
    #print(x.shape)
    x = tf.reshape(x, [batch_size, 1, 1, M *NFFT])
    #print(x.shape)
   
    #print(f'x before cp:{x}')
    # x_rg = rg_mapper(x)
    # x_ofdm = ofdm_mod(x_rg)
   

    x = tf.reshape(x, [batch_size, M, NFFT])  
    cp = x[:, :, -CP_LEN:]              
    x_cp = tf.concat([cp, x], axis=2)   
    x_cp = tf.reshape(x_cp, [batch_size, M * (NFFT + CP_LEN)])
    #print(f'x:{x} x cp:{x_cp}')
    # print(f"line 1 before:{sig[1]} and after: {addedMat[1]}")
    #print(f"sigCP[1]:{addedMat[1]}")

    
    x_cp = np.fft.ifft(x_cp, axis=1)  #OFDM
    x_ofdm=tf.signal.fft(x_cp)
    #print(f'x:{x}')
    #print(f'x fft:{x_ofdm}')

    #print("x_cp shape:", x_cp[:,CP_LEN:].shape)
    #print("QAM_s shape:", QAM_s.shape)
    



    
    #print(f'ofdm:{x_ofdm}')
    #print(x_ofdm)


    # now we pass a rayleigh channel and gauss noise
    # after FFT, and the single tap equalizing for the channel
    sigma_ry=32*2/(4-math.pi)
    h = create_rayleigh_channel(taps,sigma_ry)
    h_interp=interpolate_signal(h,4,NFFT+CP_LEN)
    #h_interp = resample(h, NFFT+CP_LEN)
    
    h_freq=np.fft.fft(h_interp)
    #print(f'h {h_freq}')

    
    #h_freq=np.fft.fft(h_interp)
    #print(f'h_freq:{h_freq}')
    gauss_noise=np.random.normal(loc=0.0, scale=std_gauss, size=NFFT+CP_LEN)
    gauss=np.fft.fft(gauss_noise)
    #print(f'gauss:{gauss}')
    #equalization
    
    noise=gauss/h_freq
    noise = tf.cast(noise, dtype=tf.complex64)
    #print(f'noise:{noise}')
    noise=tf.tile(noise,[M])
   

    #print(f'noise:{noise}')

    mix=x_ofdm+noise
    #mix=tf.squeeze(mix, axis=[1,2])
    #x_rg = tf.squeeze(x_rg, axis=[1, 2])
    info_bits = tf.squeeze(info_bits, axis=[1, 2])

    #print(mix.shape)
    # now to override the cp!
    sig_without_cp = []
    symbols_per_ofdm_symbol = NFFT+CP_LEN

    for i in range(M):
        start_idx = i * symbols_per_ofdm_symbol
        end_idx = (i + 1) * symbols_per_ofdm_symbol
        #print(f'mix:{mix[:,10]}')
        ofdm_symbol = mix[:, start_idx:end_idx]  
        #print(f'ofdm sym:{ofdm_symbol}')
        cp1 = ofdm_symbol[:,0:CP_LEN]
        cp2=ofdm_symbol[:,NFFT:]
        #print(f'cp1:{cp1}')
        #print(f'cp2:{cp2}')
        cp=(cp1+cp2)/2
        #print(f'cp:{cp}')
        part=ofdm_symbol[:,CP_LEN:NFFT]
        ofdm_symbol1 = tf.concat([part,cp], axis=1)
        sig_without_cp.append(ofdm_symbol1)


    sig_without_cp = tf.stack(sig_without_cp, axis=0)
    mix_final = tf.reshape(sig_without_cp, (1, -1))
    #print(f'mix :{mix_final}')
    # for value in mix_final.numpy().flatten():
    #     if value>11:
    #         #print(value)
    
    #print(f'mix without cp:{mix_final}')


    #print("target size:")
    #print(mix_final.shape)
    mix_final = tf.reshape(mix_final, (batch_size, M, NFFT))
    
    es_sense=estimate_sense(ps_ratio, mix_final,M,l,batch_size)  
    #print(f'es sense:{es_sense}')
    es_sense = computeCloset_sense(es_sense, quan(sense,l))
    #print(f'closet sense:{es_sense}')
    es_com = estimate_com(ps_ratio, mix_final, es_sense,M,batch_size)  
    #print(f'es com:{es_com}')
    es_com= tf.reshape(es_com, (batch_size, M * NFFT))
    #print(f'es com:{es_com.shape}')
    



    return es_com,info_bits,RESOURCE_GRID,QAM_s
    # final is size 1 on BATCHSIZE*M*NFFT


if __name__ == "__main__":
   
    mix,info_bits,RESOURCE_GRID,QAM_s=generate_ofdm_signal_sim(math.sqrt(1/100),0.5,1,1)
#     print("es com:")
#     print(mix)
    #print("\ninfo_bits:")
    #print(info_bits)
    # print("\nQAM:")
    # print(QAM_s)


