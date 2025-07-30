import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
import os, sys
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint







import numpy as np
import random
import h5py
import argparse

import rfcutils
import tensorflow_datasets as tfds



import glob, h5py


from src import unet_model as unet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


bsz = 64

all_datasets = ['create_simulations_dataset_m_1','create_simulations_dataset_m_50','QPSK_CommSignal2', 'QPSK2_CommSignal2', 'QAM16_CommSignal2', 'OFDMQPSK_CommSignal2',
                'QPSK_CommSignal3', 'QPSK2_CommSignal3', 'QAM16_CommSignal3', 'OFDMQPSK_CommSignal3', 'CommSignal2_CommSignal3',
                'QPSK_EMISignal1', 'QPSK2_EMISignal1', 'QAM16_EMISignal1', 'OFDMQPSK_EMISignal1', 'CommSignal2_EMISignal1',
                'QPSK_CommSignal5G1', 'QPSK2_CommSignal5G1', 'QAM16_CommSignal5G1', 'OFDMQPSK_CommSignal5G1', 'CommSignal2_CommSignal5G1']

def train_script(idx):
    dataset_type = all_datasets[idx]

    ds_train, _ = tfds.load(dataset_type, split="train[:90%]",
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='tfds'
    )
   
    ds_val, _ = tfds.load(dataset_type, split="train[90%:]",
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='tfds'
    )

    def extract_example(mixture, target):
        return mixture, target
    
    wandb.init(
    project="rfchallenge-unet-train-M=50",   
    name=f"{dataset_type}",
    config={
        "model": "UNet"
    }
)


  
    ds_train = ds_train.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(bsz)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(bsz)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
     
    for data, target in ds_train.take(1):  # Take just the first batch
        print("Training batch shape (data, target):", data.shape, target.shape) 
        print(f'data:{data} target:{target}') 
    

    window_len = 3200
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    model_pathname = os.path.join('models/create_simulations_dataset_m_50_unet_M_50', 'checkpoint')
    checkpoint = ModelCheckpoint(filepath=model_pathname, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    #checkpoint = ModelCheckpoint(filepath=model_pathname, monitor='val_loss', verbose=0, save_best_only=False, mode='min', save_weights_only=True)
    with mirrored_strategy.scope():
        nn_model = unet.get_unet_model((window_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003)
        #nn_model = unet.get_unet_model((window_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0006)
        nn_model.load_weights('/home/dsi/galgreen/tmp/rfchallenge/models/create_simulations_dataset_m_50_unet_M_50/checkpoint')
        nn_model.fit(ds_train, epochs=2000, batch_size=bsz, shuffle=True, verbose=1, validation_data=ds_val, callbacks=[
    checkpoint,
    earlystopping,
    WandbMetricsLogger(),
    WandbModelCheckpoint("model")  
])
        #nn_model.fit(ds_train, epochs=500, batch_size=bsz, shuffle=True, verbose=1, validation_data=ds_train, callbacks=[checkpoint, earlystopping])
        
if __name__ == '__main__':
    train_script(int(sys.argv[1]))
    wandb.finish()


