# generate tf data set for: Y=s+c+n
# s is sensing (FMCW) , c is communication (iid bits -> 4 QAM) , n is gauss noise N(0,std)


# to call the script:  tfds build /home/dsi/galgreen/tmp/rfchallenge/dataset_utils/tfds_scripts/create_simulations_dataset_M=1.py --data_dir tfds/

"""Dataset."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds

import glob
import h5py
import numpy as np



soi_type = 'OFDMQPSK'
interference_sig_type = 'SenseAndGauss'

_DESCRIPTION = """
Create mixture of Communication signal, Sensing signal and Gauss noise
"""
_CITATION = """"
https://ieeexplore.ieee.org/document/10411013
"""


class create_simulations_dataset_M_1(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.2.0')
    RELEASE_NOTES = {
      '0.2.0': '2025 release',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'mixture': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
                'signal': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
            }),
            supervised_keys=('mixture', 'signal'),
            homepage='https://rfchallenge.mit.edu/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = os.path.join('/', 'home', 'dsi', 'galgreen', 'tmp', 'rfchallenge', 'dataset', f'Dataset_{soi_type}_train_Mixture_M=1')

        return {
            'train': self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in glob.glob(os.path.join(path, '*.h5')):

            with h5py.File(f,'r') as h5file:
                #print("do tfds")
                mixture = np.array(h5file.get('mixture'))
                #print(f'mixture:{mixture}')
                target = np.array(h5file.get('target'))
                #print(f'target:{target}')
                sig_type = h5file.get('sig_type')[()]
                if isinstance(sig_type, bytes):
                    sig_type = sig_type.decode("utf-8")
            for i in range(mixture.shape[0]):
                #print(f'i={i}')
                yield f'data_{f}_{i}', {

                    'mixture': mixture[i],
                    'signal': target[i],
                }

               
