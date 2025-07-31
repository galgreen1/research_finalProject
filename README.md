

# Starter Code Setup: 

(Enter the place you want the project at)

git clone https://github.com/galgreen1/save_research.git rfChallange


cd rfChallange

Now you have all of the files locally

Dependencies are detailed in the yml files

# Helper functions for testing:
python sampletest_testmixture_generator.py --m[num]

for Unet:
python sampletest_tf_unet_inference.py -m[num] -l[length]
for Wavenet:
python sampletest_torch_wavenet_inference.py -m[num]

python sampletest_evaluationscript.py -m[num] -network[unet or Wavenet]

# Helper functions for simulation graph:
python rf_simulations_graphs

# Helper functions for training:
python dataset_utils/generate_dataset.py --m[num]

for Unet:
tfds build /home/dsi/galgreen/tmp/rfchallenge/dataset_utils/tfds_scripts/create_simulations_dataset_M=1.py --data_dir tfds/
or
tfds build /home/dsi/galgreen/tmp/rfchallenge/dataset_utils/tfds_scripts/create_simulations_dataset_M=50.py --data_dir tfds/
for Wavenet:
python dataset_utils/example_preprocess_npy_dataset.py -m[num]

python sampletest_evaluationscript.py -m[num] -network[unet or Wavenet]

python train_unet_model.py [index]
python train_torchwavenet.py -m[num]


Trained model weights for the UNet and WaveNet can be obtained here: 

Email:galgreen03@gmail.com




