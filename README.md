# DroneAudioSet Code

## Overall Pipeline
Go to `scripts/`

### Installation

```conda create -n droneaudioset python=3.9```
```conda activate droneaudioset```
```pip install -r requirements.txt```

For mpsenet and sslam, separate environments are required, as provided in `requirements_mpsenet.txt` and `requirements_sslam.txt`.
For more details, refer to:
MPSENet Github: https://github.com/yxlu-0102/MP-SENet/tree/main
SSLAM Github: https://github.com/ta012/SSLAM/tree/main

## Sample Audio Files
`ComputeResourcesCheck` folder contains all sample audio files
* `ComputeResourcesCheck/preprocessed-audio`: contains 6 audio file recordings containing source and drone sounds. The chosen setting was:
Volume: 80pc
Room: room1
Drone: drone1
Drone-Speaker Distance: speaker-dist-1m
Mic: mic3_8array-up
Drone-Mic Distance: mic-dist-25cm

* `ComputeResourcesCheck/beamforming`: contains 6 audio files after the `preprocess-audio` files are passed through the beamforming (MVDR) stage.
* `ComputeResourcesCheck/spectral-gating`: contains 6 audio files after the `beamforming` files are passed through the spectral gating (noise-reduce) stage.
* `ComputeResourcesCheck/mpsenet`: contains 6 audio files after the `beamforming` files are passed through the neural noise suppression (MPSENET) stage.
* `ComputeResourcesCheck/classification`: contains 6 audio files after the `mpsenet` files are passed through the classification (SSLAM) stage.

### Run through the cells of overall_pipeline.ipynb
### To run MPSENET code
 * create a sub-folder `mpsenet` in the `scripts` folder
 * copy the folder `models`, and the python files dataset.py, env.py, inference.py, and utils.py from MPSENET repo[https://github.com/yxlu-0102/MP-SENet/tree/main] into  the `scripts/mpsenet` folder
 * also copy over `best_ckpt` from MPSENET repo to `scripts` folder

 ### To run SSLAM code
 * create a sub-folder `SSLAM` in the `scripts` folder
 * copy the folder `SSLAM_inference` and the files `checkpoint_best.pt` and `label_descriptors.csv` from SSLAM repo [] to `scripts/SSLAM` folder

 ## Other files and folders
 * `audioset_labelmapping.csv`: has the mapping of the 527 audioset classes to the three categories HV (human vocals), HNV (human non-vocals), and NH (non-human) sounds
 * `avg-snr-per-setting.csv`: has the average SNR computed per setting
 * `docs`: contains all files for the webpage


