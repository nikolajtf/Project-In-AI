import os
import numpy as np
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
    create_windows_from_events
)

# Load the dataset
dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[])

# Preprocessing parameters
low_cut_hz = 4.0
high_cut_hz = 125.0
factor_new = 1e-3
init_block_size = 1000
C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                'C6',
                'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                'FCC5h',
                'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                'CPP5h',
                'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                'CCP1h',
                'CCP2h', 'CPP1h', 'CPP2h']
                
# Define preprocessors
preprocessors = [
    Preprocessor("pick_channels", ch_names=C_sensors),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Preprocess the data
preprocess(dataset, preprocessors, n_jobs=1)

# Windowing parameters
trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info["sfreq"]
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows from events
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

windows_dataset.save(
    path=str(os.getcwd()) + "/windows/data",
    overwrite=True,
)
