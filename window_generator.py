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
dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[1,2,3,4])

# Preprocessing parameters
low_cut_hz = 4.0
high_cut_hz = 38.0
factor_new = 1e-3
init_block_size = 1000

# Define preprocessors
preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
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
