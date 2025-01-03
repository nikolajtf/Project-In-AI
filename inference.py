import os
import numpy as np
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
    create_windows_from_events
)
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report  # For accuracy calculation

windows_dataset = load_concat_dataset(
    path=str(os.getcwd()) + "/windows/data",
    preload=False,
    ids_to_load=[1,3,5,7,9,11,13,15],
    # ids_to_load=[1,3,5,7,9,11,13,15,17,19,21,23,25,27],
    target_name=None,
)

# Check for GPU availability
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

# Model parameters
n_classes = 4
classes = list(range(n_classes))
n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

# Load the model for inference
model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# model = Deep4Net(
#     n_channels,
#     n_classes,
#     input_window_samples=input_window_samples,
#     final_conv_length='auto',
# )


model.load_state_dict(torch.load(str(os.getcwd()) + "/models/ShallowFBCSPNet_persons14_lr0.0001_wd0.01_bs32_acc0.8848.pth"))
model.eval()
if cuda:
    model.to(device)

# Create DataLoader
batch_size = 32
data_loader = DataLoader(windows_dataset, batch_size=batch_size, shuffle=False)

# Inference and evaluation
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in data_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and classification report
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=[str(cls) for cls in classes]))
