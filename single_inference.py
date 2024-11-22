import matplotlib.pyplot as plt
from pathlib import Path
from braindecode.datautil import load_concat_dataset
import torch
import torch.nn.functional as F  # For softmax
from braindecode.util import set_random_seeds
from braindecode.models import Deep4Net
from torch.utils.data import DataLoader
import os

label_mapping = {"feet": 0, "left_hand": 1, "rest": 2, "right_hand": 3}
inv_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping for numerical to string

# Load dataset
windows_dataset_loaded = load_concat_dataset(
    path=str(os.getcwd()) + "/windows/data",
    preload=False,
    ids_to_load=[1, 3, 5],
    target_name=None,
)
windows_dataset = windows_dataset_loaded

# Device settings
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

# Model settings
n_classes = 4
classes = list(range(n_classes))
n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

# Initialize model and load checkpoint
model = Deep4Net(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# model = ShallowFBCSPNet(
#     n_channels,
#     n_classes,
#     input_window_samples=input_window_samples,
#     final_conv_length="auto",
# )

if cuda:
    model.cuda()

model.load_state_dict(torch.load(str(os.getcwd()) + "/models/Deep4Net_persons8_lr0.0001_wd0.1_bs16_acc0.8406.pth"))
model.eval()  # Set model to evaluation mode

# Create DataLoader
batch_size = 1
data_loader = DataLoader(windows_dataset, batch_size=batch_size, shuffle=False)

# Inference and evaluation
with torch.no_grad():
    for idx, batch in enumerate(data_loader):
        # Extract data
        inputs, labels = batch[0].to(device), batch[1].to(device)
        
        # Model prediction
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        _, predicted = torch.max(probabilities, 1)  # Get predicted class
        predicted_certainty = probabilities[0, predicted.item()].item()  # Certainty of predicted class
        
        # Map numerical labels to string labels
        predicted_label_str = inv_label_mapping[predicted.item()]
        true_label_str = inv_label_mapping[labels.item()]
        
        # Print input, prediction, label, and certainty
        print(f"Sample {idx + 1}:")
        print(f"Prediction: {predicted_label_str} ({predicted_certainty:.2%} certainty), Label: {true_label_str}")
        print(f"Input shape: {inputs.shape}")

        # Plot the EEG window
        input_data = inputs.cpu().numpy().squeeze()  # Convert to numpy and remove batch dimension
        plt.figure(figsize=(10, 4))
        plt.plot(input_data.T)  # Transpose to plot channels over time
        plt.title(f"EEG Window - Prediction: {predicted_label_str} ({predicted_certainty:.2%}), Label: {true_label_str}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.show()

        # Wait for user input before moving to the next sample
        input("Press Enter to continue...")