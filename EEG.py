import os
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing.preprocess import Preprocessor, preprocess, exponential_moving_standardize
from braindecode.preprocessing.windowers import create_windows_from_events
from scipy.stats import zscore
import mne
import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from skorch.helper import predefined_split
from braindecode import EEGClassifier
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from braindecode.datautil import load_concat_dataset
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

windows_dataset = load_concat_dataset(
    path=str(os.getcwd()) + "/windows/data",
    preload=True,
    ids_to_load=[0,2,4],
    target_name=None,
)

splitted = windows_dataset.split('run')

train_set = splitted['0train']  # Session train
# valid_set = splitted['1test']  # Session evaluation

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)
n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model = model.cuda()

# -------------------------------- Traning --------------------------------
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 100

# Use early stopping to monitor validation accuracy
early_stopping = EarlyStopping(
    monitor="valid_accuracy",
    patience=5,  # Stop if no improvement after 5 epochs
    threshold=0.001,  # Minimum change to be considered as improvement
    threshold_mode="rel",  # Relative improvement
    lower_is_better=False,  # Higher accuracy is better
)

# Update the EEGClassifier with the early stopping callback and validation split
clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=ValidSplit(0.2, stratified=False),  # Reserve 20% of training data for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        early_stopping,
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
    max_epochs=n_epochs,
)

# Model training with early stopping
clf.fit(train_set, y=None)

torch.save(model.state_dict(), str(os.getcwd()) + "/models/model.pth")

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                  index=clf.history[:, 'epoch'])

# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
               valid_misclass=100 - 100 * df.valid_accuracy)

fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# df.loc[:, ['train_misclass', 'valid_misclass']].plot(
#     ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
# ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
# ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
# ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
# ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()