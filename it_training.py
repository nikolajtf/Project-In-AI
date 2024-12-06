import itertools
import os
import torch
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.datautil import load_concat_dataset
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from braindecode.util import set_random_seeds
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report  # For accuracy calculation
import pandas as pd

# Parameters to test
# persons = [1, 2, 4, 8, 14]
persons = [14]
lrs = [0.001, 0.01, 0.0001]
weight_decays = [0, 0.01, 0.1]
batch_sizes = [16, 32, 64, 128]

# Model architectures to train
# models_to_train = [("Deep4Net", Deep4Net), ("ShallowFBCSPNet", ShallowFBCSPNet)]
models_to_train = [("ShallowFBCSPNet", ShallowFBCSPNet)]

# Training parameters
n_epochs = 800

# Check for CUDA
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

# Set random seed
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
input_window_samples = None  # Will be initialized dynamically

# Function to calculate training IDs (even numbers)
def calculate_training_ids(person_count):
    return [i * 2 for i in range(person_count)]

# Function to calculate inference IDs (odd numbers)
def calculate_inference_ids(person_count):
    return [i * 2 + 1 for i in range(person_count)]

# Iterate through all parameter combinations
for model_name, model_class in models_to_train:
    for person_count, lr, weight_decay, batch_size in itertools.product(persons, lrs, weight_decays, batch_sizes):
        # Calculate training and inference IDs
        training_ids = calculate_training_ids(person_count)
        inference_ids = calculate_inference_ids(person_count)

        # ---------------------- Training ----------------------
        # Load dataset with training IDs
        windows_dataset = load_concat_dataset(
            path=str(os.getcwd()) + "/windows/data",
            preload=True,
            ids_to_load=training_ids,
            target_name=None,
        )

        splitted = windows_dataset.split('run')
        train_set = splitted['0train']  # Session train

        # Get input dimensions dynamically from the dataset
        if input_window_samples is None:
            n_chans = train_set[0][0].shape[0]
            input_window_samples = train_set[0][0].shape[1]

        # Define model
        model = model_class(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length='auto',
        )

        if cuda:
            model = model.cuda()

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor="valid_accuracy",
            patience=8,
            threshold=0.001,
            threshold_mode="rel",
            lower_is_better=False,
        )

        # Initialize classifier
        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=ValidSplit(0.2, stratified=False),
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy",
                early_stopping,
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
            ],
            device=device,
            classes=list(range(n_classes)),
            max_epochs=n_epochs,
        )

        # Train model
        print(f"Training {model_name} with: persons={person_count}, lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}")
        clf.fit(train_set, y=None)

        # Save model temporarily for inference
        temp_model_path = os.path.join(
            os.getcwd(),
            f"models/temp_{model_name}_persons{person_count}_lr{lr}_wd{weight_decay}_bs{batch_size}.pth"
        )
        torch.save(model.state_dict(), temp_model_path)
        print(f"Temporary {model_name} model saved to {temp_model_path}")

        # ---------------------- Save Training History as CSV ----------------------
        # Convert Skorch History object to DataFrame
        results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
        history_df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                                index=clf.history[:, 'epoch'])

        # Add the 'lr' column
        history_df['lr'] = lr

        # Reorder columns to match the desired order
        history_df = history_df[['train_accuracy', 'train_loss', 'valid_accuracy', 'valid_loss', 'lr']]

        # Save training history to a CSV file
        history_filename = os.path.join(
            os.getcwd(),
            f"models/{model_name}_persons{person_count}_lr{lr}_wd{weight_decay}_bs{batch_size}_history.csv"
        )

        history_df.to_csv(history_filename, index_label='epoch')

        print(f"Training history saved to {history_filename}")

        # ---------------------- Inference ----------------------
        # Load dataset with inference IDs
        inference_dataset = load_concat_dataset(
            path=str(os.getcwd()) + "/windows/data",
            preload=True,
            ids_to_load=inference_ids,
            target_name=None,
        )

        # Reload the model for evaluation
        model.load_state_dict(torch.load(temp_model_path))
        model.eval()

        # Create DataLoader for evaluation
        data_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

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

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Inference complete for {model_name}. Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=[str(cls) for cls in range(n_classes)]))

        # Save model with accuracy in the filename
        final_model_path = os.path.join(
            os.getcwd(),
            f"models/{model_name}_persons{person_count}_lr{lr}_wd{weight_decay}_bs{batch_size}_acc{accuracy:.4f}.pth"
        )
        os.rename(temp_model_path, final_model_path)  # Rename with accuracy
        print(f"Final {model_name} model saved to {final_model_path}")

        # Rename the history file to include inference accuracy in the filename
        history_filename_with_acc = os.path.join(
            os.getcwd(),
            f"models/{model_name}_persons{person_count}_lr{lr}_wd{weight_decay}_bs{batch_size}_acc{accuracy:.4f}_history.csv"
        )

        os.rename(history_filename, history_filename_with_acc)
        print(f"Training history file renamed to {history_filename_with_acc}")

print("All models trained, evaluated, and saved.")
