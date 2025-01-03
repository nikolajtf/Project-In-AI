import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
name = "ShallowFBCSPNet_persons14_lr0.0001_wd0.01_bs32_acc0.8848_history.csv"
data = pd.read_csv("models2/"+name)

md = str(name.split("_persons")[0])
pc = int(name.split("_lr")[0].split("_persons")[1])
lr = float(name.split("_wd")[0].split("_lr")[1])
wd = float(name.split("_bs")[0].split("_wd")[1])
bs = int(name.split("_acc")[0].split("_bs")[1])
# print(md,pc,lr,wd,bs)

# Extract the necessary columns
epochs = data['epoch']
train_loss = data['train_loss']
valid_loss = data['valid_loss']

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, valid_loss, label="Validation Loss", marker='s')

# Add titles and labels
plt.title(f"Training and Validation Loss. {md}: Person count: {pc}, lr: {lr}, wd: {wd}, bs: {bs}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()