import re
import matplotlib.pyplot as plt

log_file = "output.log"          # path to your log file
save_path = "loss_miou.png"  # path to save figure

# Lists to store metrics
train_losses = []
val_losses = []
val_mious = []

# Regex patterns
train_pattern = re.compile(r"Train Loss:\s*([0-9.]+)")
val_pattern = re.compile(r"Val Loss:\s*([0-9.]+)\s*\|\s*Val mIoU:\s*([0-9.]+)")

# Parse log file
with open(log_file, "r") as f:
    for line in f:
        train_match = train_pattern.search(line)
        val_match = val_pattern.search(line)

        if train_match:
            train_losses.append(float(train_match.group(1)))
        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_mious.append(float(val_match.group(2)))

# Epochs
epochs = range(1, len(train_losses)+1)

# Plot with 2 y-axes
fig, ax1 = plt.subplots(figsize=(10,6))

# Left y-axis: Loss
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.plot(epochs, train_losses, label="Train Loss", color="blue", linestyle="-")
ax1.plot(epochs, val_losses, label="Val Loss", color="orange", linestyle="-")
ax1.tick_params(axis='y', labelcolor="black")
ax1.grid(True)

# Right y-axis: mIoU
ax2 = ax1.twinx()
ax2.set_ylabel("Val mIoU")
ax2.plot(epochs, val_mious, label="Val mIoU", color="green", linestyle="-")
ax2.tick_params(axis='y', labelcolor="black")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")

# Title and save
plt.title("Training and Validation Metrics")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
print(f"Graph saved to {save_path}")
