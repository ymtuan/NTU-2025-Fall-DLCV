import re
import matplotlib.pyplot as plt

# path to your log file
log_file = "./setting_c/finetune.log"

# regex to extract numbers
pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d.]+), Acc:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+), Acc:\s+([\d.]+)"
)

epochs, train_loss, val_loss = [], [], []

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            t_loss = float(match.group(2))
            t_acc = float(match.group(3))
            v_loss = float(match.group(4))
            v_acc = float(match.group(5))

            epochs.append(epoch)
            train_loss.append(t_loss)
            val_loss.append(v_loss)

# plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.grid(True)
#plt.show()

plt.savefig("loss.png", dpi=300, bbox_inches="tight")
plt.show()