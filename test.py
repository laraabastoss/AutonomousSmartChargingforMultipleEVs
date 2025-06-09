import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = np.load("centralized_dataset.npz")
X = data["states"]
y = data["actions"]

# Define column names
num_soc = X.shape[1] - 4  # Assuming 4 other features: time, price, net_load, satisfaction
state_cols = ["time", "price", "net_load", "satisfaction"] + [f"soc_{i}" for i in range(num_soc)]
action_cols = [f"action_{i}" for i in range(y.shape[1])]
all_cols = state_cols + action_cols

# Create DataFrame
df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=all_cols)

# Compute correlation matrix
corr = df.corr()

# Plot using Seaborn
plt.figure(figsize=(26, 26))  # Zoomed out for clarity
ax = sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.5},
    xticklabels=True,
    yticklabels=True
)

# Move x-ticks to the top
ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=90, labelsize=7)
ax.tick_params(axis='y', labelsize=7)

# Title
plt.title("Feature Correlation Matrix", fontsize=18, pad=20)

# Save or show
plt.tight_layout()
plt.show()
