import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from generate_sa_pairs import DATASET_NAME

# ------------------------------
# Load dataset
# ------------------------------

dataset_name = DATASET_NAME

output_dir = f"./plots/{dataset_name}_plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(f"./datasets/{dataset_name}.csv")


# ------------------------------
# Identify columns
# ------------------------------
state_cols = [col for col in df.columns if col.startswith(("time", "price", "net_load", "satisfaction", "soc_", "connected_flag_"))]
action_cols = [col for col in df.columns if col.startswith("action_")]

# ------------------------------
# Histogram: State features
# ------------------------------
df[state_cols].hist(bins=50, figsize=(15, 10))
plt.suptitle("Distributions of State Features")
plt.tight_layout()
plt.savefig("state_histograms.png")
plt.close()

# ------------------------------
# Histogram: Action values
# ------------------------------
df[action_cols].hist(bins=50, figsize=(15, 10))
plt.suptitle("Distributions of Actions")
plt.tight_layout()
plt.savefig("action_histograms.png")
plt.close()


# ------------------------------
# Correlation matrix
# ------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix of All Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

