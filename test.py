import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("centralized_dataset.csv")
print(f"\n✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns\n")
print("Columns:", list(df.columns))

# ------------------------------
# Summary statistics
# ------------------------------
print("\n🔍 Summary Statistics:")
print(df.describe())

# ------------------------------
# Identify columns
# ------------------------------
state_cols = [col for col in df.columns if col.startswith(("time", "price", "net_load", "satisfaction", "soc_"))]
action_cols = [col for col in df.columns if col.startswith("action_")]

# ------------------------------
# Histogram: State features
# ------------------------------
print("\n📊 Plotting histograms of state features...")
df[state_cols].hist(bins=50, figsize=(15, 10))
plt.suptitle("Distributions of State Features")
plt.tight_layout()
plt.savefig("state_histograms.png")
plt.close()

# ------------------------------
# Histogram: Action values
# ------------------------------
print("\n📊 Plotting histograms of actions...")
df[action_cols].hist(bins=50, figsize=(15, 10))
plt.suptitle("Distributions of Actions")
plt.tight_layout()
plt.savefig("action_histograms.png")
plt.close()

# ------------------------------
# Action range stats
# ------------------------------
print("\n📈 Action Ranges:")
print(df[action_cols].agg(["min", "max", "mean", "std"]))

# ------------------------------
# Correlation matrix
# ------------------------------
print("\n🔗 Generating correlation heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix of All Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

print("\n✅ Analysis complete. Plots saved as:")
print(" - state_histograms.png")
print(" - action_histograms.png")
print(" - correlation_matrix.png")
