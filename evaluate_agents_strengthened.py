"""
Evaluation analysis script.

Generates summary metrics, reward distributions, t-tests, and regression metrics
for DDPG, PQC, and VQC market-making agents.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load saved policy outputs
ddpg_outputs = np.load("ddpg_policy_policy_outputs.npz")
pqc_outputs = np.load("pqc_policy_policy_outputs.npz")
vqc_outputs = np.load("vqc_policy_policy_outputs.npz")

# Extract states and actions
ddpg_states, ddpg_actions = ddpg_outputs["states"], ddpg_outputs["actions"]
pqc_states, pqc_actions = pqc_outputs["states"], pqc_outputs["actions"]
vqc_states, vqc_actions = vqc_outputs["states"], vqc_outputs["actions"]

# Compute per-agent action standard deviation
ddpg_action_std = np.std(ddpg_actions, axis=0)
pqc_action_std = np.std(pqc_actions, axis=0)
vqc_action_std = np.std(vqc_actions, axis=0)

def extract_rewards(csv_path):
    """
    Extracts reward column from CSV logs.
    """
    df = pd.read_csv(csv_path)
    for col in ["ma_reward", "reward_ma", "moving_avg_reward", "reward", "Total Reward"]:
        if col in df.columns:
            return df[col].dropna().to_numpy()
    raise ValueError(f"Expected reward column not found in {csv_path}. Found columns: {df.columns.tolist()}")

# Load reward logs
ddpg_rewards = extract_rewards("results/ddpg19.csv")
pqc_rewards = extract_rewards("results/pqc41.csv")
vqc_rewards = extract_rewards("results/vqc76.csv")

# Generate summary statistics
summary = {
    "Agent": ["DDPG", "PQC", "VQC"],
    "Mean Reward": [np.mean(ddpg_rewards), np.mean(pqc_rewards), np.mean(vqc_rewards)],
    "Reward Std": [np.std(ddpg_rewards), np.std(pqc_rewards), np.std(vqc_rewards)],
    "Action Std": [np.mean(ddpg_action_std), np.mean(pqc_action_std), np.mean(vqc_action_std)],
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv("policy_summary_metrics.csv", index=False)

# Plot reward distributions
plt.figure()
sns.kdeplot(ddpg_rewards, label="DDPG")
sns.kdeplot(pqc_rewards, label="PQC")
sns.kdeplot(vqc_rewards, label="VQC")
plt.title("Reward Distributions")
plt.xlabel("Moving Average Reward")
plt.legend()
plt.savefig("reward_distributions.png")
plt.close()

# T-tests for reward variance
print("Reward Variance T-Tests")
print("DDPG vs PQC:", ttest_ind(ddpg_rewards, pqc_rewards, equal_var=False))
print("DDPG vs VQC:", ttest_ind(ddpg_rewards, vqc_rewards, equal_var=False))
print("PQC vs VQC:", ttest_ind(pqc_rewards, vqc_rewards, equal_var=False))

# T-tests for action variance
print("\nAction Variance T-Tests")
print("DDPG vs PQC:", ttest_ind(ddpg_action_std, pqc_action_std, equal_var=False))
print("DDPG vs VQC:", ttest_ind(ddpg_action_std, vqc_action_std, equal_var=False))
print("PQC vs VQC:", ttest_ind(pqc_action_std, vqc_action_std, equal_var=False))

# Action regression metrics: R2 and MSE
print("\nAction Regression Metrics")
print("R2 PQC:", r2_score(ddpg_actions.flatten(), pqc_actions.flatten()))
print("R2 VQC:", r2_score(ddpg_actions.flatten(), vqc_actions.flatten()))
print("MSE PQC:", mean_squared_error(ddpg_actions.flatten(), pqc_actions.flatten()))
print("MSE VQC:", mean_squared_error(ddpg_actions.flatten(), vqc_actions.flatten()))
