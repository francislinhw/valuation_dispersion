"""
Title: Simulating Information Spillover, AIA and EIC
Author: Francis & ChatGPT
Description:
    This script simulates a mini financial market of 20 stocks, where:
    - F01 releases a financial statement.
    - Peers in the same group (F02–F10) experience information spillover.
    - AIA (Actual Information Attention) reflects observed attention.
    - EIC (Expected Information Consumption) is estimated via a Logit model.
    - The script produces 3 plots to visualize:
        1. Group returns vs. others
        2. AIA on event day
        3. EIC probabilities (predicted attention)
"""

# ===================================================
# Import libraries
# ===================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ===================================================
# 1. Simulation setup
# ===================================================
np.random.seed(42)

tickers = [f"F{i:02d}" for i in range(1, 21)]
groupA = tickers[:10]  # Group A = related firms
groupB = tickers[10:]  # Group B = unrelated
dates = pd.date_range("2020-01-01", "2020-06-30", freq="B")

# Build full panel
panel = pd.DataFrame(
    [(d, t) for d in dates for t in tickers], columns=["date", "permno"]
)
panel["groupA"] = panel["permno"].isin(groupA).astype(int)

# ===================================================
# 2. Define events
# ===================================================
focal_day = dates[100]
focal_firm = "F01"

panel["EDAY"] = 0
panel.loc[(panel["date"] == focal_day) & (panel["permno"] == focal_firm), "EDAY"] = 1

# Mark peers on event day
panel["peer_today"] = 0
panel.loc[(panel["date"] == focal_day) & (panel["groupA"] == 1), "peer_today"] = 1

# ===================================================
# 3. Simulate returns (announcement & spillover)
# ===================================================
panel["ret"] = np.random.normal(0.0003, 0.01, len(panel))

# F01 has a big positive return on announcement
panel.loc[(panel["date"] == focal_day) & (panel["permno"] == focal_firm), "ret"] += 0.02

# Peers gain moderately
mask_peers = (
    (panel["date"] == focal_day)
    & (panel["groupA"] == 1)
    & (panel["permno"] != focal_firm)
)
panel.loc[mask_peers, "ret"] += 0.008

# ===================================================
# 4. Generate Actual Information Attention (AIA)
# ===================================================
# Probability of attention depends on event & group proximity
base = 0.05
p_aia = (
    base + 0.6 * panel["EDAY"] + 0.4 * panel["peer_today"] + 0.1 * panel["groupA"]
).clip(0, 0.95)
panel["AIA"] = (np.random.rand(len(panel)) < p_aia).astype(int)

# ===================================================
# 5. Train Expected Information Consumption (EIC)
# ===================================================
train = panel[panel["date"] < focal_day].copy()
test = panel[panel["date"] == focal_day].copy()

X_train = sm.add_constant(train[["EDAY", "peer_today", "groupA"]])
y_train = train["AIA"]

logit_model = sm.Logit(y_train, X_train).fit_regularized(
    alpha=1.0, L1_wt=0.0, disp=False
)

test["EIC_prob"] = logit_model.predict(
    sm.add_constant(test[["EDAY", "peer_today", "groupA"]])
)
threshold = test["EIC_prob"].quantile(0.75)
test["EIC"] = (test["EIC_prob"] > threshold).astype(int)

# ===================================================
# 6. Visualization
# ===================================================

# (1) Cumulative return: Group A vs. Group B
cum = (
    panel.pivot_table(index="date", columns="permno", values="ret")
    .fillna(0)
    .add(1)
    .cumprod()
)
cumA = cum[groupA].mean(axis=1)
cumB = cum[groupB].mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(cumA, label="Group A (linked firms)")
plt.plot(cumB, label="Group B (others)")
plt.axvline(focal_day, color="gray", linestyle="--", label="Event day")
plt.title("Cumulative Returns: Peer Group vs. Others")
plt.xlabel("Date")
plt.ylabel("Cumulative Gross Return")
plt.legend()
plt.tight_layout()
plt.show()

# (2) AIA on event day
plt.figure(figsize=(9, 5))
x = np.arange(len(test))
plt.bar(
    x, test["AIA"], color=["red" if t in groupA else "blue" for t in test["permno"]]
)
plt.xticks(x, test["permno"], rotation=60)
plt.title(f"AIA on Event Day ({focal_day.date()})")
plt.xlabel("Stocks")
plt.ylabel("AIA (0/1)")
plt.tight_layout()
plt.show()

# (3) EIC probabilities (predicted attention)
plt.figure(figsize=(9, 5))
plt.bar(
    x,
    test["EIC_prob"],
    color=plt.cm.coolwarm(test["EIC_prob"] / test["EIC_prob"].max()),
)
plt.xticks(x, test["permno"], rotation=60)
plt.title(f"EIC Probabilities Predicted by Logit (threshold={threshold:.2f})")
plt.xlabel("Stocks")
plt.ylabel("EIC Probability")
plt.tight_layout()
plt.show()

# ===================================================
# END
# ===================================================
print("✅ Simulation complete. Three charts generated successfully.")
