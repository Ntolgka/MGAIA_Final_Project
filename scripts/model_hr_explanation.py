import os
import json
import pandas as pd
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    ROOT,
    "data",
    "processed",
    "master_dataset_with_stress.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "modeling",
    "hr_explanation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Rows:", len(df))

# ---------------------------------
# Keep only usable rows
# ---------------------------------

df = df.dropna(
    subset=[
        "heart_rate",
        "survivor_status",
        "game_phase",
        "role"
    ]
)

print("Rows after cleaning:", len(df))

# ---------------------------------
# OLS Regression
# ---------------------------------

formula = """
heart_rate
~
C(
    survivor_status,
    Treatment(reference='healthy')
)
+
C(
    game_phase,
    Treatment(reference='early')
)
+
C(
    role,
    Treatment(reference='survivor')
)
"""

print("\nTraining OLS model...\n")

model = smf.ols(
    formula=formula,
    data=df
).fit()

print(model.summary())

# ---------------------------------
# Save summary
# ---------------------------------

summary_txt = os.path.join(
    OUTPUT_DIR,
    "ols_summary.txt"
)

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

print("Saved:", summary_txt)

# ---------------------------------
# Coefficient table
# ---------------------------------

coef_df = pd.DataFrame({
    "variable": model.params.index,
    "coef": model.params.values,
    "p_value": model.pvalues.values
})

coef_csv = os.path.join(
    OUTPUT_DIR,
    "regression_coefficients.csv"
)

coef_df.to_csv(
    coef_csv,
    index=False
)

print("Saved:", coef_csv)

# ---------------------------------
# Plot coefficients
# ---------------------------------

plot_df = coef_df[
    coef_df["variable"] != "Intercept"
].copy()

plot_df = plot_df.sort_values(
    "coef"
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["variable"],
    plot_df["coef"]
)

plt.xlabel("Coefficient (bpm)")
plt.ylabel("Variable")

plt.tight_layout()

fig_path = os.path.join(
    OUTPUT_DIR,
    "regression_coefficients.png"
)

plt.savefig(fig_path, dpi=300)
plt.close()

print("Saved:", fig_path)

# ---------------------------------
# Mean HR by status
# ---------------------------------

status_summary = (
    df.groupby("survivor_status")["heart_rate"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

status_csv = os.path.join(
    OUTPUT_DIR,
    "hr_by_status.csv"
)

status_summary.to_csv(
    status_csv,
    index=False
)

# ---------------------------------
# Mean HR by phase
# ---------------------------------

phase_summary = (
    df.groupby("game_phase")["heart_rate"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

phase_csv = os.path.join(
    OUTPUT_DIR,
    "hr_by_phase.csv"
)

phase_summary.to_csv(
    phase_csv,
    index=False
)

# ---------------------------------
# Save key findings
# ---------------------------------

result = {
    "n_rows": int(len(df)),
    "r_squared": float(model.rsquared),
    "adj_r_squared": float(model.rsquared_adj)
}

json_path = os.path.join(
    OUTPUT_DIR,
    "model_summary.json"
)

with open(json_path, "w") as f:
    json.dump(result, f, indent=4)

print("\nDone.")