# ============================================================
#  Student Marks Prediction using Linear Regression
#  Tools : Python, Pandas, Scikit-learn, Matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# ──────────────────────────────────────────────
# 1.  DATASET
# ──────────────────────────────────────────────
data = {
    "study_hours": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    "exam_marks":  [17,  20,  24,  29,  33,  39,  43,  50,  55,
                    58,  64,  69,  74,  79,  85,  88,  92,  96,  99],
}

df = pd.DataFrame(data)

print("=" * 55)
print("       STUDENT MARKS PREDICTION — LINEAR REGRESSION")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(df.to_string(index=False))

# ──────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────
print("\n── Descriptive Statistics ──")
print(df.describe().round(2))

correlation = df["study_hours"].corr(df["exam_marks"])
print(f"\nPearson Correlation (hours ↔ marks): {correlation:.4f}")

# ──────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X = df[["study_hours"]]          # feature  (2-D for sklearn)
y = df["exam_marks"]             # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain samples : {len(X_train)}   Test samples : {len(X_test)}")

# ──────────────────────────────────────────────
# 4.  MODEL — Ordinary Least Squares
# ──────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

slope     = model.coef_[0]
intercept = model.intercept_

print("\n── Model Parameters ──")
print(f"  Slope     (m) : {slope:.4f}  ← marks gained per study hour")
print(f"  Intercept (b) : {intercept:.4f}  ← baseline predicted marks")
print(f"\n  Equation  : marks = {slope:.4f} × hours + ({intercept:.4f})")

# ──────────────────────────────────────────────
# 5.  EVALUATION
# ──────────────────────────────────────────────
y_pred_test  = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_pred_all   = model.predict(X)

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test,  y_pred_test)
mse_val  = mean_squared_error(y_test, y_pred_test)
rmse_val = np.sqrt(mse_val)
mae_val  = mean_absolute_error(y_test, y_pred_test)

print("\n── Model Evaluation ──")
print(f"  R²  (train)  : {r2_train:.4f}")
print(f"  R²  (test)   : {r2_test:.4f}  ← {r2_test*100:.2f}% variance explained")
print(f"  MSE          : {mse_val:.4f}")
print(f"  RMSE         : {rmse_val:.4f}")
print(f"  MAE          : {mae_val:.4f}")

# Residuals
residuals = y - y_pred_all

# ──────────────────────────────────────────────
# 6.  GRADE HELPER
# ──────────────────────────────────────────────
def get_grade(mark):
    if mark >= 90: return "A+"
    if mark >= 75: return "A"
    if mark >= 60: return "B"
    if mark >= 45: return "C"
    if mark >= 35: return "D"
    return "F"

# Full predictions table
df["predicted_marks"] = y_pred_all.round(1)
df["residual"]        = residuals.round(2)
df["grade"]           = df["exam_marks"].apply(get_grade)

print("\n── Full Predictions Table ──")
print(df.to_string(index=False))

# ──────────────────────────────────────────────
# 7.  CUSTOM PREDICTION
# ──────────────────────────────────────────────
def predict_marks(hours: float) -> dict:
    """Predict exam marks for a given number of study hours."""
    hours = max(0, min(hours, 24))          # clamp to sensible range
    marks = float(model.predict([[hours]])[0])
    marks = max(0, min(100, marks))         # clamp to [0, 100]
    return {"hours": hours, "marks": round(marks, 1), "grade": get_grade(marks)}

print("\n── Custom Predictions ──")
for h in [2.0, 4.5, 6.0, 7.5, 9.0, 11.0]:
    p = predict_marks(h)
    print(f"  {p['hours']:5.1f} hrs  →  {p['marks']:5.1f} marks  [{p['grade']}]")

# ──────────────────────────────────────────────
# 8.  VISUALISATIONS  (4-panel figure)
# ──────────────────────────────────────────────
DARK   = "#0F1117"
CARD   = "#181B24"
ACCENT = "#5B8DEF"
GREEN  = "#34D399"
YELLOW = "#FBBF24"
TEXT   = "#E8EAF0"
MUTED  = "#6B7280"
GRID   = "#1E2230"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
})

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                        left=0.07, right=0.97, top=0.91, bottom=0.08)

reg_x = np.linspace(0, 11, 200)
reg_y = slope * reg_x + intercept

# ── Panel 1: Scatter + Regression Line ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X_train, y_train, color=ACCENT, s=70, zorder=3,
            label="Train data", alpha=0.9, edgecolors=DARK, linewidths=0.8)
ax1.scatter(X_test,  y_test,  color=YELLOW, s=70, zorder=3,
            label="Test data",  alpha=0.9, edgecolors=DARK, linewidths=0.8, marker="D")
ax1.plot(reg_x, reg_y, color=GREEN, linewidth=2,
         linestyle="--", label="Regression line", zorder=2)
ax1.set_title("Data & Regression Line", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Study Hours")
ax1.set_ylabel("Exam Marks")
ax1.grid(True, linestyle="--")
ax1.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
ax1.set_xlim(0, 11)
ax1.set_ylim(0, 108)

# Annotate R²
ax1.text(0.05, 0.93, f"R² = {r2_test:.4f}", transform=ax1.transAxes,
         fontsize=10, color=GREEN, fontweight="bold",
         bbox=dict(facecolor=DARK, edgecolor=GRID, boxstyle="round,pad=0.4"))

# ── Panel 2: Residuals Plot ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colors = [GREEN if r >= 0 else "#F87171" for r in residuals]
ax2.bar(df["study_hours"], residuals, color=colors, alpha=0.8, width=0.35, zorder=3)
ax2.axhline(0, color=ACCENT, linewidth=1.5, linestyle="--")
ax2.set_title("Residuals (Actual − Predicted)", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Study Hours")
ax2.set_ylabel("Residual")
ax2.grid(True, axis="y", linestyle="--")

# ── Panel 3: Actual vs Predicted ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y, y_pred_all, color=ACCENT, s=65, zorder=3,
            edgecolors=DARK, linewidths=0.8, label="Samples")
perfect = np.linspace(0, 100, 100)
ax3.plot(perfect, perfect, color=GREEN, linewidth=1.8,
         linestyle="--", label="Perfect fit (y = x)")
ax3.set_title("Actual vs Predicted Marks", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Actual Marks")
ax3.set_ylabel("Predicted Marks")
ax3.grid(True, linestyle="--")
ax3.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# ── Panel 4: Prediction Curve ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
hours_range = np.linspace(0, 12, 300)
marks_range = np.clip(slope * hours_range + intercept, 0, 100)

ax4.fill_between(hours_range, marks_range, alpha=0.15, color=ACCENT)
ax4.plot(hours_range, marks_range, color=ACCENT, linewidth=2.5, label="Predicted marks")

# Highlight grade bands
bands = [(90, 100, "A+", "#34D39933"), (75, 90, "A", "#34D39922"),
         (60, 75, "B", "#5B8DEF22"), (45, 60, "C", "#FBBF2422"),
         (35, 45, "D", "#F8717122"), (0,  35, "F", "#F8717133")]
for lo, hi, lbl, col in bands:
    ax4.axhspan(lo, hi, alpha=1, color=col, zorder=0)
    ax4.text(11.6, (lo + hi) / 2, lbl, va="center", fontsize=8,
             color=TEXT, fontweight="bold")

# Mark current example (6 hours)
ex_h, ex_m = 6.0, float(np.clip(slope * 6 + intercept, 0, 100))
ax4.scatter([ex_h], [ex_m], color=GREEN, s=100, zorder=5,
            edgecolors=DARK, linewidths=1)
ax4.annotate(f" {ex_h}h → {ex_m:.1f} marks",
             xy=(ex_h, ex_m), fontsize=9, color=GREEN,
             xytext=(ex_h + 0.3, ex_m - 7))

ax4.set_title("Prediction Curve with Grade Bands", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax4.set_xlabel("Study Hours")
ax4.set_ylabel("Predicted Marks")
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 105)
ax4.grid(True, linestyle="--")

# ── Super-title ──────────────────────────────────────────────
fig.suptitle(
    "Student Marks Prediction — Linear Regression",
    fontsize=15, fontweight="bold", color=TEXT, y=0.97,
)

plt.savefig("/mnt/user-data/outputs/student_marks_prediction.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.show()
print("\n✅  Plot saved → student_marks_prediction.png")

# ──────────────────────────────────────────────
# 9.  INTERACTIVE TERMINAL PREDICTOR
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("           INTERACTIVE PREDICTOR")
print("=" * 55)
while True:
    raw = input("\nEnter study hours (0–12) or 'q' to quit: ").strip()
    if raw.lower() == "q":
        print("Bye! Keep studying 📚")
        break
    try:
        h = float(raw)
        p = predict_marks(h)
        bar = "█" * int(p["marks"] / 5) + "░" * (20 - int(p["marks"] / 5))
        print(f"  ┌─────────────────────────────────┐")
        print(f"  │  Hours  : {p['hours']:5.1f} hrs               │")
        print(f"  │  Marks  : {p['marks']:5.1f} / 100             │")
        print(f"  │  Grade  :  {p['grade']:<3}                    │")
        print(f"  │  [{bar}]  │")
        print(f"  └─────────────────────────────────┘")
    except ValueError:
        print("  ⚠  Please enter a valid number.")
