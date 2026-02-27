"""
RL-TBoost Figure Generator
===========================
Generates a 4-panel publication-quality figure illustrating:
  Panel A — RL training loop: loss curves + topological shape preservation criterion
  Panel B — Persistence diagram (mock TDA features, H0/H1)
  Panel C — ROC curves comparing RL-TBoost vs baselines
  Panel D — Y1 mortality risk stratification (Kaplan-Meier style)

Replace the synthetic data blocks (marked with # ← REPLACE) with your actual results.

Usage:
    python rl_tboost_figure.py
    # Saves: rl_tboost_figure.png  (300 dpi, publication quality)
              rl_tboost_figure.svg  (vector, for slides / papers)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

# ─────────────────────────────────────────────
# AESTHETIC SETUP
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
})

BG      = "#ffffff"
PANEL   = "#f8f9fa"
BORDER  = "#dee2e6"
TEXT    = "#1a1a2e"
MUTED   = "#6c757d"
TEAL    = "#0d9e7e"
VIOLET  = "#7c3aed"
ROSE    = "#e11d48"
AMBER   = "#d97706"
BLUE    = "#2563eb"
GREEN   = "#16a34a"
RED     = "#dc2626"

# ─────────────────────────────────────────────
# SYNTHETIC DATA  (← REPLACE with actual data)
# ─────────────────────────────────────────────
np.random.seed(42)
epochs = np.arange(1, 101)

# Panel A: loss curves
def smooth(x, w=8):
    return np.convolve(x, np.ones(w)/w, mode='same')

train_loss = smooth(0.72 * np.exp(-epochs/40) + 0.08 + np.random.normal(0, 0.012, 100))
test_loss  = smooth(0.80 * np.exp(-epochs/45) + 0.12 + np.random.normal(0, 0.018, 100))

# Topological stop epochs (where shape is preserved = green dot, violated = red X)
topo_ok      = [18, 25, 31, 38, 45, 52, 60, 67, 74]   # ← REPLACE with actual epochs
topo_stopped = [22, 42, 56, 71, 85]                    # ← REPLACE

# Panel B: persistence diagram points (birth, death) for H0 and H1
np.random.seed(7)
n_h0, n_h1 = 28, 14
h0_birth = np.random.uniform(0.0, 0.6, n_h0)
h0_death = h0_birth + np.random.exponential(0.22, n_h0)
h0_death = np.clip(h0_death, h0_birth + 0.01, 1.4)

h1_birth = np.random.uniform(0.1, 0.7, n_h1)
h1_death = h1_birth + np.random.exponential(0.15, n_h1)
h1_death = np.clip(h1_death, h1_birth + 0.01, 1.4)

# Panel C: ROC curves  ← REPLACE y_true / y_scores with your actual arrays
n_pts = 300
y_true = np.random.binomial(1, 0.28, n_pts)

def make_scores(auc_target, noise=0.10):
    scores = np.where(y_true == 1,
                      np.random.beta(5, 2, n_pts),
                      np.random.beta(2, 5, n_pts))
    scores += np.random.normal(0, noise, n_pts)
    return np.clip(scores, 0, 1)

scores_rltboost  = make_scores(0.86, noise=0.07)   # ← REPLACE
scores_baseline1 = make_scores(0.74, noise=0.13)   # ← REPLACE (e.g., plain DL)
scores_baseline2 = make_scores(0.69, noise=0.16)   # ← REPLACE (e.g., LR / RF)

fpr_rt, tpr_rt, _ = roc_curve(y_true, scores_rltboost)
fpr_b1, tpr_b1, _ = roc_curve(y_true, scores_baseline1)
fpr_b2, tpr_b2, _ = roc_curve(y_true, scores_baseline2)
auc_rt = auc(fpr_rt, tpr_rt)
auc_b1 = auc(fpr_b1, tpr_b1)
auc_b2 = auc(fpr_b2, tpr_b2)

# Panel D: survival curves  ← REPLACE with Kaplan-Meier output from lifelines
t = np.linspace(0, 365, 200)

def km_curve(hazard, noise=0.008):
    s = np.exp(-hazard * t / 365)
    s += np.cumsum(np.random.normal(0, noise, 200))
    return np.clip(s, 0, 1)

surv_low  = km_curve(0.08)   # ← REPLACE: low-risk group
surv_high = km_curve(0.52)   # ← REPLACE: high-risk group
ci_low_u  = np.clip(surv_low  + 0.04, 0, 1)
ci_low_l  = np.clip(surv_low  - 0.04, 0, 1)
ci_high_u = np.clip(surv_high + 0.06, 0, 1)
ci_high_l = np.clip(surv_high - 0.06, 0, 1)

# ─────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor=BG)
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
               left=0.07, right=0.97, top=0.92, bottom=0.07)

panel_style = dict(facecolor=PANEL)

# ── PANEL A ── RL Training: loss curves + topo stops ──────────────────────────
ax_a = fig.add_subplot(gs[0, 0], **panel_style)
ax_a.set_facecolor(PANEL)
for spine in ax_a.spines.values():
    spine.set_color(BORDER)

ax_a.plot(epochs, train_loss, color=TEAL,   lw=2.0, label="Train loss",      zorder=3)
ax_a.plot(epochs, test_loss,  color=BLUE,   lw=2.0, label="Test loss",       zorder=3)
ax_a.fill_between(epochs, test_loss - 0.012, test_loss + 0.012,
                  color=BLUE, alpha=0.12, zorder=2)

# Topological decision markers
for e in topo_ok:
    ax_a.scatter(e, test_loss[e-1], marker='o', s=55, color=GREEN,
                 zorder=5, linewidths=0)
for e in topo_stopped:
    ax_a.scatter(e, test_loss[e-1], marker='X', s=65, color=RED,
                 zorder=5, linewidths=0)

# Reward signal annotation
ax_a.axhline(test_loss[20], color=AMBER, lw=0.8, ls='--', alpha=0.5)
ax_a.text(75, test_loss[20] + 0.006, "shape threshold", color=AMBER,
          fontsize=8, alpha=0.75)

# Legend handles
handles = [
    Line2D([0], [0], color=TEAL, lw=2,   label="Train loss"),
    Line2D([0], [0], color=BLUE, lw=2,   label="Test loss"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN,
           markersize=8, label="Shape preserved ✓"),
    Line2D([0], [0], marker='X', color='w', markerfacecolor=RED,
           markersize=8, label="Shape violated — stop"),
]
ax_a.legend(handles=handles, fontsize=8, framealpha=0.15,
            facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, loc='upper right')

ax_a.set_xlabel("Epoch", color=MUTED, fontsize=10)
ax_a.set_ylabel("Loss", color=MUTED, fontsize=10)
ax_a.set_title("A  —  Topologically-Guided RL Training Loop",
               color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=10)
ax_a.tick_params(colors=MUTED, labelsize=9)
ax_a.grid(True, color=BORDER)

# ── PANEL B ── Persistence Diagram ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1], **panel_style)
ax_b.set_facecolor(PANEL)
for spine in ax_b.spines.values():
    spine.set_color(BORDER)

diag_max = 1.4
ax_b.plot([0, diag_max], [0, diag_max], color=BORDER, lw=1.0, ls='--', zorder=1)

ax_b.scatter(h0_birth, h0_death, s=40, color=VIOLET, alpha=0.85,
             edgecolors='none', label=f"H₀  (components)  n={n_h0}", zorder=3)
ax_b.scatter(h1_birth, h1_death, s=40, color=ROSE,   alpha=0.85,
             edgecolors='none', label=f"H₁  (loops)       n={n_h1}", zorder=3)

# Persistence bars (lifetime visual)
for b, d in zip(h0_birth[:6], h0_death[:6]):
    ax_b.plot([b, b], [b, d], color=VIOLET, lw=0.7, alpha=0.25, zorder=2)
for b, d in zip(h1_birth[:4], h1_death[:4]):
    ax_b.plot([b, b], [b, d], color=ROSE,   lw=0.7, alpha=0.25, zorder=2)

ax_b.set_xlim(0, diag_max)
ax_b.set_ylim(0, diag_max * 1.05)
ax_b.set_xlabel("Birth", color=MUTED, fontsize=10)
ax_b.set_ylabel("Death", color=MUTED, fontsize=10)
ax_b.set_title("B  —  Persistence Diagram (TDA State Features)",
               color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=10)
ax_b.legend(fontsize=8, framealpha=0.15, facecolor=BG,
            edgecolor=BORDER, labelcolor=TEXT)
ax_b.tick_params(colors=MUTED, labelsize=9)
ax_b.grid(True, color=BORDER)
ax_b.text(0.72, 0.08, "birth = death\n(zero persistence)",
          color=MUTED, fontsize=7, transform=ax_b.transAxes, ha='right')

# ── PANEL C ── ROC curves ─────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0], **panel_style)
ax_c.set_facecolor(PANEL)
for spine in ax_c.spines.values():
    spine.set_color(BORDER)

ax_c.plot([0, 1], [0, 1], color=BORDER, lw=1.0, ls='--', zorder=1)
ax_c.plot(fpr_rt, tpr_rt, color=TEAL,   lw=2.5, zorder=4,
          label=f"RL-TBoost    AUC = {auc_rt:.3f}")
ax_c.plot(fpr_b1, tpr_b1, color=BLUE,   lw=1.8, zorder=3, ls='-.',
          label=f"DL baseline  AUC = {auc_b1:.3f}")
ax_c.plot(fpr_b2, tpr_b2, color=AMBER,  lw=1.8, zorder=3, ls=':',
          label=f"Classical    AUC = {auc_b2:.3f}")
ax_c.fill_between(fpr_rt, tpr_rt, alpha=0.08, color=TEAL, zorder=2)

ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1.02)
ax_c.set_xlabel("False Positive Rate", color=MUTED, fontsize=10)
ax_c.set_ylabel("True Positive Rate",  color=MUTED, fontsize=10)
ax_c.set_title("C  —  ROC Curves — Y1 Mortality Prediction",
               color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=10)
ax_c.legend(fontsize=9, framealpha=0.15, facecolor=BG,
            edgecolor=BORDER, labelcolor=TEXT, loc='lower right')
ax_c.tick_params(colors=MUTED, labelsize=9)
ax_c.grid(True, color=BORDER)

# ── PANEL D ── Survival curves ────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1], **panel_style)
ax_d.set_facecolor(PANEL)
for spine in ax_d.spines.values():
    spine.set_color(BORDER)

ax_d.plot(t, surv_low,  color=GREEN, lw=2.5, label="Low-risk group",  zorder=4)
ax_d.plot(t, surv_high, color=RED,   lw=2.5, label="High-risk group", zorder=4)
ax_d.fill_between(t, ci_low_l,  ci_low_u,  color=GREEN, alpha=0.12, zorder=2)
ax_d.fill_between(t, ci_high_l, ci_high_u, color=RED,   alpha=0.12, zorder=2)

# Annotate at day 365
ax_d.axvline(365, color=BORDER, lw=1.0, ls='--', alpha=0.6)
ax_d.text(368, surv_low[-1]  + 0.02, f"{surv_low[-1]:.0%}",
          color=GREEN, fontsize=9, fontweight='bold')
ax_d.text(368, surv_high[-1] + 0.02, f"{surv_high[-1]:.0%}",
          color=RED,   fontsize=9, fontweight='bold')

ax_d.set_xlim(0, 390)
ax_d.set_ylim(0, 1.05)
ax_d.set_xlabel("Days post-transplant", color=MUTED, fontsize=10)
ax_d.set_ylabel("Survival probability",  color=MUTED, fontsize=10)
ax_d.set_title("D  —  Y1 Survival by RL-TBoost Risk Group",
               color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=10)
ax_d.legend(fontsize=9, framealpha=0.15, facecolor=BG,
            edgecolor=BORDER, labelcolor=TEXT)
ax_d.tick_params(colors=MUTED, labelsize=9)
ax_d.grid(True, color=BORDER)
ax_d.text(0.02, 0.06, "95% CI shaded", color=MUTED, fontsize=8,
          transform=ax_d.transAxes)

# ── SUPTITLE ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "RL-TBoost  |  Reinforcement Learning + TDA for Lung Transplant Y1 Mortality",
    color=TEXT, fontsize=13, fontweight='bold', y=0.975
)
fig.text(0.5, 0.005,
         "MorillaLab  ·  Synthetic data for illustration — replace marked blocks with actual results",
         ha='center', fontsize=8, color=MUTED)

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_png = "rl_tboost_figure.png"
out_svg = "rl_tboost_figure.svg"
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor=BG)
fig.savefig(out_svg,           bbox_inches='tight', facecolor=BG)
print(f"Saved: {out_png}  {out_svg}")
plt.show()
