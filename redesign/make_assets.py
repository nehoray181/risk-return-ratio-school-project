import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY   = "#15224A"
NAVY_D = "#0E1733"
TEAL   = "#0E8F8B"
TEAL_D = "#0B6E6B"
AMBER  = "#E3A91C"
AMBER_D= "#A87708"
GRAY   = "#5A6273"
ICE    = "#C9D6F0"

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "Georgia",
    "savefig.transparent": True,
})

def formula(tex, path, color=NAVY, fs=30):
    fig = plt.figure(figsize=(6, 1.6))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=fs, color=color)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

formula(r"$\dfrac{R_p - R_f}{\sigma_p}$", "assets/f_sharpe.png")
formula(r"$\dfrac{R_p - R_f}{\sigma_{\mathrm{down}}}$", "assets/f_sortino.png")
formula(r"$\dfrac{R_p - R_f}{\beta_p}$", "assets/f_treynor.png")
formula(r"$\dfrac{R_p - R_b}{\mathrm{TE}}$", "assets/f_info.png")
formula(r"$\dfrac{R_{\mathrm{ann}}}{|\mathrm{MaxDD}|}$", "assets/f_calmar.png")
formula(r"$\mathrm{Score} = \mathrm{diff} \times r \times b$", "assets/f_score.png", color=TEAL_D, fs=34)
formula(r"$\mathrm{diff} \times r \times b$", "assets/f_score_white.png", color="#FFFFFF", fs=34)
formula(r"$y = a \cdot e^{\,b\,t}$", "assets/f_model.png", color=NAVY, fs=32)
formula(r"$\mathrm{diff} = \mathrm{ExpReg}(t_{\mathrm{last}}) - P(t_{\mathrm{last}})$", "assets/f_diff.png", fs=26)

# ---- synthetic price-vs-trend series (shared by hero + concept) ----
rng = np.random.default_rng(7)
t = np.arange(260)
trend = 100 * np.exp(0.0028 * t)
noise = np.cumsum(rng.normal(0, 0.9, t.size))
wave = 6 * np.sin(t / 26) - 0.04 * t
price = trend + noise + wave - 4

# ---- hero chart for dark title slide ----
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(t, trend, color=ICE, lw=2.2, ls=(0, (6, 3)), alpha=0.9)
ax.plot(t, price, color="#4FD1CC", lw=2.6)
i = 208
ax.annotate("", xy=(i, trend[i]), xytext=(i, price[i]),
            arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.0))
ax.text(i + 8, (trend[i] + price[i]) / 2, "diff", color=AMBER,
        fontsize=15, style="italic", va="center", family="Georgia")
ax.text(t[-1] - 4, trend[-1] + 3, r"$y = a\,e^{bt}$", color=ICE, fontsize=14,
        ha="right", va="bottom")
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
fig.savefig("assets/hero.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---- concept chart for slide 4 (white bg) ----
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(t, trend, color=NAVY, lw=2.0, ls=(0, (6, 3)), label="Exponential trend")
ax.plot(t, price, color=TEAL, lw=2.4, label="Price")
i = 150
ax.annotate("", xy=(i, trend[i]), xytext=(i, price[i]),
            arrowprops=dict(arrowstyle="<->", color=AMBER_D, lw=2.0))
ax.text(i - 7, (trend[i] + price[i]) / 2, "diff", color=AMBER_D,
        fontsize=14, style="italic", va="center", ha="right", family="Georgia")
j = 244
ax.annotate("", xy=(j + 13, trend[j] * 0.985), xytext=(j - 4, price[j] + 2),
            arrowprops=dict(arrowstyle="-|>", color=TEAL_D, lw=2.6,
                            connectionstyle="arc3,rad=-0.35"))
ax.text(j - 18, price[j] - 10, "reversion\nto trend", color=TEAL_D, fontsize=11,
        ha="right", va="top", family="Georgia")
ax.set_xlim(-4, 292)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper left", frameon=False, fontsize=11,
          prop={"family": "Georgia", "size": 11})
fig.savefig("assets/concept.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---- regime bar chart (values read off the original deck's chart) ----
regimes = [
    ("COVID crash",       48), ("2022 bear/chop",   23),
    ("early-2025",         7), ("2024 bull",         2.5),
    ("2021 momentum top",  2), ("recent (2026)",    -9),
    ("2023 AI recovery", -12), ("post-COVID rally", -15),
    ("2022 bear decline", -42),
]
labels = [r[0] for r in regimes][::-1]
vals = [r[1] for r in regimes][::-1]
colors = [TEAL if v > 0 else AMBER for v in vals]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
bars = ax.barh(labels, vals, color=colors, height=0.62, zorder=3)
for b, v in zip(bars, vals):
    off = 1.2 if v > 0 else -1.2
    ax.text(v + off, b.get_y() + b.get_height() / 2, f"{v:+.0f}".replace("+2.5", "+3"),
            va="center", ha="left" if v > 0 else "right",
            fontsize=11.5, color=NAVY, family="Georgia")
ax.axvline(0, color=NAVY, lw=1.2, zorder=4)
ax.set_xlim(-52, 58)
ax.set_xlabel("Directional-accuracy advantage of the new metric (points)",
              fontsize=12, color=GRAY, family="Georgia")
ax.tick_params(axis="y", labelsize=12.5, colors=NAVY, length=0)
ax.tick_params(axis="x", labelsize=10.5, colors=GRAY, length=0)
for tl in ax.get_yticklabels(): tl.set_family("Georgia")
for s in ax.spines.values(): s.set_visible(False)
ax.grid(axis="x", color="#D8DDE8", lw=0.8, zorder=0)
fig.savefig("assets/regimes.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# ---- case-study sparklines ----
def spark(path, ys, color):
    fig, ax = plt.subplots(figsize=(2.2, 0.95))
    x = np.linspace(0, 1, len(ys))
    ax.plot(x, ys, color=color, lw=3.0, solid_capstyle="round")
    ax.scatter([x[-1]], [ys[-1]], color=color, s=28, zorder=5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

x = np.linspace(0, 1, 60)
spark("assets/spark_a.png", np.where(x < 0.45, 1 - 1.8 * x, 0.19 + 2.0 * (x - 0.45)), TEAL)   # V rebound
spark("assets/spark_b.png", 0.5 + 0.22 * np.sin(x * 19) - 0.12 * x, TEAL)                      # choppy
spark("assets/spark_c.png", 0.15 + 0.85 * x + 0.05 * np.sin(x * 12), AMBER)                    # rally
spark("assets/spark_d.png", 0.95 - 0.85 * x + 0.05 * np.sin(x * 12), AMBER)                    # decline
print("assets done")
