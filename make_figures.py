import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from finflow_sim import *
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.linewidth": 0.6,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {FRAUD: "#c0392b", PAYMENT: "#2e86c1", BALANCE: "#27ae60", REPORT: "#e1a730"}
d = json.load(open("results.json")); res = d["res"]
W = 3.4

def box(ax, x, y, w, h, text, fc="#ffffff", ec="#333", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                 fc=fc, ec=ec, lw=0.7))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333", mutation_scale=7))

# ---- Fig 1 architecture
fig, ax = plt.subplots(figsize=(W, 2.9)); ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1); ax.axis("off")
box(ax, 0.02, 0.84, 0.30, 0.12, "Workload generator\n(scenario, seed)", "#eef2f7")
box(ax, 0.37, 0.84, 0.26, 0.12, "Task classifier", "#eef2f7")
arrow(ax, 0.32, 0.90, 0.37, 0.90)
lv = [("L0 Fraud check\npreemptive, FCFS", FRAUD), ("L1 Payment\nEDF + admission", PAYMENT),
      ("L2 Balance update\nRound Robin (q=4)", BALANCE), ("L3 Daily report\nbackground", REPORT)]
for k, (t, c) in enumerate(lv):
    y = 0.64 - k*0.16
    box(ax, 0.32, y, 0.36, 0.12, t, fc=matplotlib.colors.to_rgba(COL[c], 0.25), ec=COL[c])
    arrow(ax, 0.50, 0.84, 0.50, 0.84) if k == 0 else None
arrow(ax, 0.50, 0.84, 0.50, 0.76)
box(ax, 0.74, 0.40, 0.24, 0.24, "Dispatcher", "#fdf2e9", bold=True)
for k in range(4):
    y = 0.70 - k*0.16
    arrow(ax, 0.68, y, 0.74, 0.58 - k*0.05)
box(ax, 0.74, 0.16, 0.24, 0.13, "CPU (1 core)\n1 tick = 1 ms", "#eef2f7")
arrow(ax, 0.86, 0.40, 0.86, 0.29)
box(ax, 0.02, 0.36, 0.25, 0.20, "Load monitor\nEWMA L(t)", "#f4ecf7")
box(ax, 0.02, 0.10, 0.25, 0.18, "Aging monitor\nwait \u2265 A", "#f4ecf7")
arrow(ax, 0.27, 0.42, 0.32, 0.38); arrow(ax, 0.27, 0.19, 0.32, 0.22)
box(ax, 0.38, 0.0, 0.60, 0.08, "Metrics collector and plotter", "#eef2f7", fs=7)
arrow(ax, 0.86, 0.16, 0.86, 0.08)
fig.savefig("fig/fig1_architecture.png"); plt.close(fig)

# ---- Fig 2 flowchart
fig, ax = plt.subplots(figsize=(W, 4.1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.axis("off")
steps = [("Tick t: enqueue arrivals,\nupdate load L(t)", "#eef2f7"),
         ("Drop payments with\nt + r\u1d62 > d\u1d62 (admission)", "#eef2f7"),
         ("Fraud queue empty?", "#fff"),
         ("Report waited \u2265 A\nand foreground busy?", "#fff"),
         ("Payment queue empty?", "#fff"),
         ("L(t) < \u03b8 ?  add reports\nto RR ring", "#fff"),
         ("RR ring empty?", "#fff"),
         ("Run oldest report\n(or idle)", COL[REPORT])]
right = [None, None, ("Run oldest fraud\ncheck", COL[FRAUD]), ("Boost report for\none quantum", COL[REPORT]),
         ("Run payment with\nearliest deadline", COL[PAYMENT]), None, ("Run next RR task\n(quantum 4)", COL[BALANCE]), None]
hh, gap = 0.085, 0.122
for k, (t, c) in enumerate(steps):
    y = 0.92 - k*gap
    box(ax, 0.06, y, 0.48, hh, t, fc=matplotlib.colors.to_rgba(c, 0.3) if c != "#fff" else "#fff", fs=6.5)
    if k < len(steps)-1:
        arrow(ax, 0.30, y, 0.30, y - (gap - hh))
        if right[k] is None and k >= 2:
            pass
    if right[k]:
        t2, c2 = right[k]
        box(ax, 0.64, y, 0.33, hh, t2, fc=matplotlib.colors.to_rgba(c2, 0.3), ec=c2, fs=6.5)
        arrow(ax, 0.54, y + hh/2, 0.64, y + hh/2)
        lab = "Yes" if k == 3 else "No"
        ax.text(0.59, y + hh/2 + 0.012, lab, ha="center", fontsize=6)
        ax.text(0.32, y - 0.022, "No" if k == 3 else "Yes", fontsize=6)
fig.savefig("fig/fig2_flowchart.png"); plt.close(fig)

# ---- Fig 3 DHR bars
scs = list(SCENARIOS); x = np.arange(len(scs)); bw = 0.16
fig, ax = plt.subplots(figsize=(W, 2.2))
pal = ["#95a5a6", "#7f8c8d", "#5d6d7e", "#85c1e9", "#1f618d"]
for i, s in enumerate(SCHEDULERS):
    v = [res[sc][s]["dhr"] for sc in scs]; e = [res[sc][s]["dhr_sd"] for sc in scs]
    ax.bar(x + (i-2)*bw, v, bw, yerr=e, label=s, color=pal[i], error_kw=dict(lw=0.5, capsize=1.2))
ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in scs]); ax.set_ylabel("Payment deadline hit rate (%)")
ax.set_ylim(0, 105); ax.legend(ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.17), frameon=False)
ax.grid(axis="y", lw=0.3, alpha=0.6); ax.set_axisbelow(True)
fig.savefig("fig/fig3_dhr.png"); plt.close(fig)

# ---- Fig 4 payment backlog + load
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
for s, c in [("FCFS", "#95a5a6"), ("Priority", "#5d6d7e"), ("EDF", "#85c1e9"), ("FinFlow", "#1f618d")]:
    sch = make(s); _, (g, q) = simulate(sch, generate("Combined stress", 1), record=True)
    a1.plot([z[PAYMENT] for z in q], lw=0.8, color=c, label=s)
    if s == "FinFlow": load = sch.load_trace
a1.set_ylabel("Pending payments"); a1.legend(fontsize=6, frameon=False, ncol=2)
a1.axvspan(*RUSH_WIN, color="#2e86c1", alpha=0.08); a1.axvspan(*SPIKE_WIN, color="#c0392b", alpha=0.08)
a2.plot(load, lw=0.8, color="#7d3c98"); a2.axhline(0.5, ls="--", lw=0.6, color="k")
a2.set_ylabel("FinFlow L(t)"); a2.set_xlabel("Time (ticks)"); a2.set_ylim(0, 1.05)
a2.text(3050, 0.53, "\u03b8 = 0.5", fontsize=6)
for a in (a1, a2): a.grid(lw=0.3, alpha=0.5)
fig.savefig("fig/fig4_backlog.png"); plt.close(fig)

# ---- Fig 5 Gantt around rush onset
t0, t1 = 1000, 1100
fig, ax = plt.subplots(figsize=(W, 1.4))
for row, s in enumerate(["FinFlow", "EDF", "FCFS"]):
    _, (g, q) = simulate(make(s), generate("Combined stress", 1), record=True)
    t = t0
    while t < t1:
        c = g[t]; u = t
        while u < t1 and g[u] == c: u += 1
        if c: ax.broken_barh([(t, u - t)], (row - 0.35, 0.7), color=COL[c], lw=0)
        t = u
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["FinFlow", "EDF", "FCFS"]); ax.set_xlim(t0, t1)
ax.set_xlabel("Time (ticks)"); ax.invert_yaxis()
ax.legend(handles=[Patch(color=COL[c], label=c.capitalize()) for c in COL], ncol=4, fontsize=6,
          loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
fig.savefig("fig/fig5_gantt.png"); plt.close(fig)

# ---- Fig 6 fraud response
fig, ax = plt.subplots(figsize=(W, 2.0))
for i, s in enumerate(SCHEDULERS):
    ax.bar(x + (i-2)*bw, [res[sc][s]["fraud_avg"] for sc in scs], bw, label=s, color=pal[i])
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in scs])
ax.set_ylabel("Mean fraud-check\nresponse time (ticks, log)")
ax.legend(ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
ax.grid(axis="y", lw=0.3, alpha=0.6, which="both"); ax.set_axisbelow(True)
fig.savefig("fig/fig6_fraud.png"); plt.close(fig)
print("done")
