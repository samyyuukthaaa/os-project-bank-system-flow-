"""Live demo for FinFlow. Run:  python3 demo.py   (takes ~10 s)
Shows (1) the scheduler comparison, (2) a tick-level view of the rush onset,
(3) what each FinFlow mechanism contributes (ablation)."""
import statistics as st
from finflow_sim import *

SEEDS = range(1, 11)
SYM = {FRAUD: "F", PAYMENT: "P", BALANCE: "B", REPORT: "R", None: "."}

def avg(sc, mk, key):
    return st.mean(simulate(mk(), generate(sc, s))[0][key] for s in SEEDS)

def worst_fraud(sc, mk):
    return max(simulate(mk(), generate(sc, s))[0]["fraud_max"] for s in SEEDS)

print("\n=== 1. Scheduler comparison (mean of 10 seeds) ===")
for sc in ["Payment rush", "Combined stress"]:
    print(f"\n{sc}")
    print(f"{'Scheduler':12s} {'Payment DHR %':>14s} {'Fraud mean ms':>14s} {'Fraud worst ms':>15s}")
    for s in SCHEDULERS:
        mk = lambda s=s: make(s)
        print(f"{s:12s} {avg(sc, mk, 'dhr'):14.1f} {avg(sc, mk, 'fraud_avg'):14.1f} {worst_fraud(sc, mk):15d}")

print("\n=== 2. Who runs the CPU at the rush onset (ticks 1000-1079, Combined stress, seed 1) ===")
print("F=fraud P=payment B=balance R=report .=idle")
for s in ["FCFS", "EDF", "FinFlow"]:
    _, (g, _) = simulate(make(s), generate("Combined stress", 1), record=True)
    print(f"{s:8s} " + "".join(SYM[c] for c in g[1000:1080]))

print("\n=== 3. Ablation on Combined stress (what each mechanism buys) ===")
variants = {"Full FinFlow": {}, "No admission control": {"admission": False},
            "No aging": {"aging": False}, "No load gating": {"load_aware": False}}
print(f"{'Variant':22s} {'DHR %':>7s} {'Balance ms':>11s} {'Report max-wait':>16s}")
for v, kw in variants.items():
    mk = lambda kw=kw: FinFlow(**kw)
    print(f"{v:22s} {avg('Combined stress', mk, 'dhr'):7.1f} {avg('Combined stress', mk, 'bal_avg'):11.1f} {avg('Combined stress', mk, 'rep_gap'):16.0f}")
