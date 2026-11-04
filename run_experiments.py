import json, statistics as st
from finflow_sim import *
SEEDS = range(1, 11)
res = {}
for sc in SCENARIOS:
    res[sc] = {}
    for s in SCHEDULERS:
        runs = [simulate(make(s), generate(sc, seed))[0] for seed in SEEDS]
        agg = {k: st.mean(r[k] for r in runs) for k in runs[0]}
        agg["dhr_sd"] = st.pstdev(r["dhr"] for r in runs)
        agg["fraud_max_all"] = max(r["fraud_max"] for r in runs)
        res[sc][s] = agg
# ablation on combined stress
abl = {}
variants = {"FinFlow (full)": {}, "No admission control": {"admission": False},
            "No aging": {"aging": False}, "No load gating": {"load_aware": False}}
for sc in ["Normal day", "Combined stress"]:
    abl[sc] = {}
    for v, kw in variants.items():
        runs = [simulate(FinFlow(**kw), generate(sc, seed))[0] for seed in SEEDS]
        abl[sc][v] = {k: st.mean(r[k] for r in runs) for k in runs[0]}
# sensitivity: aging limit
sens = {}
for A in [100, 200, 300, 500, 800, 10**9]:
    runs = [simulate(FinFlow(aging_limit=A), generate("Combined stress", seed))[0] for seed in SEEDS]
    sens[A] = {k: st.mean(r[k] for r in runs) for k in runs[0]}
json.dump(dict(res=res, abl=abl, sens={str(k): v for k, v in sens.items()}), open("results.json", "w"), indent=1)
def show(d):
    for sc, row in d.items():
        print("==", sc)
        for s, m in row.items():
            print(f"{s:22s} DHR {m['dhr']:5.1f} rush {m['rush_dhr']:5.1f} drop {m['pay_dropped']:5.1f} frA {m['fraud_avg']:6.1f} frM {m['fraud_max']:6.1f} bal {m['bal_avg']:6.1f} rep {m['rep_done']:.1f}/{m['rep_total']} repTA {m["rep_avg_ta"]:7.1f} gap {m["rep_gap"]:6.0f} max {m['rep_max_ta']:7.1f} thr {m['throughput']:5.1f} util {m['util']:5.1f} cs {m['switches']:6.0f}")
show(res); show(abl)
for A, m in sens.items(): print(A, round(m['dhr'],1), round(m['rep_avg_ta']), round(m['rep_max_ta']), m['rep_done'])
