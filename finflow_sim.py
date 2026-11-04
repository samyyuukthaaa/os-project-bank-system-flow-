"""FinFlow: discrete-time simulation of a hybrid multilevel scheduler for a
banking-style mixed workload, plus four baseline schedulers.

1 tick = 1 ms of CPU time on a single core. All parameters are modelling
assumptions, not measurements of a real banking system.
"""
import random
from collections import deque
from dataclasses import dataclass, field

FRAUD, PAYMENT, BALANCE, REPORT = "fraud", "payment", "balance", "report"
CLASS_PRIO = {FRAUD: 0, PAYMENT: 1, BALANCE: 2, REPORT: 3}
# implicit deadlines used only by the pure EDF baseline for non-payment tasks
IMPLICIT_DL = {FRAUD: 20, BALANCE: 60, REPORT: 4000}

H = 3000          # arrival horizon (ticks)
T_END = 4000      # simulation end (allows 1000 ticks to drain)
QUANTUM = 4


@dataclass
class Task:
    tid: int
    cls: str
    arrival: int
    burst: int
    deadline: int
    remaining: int = 0
    start: int = -1
    finish: int = -1
    dropped: bool = False
    last_ran: int = -1
    max_gap: int = 0
    def __post_init__(self):
        self.remaining = self.burst
        self.last_ran = self.arrival


# --------------------------------------------------------------------------
# Workload generation
# --------------------------------------------------------------------------
SCENARIOS = {
    "Normal day":     dict(rush=False, spike=False),
    "Payment rush":   dict(rush=True,  spike=False),
    "Fraud spike":    dict(rush=False, spike=True),
    "Combined stress": dict(rush=True, spike=True),
}
RATES = dict(fraud=0.03, payment=0.07, balance=0.12)
RUSH_WIN, RUSH_RATE = (1000, 1500), 0.20
SPIKE_WIN, SPIKE_RATE = (1000, 1300), 0.30


def generate(scenario, seed):
    rng = random.Random(seed)
    cfg = SCENARIOS[scenario]
    tasks, tid = [], 0
    for t in range(H):
        fr = RATES["fraud"]
        pr = RATES["payment"]
        if cfg["spike"] and SPIKE_WIN[0] <= t < SPIKE_WIN[1]:
            fr = SPIKE_RATE
        if cfg["rush"] and RUSH_WIN[0] <= t < RUSH_WIN[1]:
            pr = RUSH_RATE
        if rng.random() < fr:  # */change can be done here/*
            b = rng.randint(1, 3)
            tasks.append(Task(tid, FRAUD, t, b, t + IMPLICIT_DL[FRAUD])); tid += 1
        if rng.random() < pr:
            b = rng.randint(3, 8)
            d = t + int(round(b * rng.uniform(2.5, 5.0)))
            tasks.append(Task(tid, PAYMENT, t, b, d)); tid += 1
        if rng.random() < RATES["balance"]:
            b = rng.randint(1, 2)
            tasks.append(Task(tid, BALANCE, t, b, t + IMPLICIT_DL[BALANCE])); tid += 1
        if t in (0, 750, 1500, 2250):
            b = rng.randint(80, 150)
            tasks.append(Task(tid, REPORT, t, b, t + IMPLICIT_DL[REPORT])); tid += 1
    return tasks


# --------------------------------------------------------------------------
# Schedulers. Each exposes add(task,t) and pick(t) -> Task|None.
# pick() is called once per tick; the returned task runs for one tick.
# --------------------------------------------------------------------------
class FCFS:
    name = "FCFS"
    def __init__(self): self.q = deque(); self.cur = None
    def add(self, task, t): self.q.append(task)
    def pick(self, t):
        if self.cur and self.cur.remaining > 0:
            return self.cur
        self.cur = self.q.popleft() if self.q else None
        return self.cur


class RoundRobin:
    name = "Round Robin"
    def __init__(self, q=QUANTUM): self.q = deque(); self.cur = None; self.used = 0; self.quantum = q
    def add(self, task, t): self.q.append(task)
    def pick(self, t):
        if self.cur and self.cur.remaining > 0 and self.used < self.quantum:
            self.used += 1; return self.cur
        if self.cur and self.cur.remaining > 0:
            self.q.append(self.cur)
        self.cur = self.q.popleft() if self.q else None
        self.used = 1
        return self.cur


class StaticPriority:
    name = "Priority"
    def __init__(self): self.ready = []
    def add(self, task, t): self.ready.append(task)
    def pick(self, t):
        self.ready = [x for x in self.ready if x.remaining > 0]
        if not self.ready: return None
        return min(self.ready, key=lambda x: (CLASS_PRIO[x.cls], x.arrival, x.tid))


class EDF:
    name = "EDF"
    def __init__(self): self.ready = []
    def add(self, task, t): self.ready.append(task)
    def pick(self, t):
        self.ready = [x for x in self.ready if x.remaining > 0]
        if not self.ready: return None
        return min(self.ready, key=lambda x: (x.deadline, x.arrival, x.tid))


class FinFlow:
    """Hybrid multilevel scheduler.
    L0 fraud (preemptive, FCFS within level)
    L1 payments (EDF + admission control)
    L2 balance updates (Round Robin); reports join L2 when load is low
    L3 reports (background) with aging boost to bound starvation.
    """
    name = "FinFlow"
    def __init__(self, load_threshold=0.5, beta=0.98, aging_limit=300,
                 boost_len=QUANTUM, admission=True, aging=True, load_aware=True):
        self.fraud, self.pay, self.bal, self.rep = [], [], deque(), []
        self.load = 0.0; self.beta = beta; self.lth = load_threshold
        self.A = aging_limit; self.boost_len = boost_len
        self.admission, self.aging, self.load_aware = admission, aging, load_aware
        self.rr_cur = None; self.rr_used = 0
        self.boosted = None; self.boost_used = 0
        self.dropped = []
        self.load_trace = []
    def add(self, task, t):
        {FRAUD: self.fraud, PAYMENT: self.pay, REPORT: self.rep}.get(task.cls, self.bal).append(task)

    def _admit(self, t):
        if not self.admission: return
        keep = []
        for p in self.pay:
            if t + p.remaining > p.deadline and p.remaining > 0:
                p.dropped = True; self.dropped.append(p)
            else:
                keep.append(p)
        self.pay = keep

    def pick(self, t):
        self.fraud = [x for x in self.fraud if x.remaining > 0]
        self.pay = [x for x in self.pay if x.remaining > 0]
        self.rep = [x for x in self.rep if x.remaining > 0]
        self._admit(t)
        fg_busy = 1.0 if (self.fraud or self.pay or any(b.remaining > 0 for b in self.bal)) else 0.0
        self.load = self.beta * self.load + (1 - self.beta) * fg_busy
        self.load_trace.append(self.load)

        # L0: fraud checks always win
        if self.fraud:
            return min(self.fraud, key=lambda x: (x.arrival, x.tid))

        # aging boost: a report that has not run for A ticks gets one short slice
        if self.aging:
            if self.boosted and self.boosted.remaining > 0 and self.boost_used < self.boost_len:
                self.boost_used += 1; return self.boosted
            self.boosted = None
            starving = [r for r in self.rep if t - r.last_ran >= self.A]
            if starving and (self.pay or self.bal):
                self.boosted = min(starving, key=lambda r: r.last_ran)
                self.boost_used = 1
                return self.boosted

        # L1: payments by earliest deadline
        if self.pay:
            return min(self.pay, key=lambda x: (x.deadline, x.arrival, x.tid))

        # L2: round robin over balance updates (+ reports when load is low)
        low_load = (not self.load_aware) or self.load < self.lth
        if low_load:
            for r in self.rep:
                if r not in self.bal: self.bal.append(r)
        elif self.load_aware:
            # load has risen: reports leave the RR ring and return to background
            self.bal = deque(x for x in self.bal if x.cls != REPORT or x is self.rr_cur)
        pick = self._rr()
        if pick: return pick

        # L3: background reports when CPU would otherwise idle
        if self.rep:
            return min(self.rep, key=lambda x: (x.arrival, x.tid))
        return None

    def _rr(self):
        c = self.rr_cur
        if c and c.remaining > 0 and self.rr_used < QUANTUM:
            self.rr_used += 1; return c
        if c and c.remaining > 0:
            if c.cls != REPORT or (not self.load_aware) or (self.load < self.lth):
                self.bal.append(c)
        self.bal = deque(x for x in self.bal if x.remaining > 0)
        self.rr_cur = self.bal.popleft() if self.bal else None
        self.rr_used = 1
        return self.rr_cur


# --------------------------------------------------------------------------
# Simulation engine and metrics
# --------------------------------------------------------------------------
def simulate(sched, tasks, record=False):
    tasks = sorted(tasks, key=lambda x: (x.arrival, x.tid))
    i, n = 0, len(tasks)
    prev, switches, busy = None, 0, 0
    gantt, qlen = [], []
    for t in range(T_END):
        while i < n and tasks[i].arrival == t:
            sched.add(tasks[i], t); i += 1
        task = sched.pick(t)
        if record:
            ready = [x for x in tasks[:i] if x.remaining > 0 and not x.dropped]
            qlen.append({c: sum(1 for x in ready if x.cls == c) for c in CLASS_PRIO})
            gantt.append(task.cls if task else None)
        if task is None:
            continue
        if task is not prev:
            switches += 1
        prev = task
        if task.start < 0: task.start = t
        task.max_gap = max(task.max_gap, t - task.last_ran)
        task.remaining -= 1; task.last_ran = t + 1; busy += 1
        if task.remaining == 0:
            task.finish = t + 1
    return metrics(tasks, switches, busy), (gantt, qlen)


def metrics(tasks, switches, busy):
    by = {c: [x for x in tasks if x.cls == c] for c in CLASS_PRIO}
    pays = by[PAYMENT]
    met = sum(1 for p in pays if p.finish >= 0 and p.finish <= p.deadline)
    rush_pays = [p for p in pays if RUSH_WIN[0] <= p.arrival < RUSH_WIN[1] + 200]
    rush_met = sum(1 for p in rush_pays if 0 <= p.finish <= p.deadline)
    fr = [x.finish - x.arrival for x in by[FRAUD] if x.finish >= 0]
    bal = [x.finish - x.arrival for x in by[BALANCE] if x.finish >= 0]
    reps = by[REPORT]
    rep_done = [x for x in reps if x.finish >= 0]
    rep_ta = [x.finish - x.arrival for x in rep_done]
    # longest interval a report spent ready but not running (starvation gap)
    done = sum(1 for x in tasks if x.finish >= 0)
    return dict(
        pay_total=len(pays), pay_met=met, dhr=100.0 * met / max(1, len(pays)),
        rush_dhr=100.0 * rush_met / max(1, len(rush_pays)),
        pay_dropped=sum(1 for p in pays if p.dropped),
        fraud_avg=sum(fr) / max(1, len(fr)), fraud_max=max(fr) if fr else 0,
        bal_avg=sum(bal) / max(1, len(bal)),
        rep_done=len(rep_done), rep_total=len(reps),
        rep_avg_ta=sum(rep_ta) / max(1, len(rep_ta)) if rep_ta else float("nan"),
        rep_max_ta=max(rep_ta) if rep_ta else float("nan"),
        throughput=1000.0 * done / T_END, util=100.0 * busy / T_END,
        switches=switches,
        rep_gap=max((x.max_gap for x in reps), default=0),
    )


def make(name, **kw):
    return {"FCFS": FCFS, "Round Robin": RoundRobin, "Priority": StaticPriority,
            "EDF": EDF, "FinFlow": FinFlow}[name](**kw)


SCHEDULERS = ["FCFS", "Round Robin", "Priority", "EDF", "FinFlow"]
