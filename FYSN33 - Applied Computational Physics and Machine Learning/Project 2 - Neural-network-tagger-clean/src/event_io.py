#src/event_io.py

import math

#Mostly from the data-exercise-template
class Particle:
    def __init__(self, event_id, pid, pt, eta, phi, e, m, truth=False):
        self.event_id = event_id
        self.pid = pid
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.e = e
        self.m = m
        self.truth = truth  # only meaningful for jets in MC

class Event:
    def __init__(self, eid, particles=None):
        self.event_id = eid
        self.particles = [] if particles is None else particles

    def add(self, p: Particle):
        assert p.event_id == self.event_id
        self.particles.append(p)

    def jets(self):
        return [p for p in self.particles if abs(p.pid) == 90]

    def leptons(self):
        return [p for p in self.particles if abs(p.pid) != 90]

    def leading_jet(self):
        js = self.jets()
        return max(js, key=lambda p: p.pt) if js else None

    def leading_lepton(self):
        ls = self.leptons()
        return max(ls, key=lambda p: p.pt) if ls else None

# -----------------------------
# IO: load CSV (MC or data)
# -----------------------------
def load_events_csv(path):
    """
    Reads jets.csv or pythia.csv and returns a list[Event] sorted by event_id.
    - jets.csv has 7 columns (no truth)
    - pythia.csv has 8 columns (last is truth label for jets)
    """
    events = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue

            toks = line.split(",")
            if len(toks) == 8:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s, truth_s = toks
                truth = bool(int(truth_s))
            elif len(toks) == 7:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s = toks
                truth = False
            else:
                # unexpected line format
                continue

            eid  = int(eid_s)
            pid  = int(pid_s)
            pt   = float(pt_s)
            eta  = float(eta_s)
            phi  = float(phi_s)
            en   = float(e_s)
            mass = float(m_s)

            if eid not in events:
                events[eid] = Event(eid)

            events[eid].add(Particle(eid, pid, pt, eta, phi, en, mass, truth))

    return [events[k] for k in sorted(events)]
