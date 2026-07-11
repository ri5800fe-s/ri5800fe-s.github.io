#src/feature.py

import math
import numpy as np

def dphi(phi1, phi2):
    """
    Delta-phi wrapped into [0, pi].
    """
    d = phi1 - phi2
    return abs((d + math.pi) % (2 * math.pi) - math.pi)

def dR(eta1, phi1, eta2, phi2): #from the slides
    """
    Delta-R = sqrt((deta)^2 + (dphi)^2)
    """
    return math.sqrt((eta1 - eta2) ** 2 + dphi(phi1, phi2) ** 2)

# -----------------------------
# Feature engineering
# -----------------------------
def build_features(e):
    j = e.leading_jet()
    l = e.leading_lepton()
    if (j is None) or (l is None):
        return None

    djphi = dphi(j.phi, l.phi)
    djR   = dR(j.eta, j.phi, l.eta, l.phi)

    # safe ratios
    pt_ratio = j.pt / l.pt if l.pt > 0 else 0.0
    pt_asym  = (j.pt - l.pt) / (j.pt + l.pt) if (j.pt + l.pt) > 0 else 0.0

    x = np.array([
        j.pt, j.eta, j.phi,
        l.pt, l.eta, l.phi,
        djphi, djR,
        pt_ratio, pt_asym
    ], dtype=np.float32)

    return x