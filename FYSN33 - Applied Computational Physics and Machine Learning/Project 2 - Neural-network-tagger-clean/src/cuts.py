#src/cuts.py

from features import dphi

def pass_cuts(e, min_pt_j, min_pt_l, min_dphi, eta_j_max):
    j = e.leading_jet()
    l = e.leading_lepton()
    if (j is None) or (l is None):
        return False
    return (
        (j.pt >= min_pt_j) and
        (l.pt >= min_pt_l) and
        (dphi(j.phi, l.phi) >= min_dphi) and
        (abs(j.eta) <= eta_j_max)
    )
