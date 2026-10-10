"""Backward Bellman calibration (METHODOLOGY sections 3-4): real continuation value W_theta and calibration model h_phi.

Remaining-goal length L = 1..LMAX (all lengths <= L are refit at each L, with the shorter-suffix value W_prev frozen):
  paired Bellman targets : Z_R = y_R + W_prev(x_R+; g_{k+1:K}),  Z_S = y_S + W_prev(x_S+; g_{k+1:K})
  calibration model      : h(x, g_{k:K}, Z_S) -> Z_R                      (fit on the paired targets)
  simulation targets     : Z_S = y_S + W_prev(x_S+; g_{k+1:K}),  Z~_R = h(x, g_{k:K}, Z_S)
  continuation value     : W(x; g_{k:K}) <- Z_R (paired) and Z~_R (simulation); paired rows are repeated so that they
                           make up PAIRED_SHARE of W's training rows
Estimators:
  direct amortized       : W(x1; g_{1:K})
  simulation-backed      : mean_r sum_k y_k^S  +  sum_k mean_r [h(X_k, g_{k:K}, Z_S,k) - Z_S,k],
                           Z_S,k = y_k^S + W(X_{k+1}; g_{k+1:K}), over target-sequence simulation rollouts r
In code, a remaining-goal sequence g_{k:K} is a "suffix" and its tail g_{k+1:K} a "tail".
"""
import numpy as np

from .backward import W_eval, build_rows
from .models import LMAX, Regressor, encode, frozen

Z_SCALE = {"safety": 1.0, "path": 0.1, "time": 1.0, "ttc": 1.0, "expo": 1.0}
PAIRED_SHARE = 0.75


def h_features(X, S, z, cost):
    return encode(X, S, extra=(np.asarray(z) * Z_SCALE[cost])[:, None])


def paired_repeats(n_paired, n_sim, share=PAIRED_SHARE):
    """How often each paired row is repeated so that paired rows make up `share` of the training rows."""
    return max(1, int(round(share / (1 - share) * n_sim / max(n_paired, 1))))


def train_ours(paired, paired_val, sim, cost, seed=0, L_max=LMAX, n_suf=6, n_suf_sim=2):
    rng = np.random.default_rng(seed)
    d = 4 + 3 * LMAX
    W, h, W_prev = Regressor(d, seed=seed), Regressor(d + 1, seed=seed + 100), None
    hist = []
    for L in range(1, L_max + 1):
        P = {"tr": dict(F=[], Fh=[], ZR=[]), "va": dict(F=[], Fh=[], ZR=[])}
        Sim = dict(X=[], S=[], ZS=[])
        for ell in range(1, L + 1):
            for split, recs in (("tr", paired), ("va", paired_val)):
                X, S, yR, XR, T, I = build_rows(recs, "R", cost, ell, rng, n_suf)
                yS = np.array([recs[i]["S"][f"y_{cost}"] for i in I]); XS = np.array([recs[i]["S"]["x_plus"] for i in I])
                ZR = yR + W_eval(W_prev, XR, T); ZS = yS + W_eval(W_prev, XS, T)
                P[split]["F"].append(encode(X, S)); P[split]["Fh"].append(h_features(X, S, ZS, cost)); P[split]["ZR"].append(ZR)
            X, S, yS, XS, T, _ = build_rows(sim, "S", cost, ell, rng, n_suf_sim)
            Sim["X"].append(X); Sim["S"] += S; Sim["ZS"].append(yS + W_eval(W_prev, XS, T))
        cat = lambda d_, k: np.vstack(d_[k]) if k.startswith("F") else np.concatenate(d_[k])
        # 1) calibration model on the paired Bellman targets
        hinfo = h.fit(cat(P["tr"], "Fh"), cat(P["tr"], "ZR"), cat(P["va"], "Fh"), cat(P["va"], "ZR"), seed=seed + L)
        # 2) corrected simulation targets
        Xs, ZSs = np.vstack(Sim["X"]), np.concatenate(Sim["ZS"])
        ZRs = h.predict(h_features(Xs, Sim["S"], ZSs, cost))
        Fs = encode(Xs, Sim["S"])
        # 3) continuation value on paired (real) targets + corrected simulation targets
        Fp, Zp = cat(P["tr"], "F"), cat(P["tr"], "ZR")
        rep = paired_repeats(len(Fp), len(Fs))
        F_all = np.vstack([np.repeat(Fp, rep, axis=0), Fs]); Z_all = np.concatenate([np.repeat(Zp, rep), ZRs])
        winfo = W.fit(F_all, Z_all, cat(P["va"], "F"), cat(P["va"], "ZR"), batch=4096, seed=seed + 10 * L)
        hist.append(dict(L=L, h_val_mse=hinfo["val_mse"], W_val_mse=winfo["val_mse"], n_paired_rows=len(Fp), n_sim_rows=len(Fs), rep=rep))
        W_prev = frozen(W)
    return W_prev, frozen(h), hist


def direct(W, x1s, routes):
    return W.predict(encode(np.array(x1s), [list(r) for r in routes]))


def sim_backed(W, h, rollouts, goals, cost):
    """Simulation-backed estimator. rollouts: target-sequence simulation rollouts, list of dict(X=(K+1,4), y={cost: (K,)})."""
    K, R = len(goals), len(rollouts)
    tails = [list(goals[k + 1:]) for k in range(K)] * R
    sufs = [list(goals[k:]) for k in range(K)] * R
    Xn = np.vstack([r["X"][1:] for r in rollouts]); Xc = np.vstack([r["X"][:-1] for r in rollouts])
    y = np.concatenate([r["y"][cost] for r in rollouts])
    ZS = y + W_eval(W, Xn, tails)
    hk = h.predict(h_features(Xc, sufs, ZS, cost))
    corr = (hk - ZS).reshape(R, K).mean(0)
    J_sim = y.reshape(R, K).sum(1).mean()
    return float(J_sim + corr.sum())
