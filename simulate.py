"""Run the baseline replication simulations and produce evidence figures.

Simulations (continuous-time, scipy.signal.lsim):
  1. Unit-step response: open loop, PI (Kp=7, Ki=200), LQR (Q,R as published)
     -> compare with paper Fig. 5 / Table II.
  2. Reference-speed tracking 1000 -> 1500 rpm (step at t=0.5 s) for PI and LQR
     -> speed (rpm), armature voltage (V), armature current (A)
     -> compare with paper Fig. 6.

Output: prints a metrics summary; saves figures to ./figures.
"""
import os
import numpy as np
from scipy import signal

from motor_model import A, B, C, D, RPM2RAD, check_model
from controllers import KP_PUB, KI_PUB, Q_PUB, R_PUB, lqr_gains

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.35,
                     "lines.linewidth": 1.4})


# ---------------- transient metrics ----------------
def step_metrics(t, y, final=None, settle_frac=0.02, rise=(0.1, 0.9)):
    """10-90% rise time, 2% settling time, % overshoot (relative to final)."""
    y = np.asarray(y)
    if final is None:
        final = y[-1]
    if abs(final) < 1e-12:
        return np.nan, np.nan, 0.0, final
    # rise time
    y_lo, y_hi = rise[0] * final, rise[1] * final
    i_lo = np.argmax(np.abs(y) >= np.abs(y_lo)) if np.any(np.abs(y) >= np.abs(y_lo)) else -1
    i_hi = np.argmax(np.abs(y) >= np.abs(y_hi)) if np.any(np.abs(y) >= np.abs(y_hi)) else -1
    tr = (t[i_hi] - t[i_lo]) if (i_lo >= 0 and i_hi >= 0 and i_hi >= i_lo) else np.nan
    # settling time: last exit from 2% band
    band = settle_frac * abs(final)
    out = np.where(np.abs(y - final) > band)[0]
    ts = t[out[-1]] if out.size else 0.0
    # overshoot
    peak = np.max(y)
    mp = max(0.0, 100.0 * (peak - final) / final) if final > 0 else 0.0
    return tr, ts, mp, final


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------- closed-loop systems ----------------
def pi_closed_loop(Kp, Ki):
    """State [omega, i_a, z]; input r; output omega. z = int(r - omega)."""
    n = A.shape[0]
    Acl = np.block([[A - B @ (Kp * C), B * Ki], [-C, np.zeros((1, 1))]])
    Bcl = np.vstack([B * Kp, np.ones((1, 1))])
    Ccl = np.block([C, np.zeros((1, 1))])
    return Acl, Bcl, Ccl, np.zeros((1, 1))


def lqr_closed_loop(K):
    """State [omega, i_a, z]; input r; output omega. u = -K*[x; z]."""
    n = A.shape[0]
    A_aug = np.block([[A, np.zeros((n, 1))], [-C, np.zeros((1, 1))]])
    B_aug = np.vstack([B, np.zeros((1, 1))])
    Acl = A_aug - B_aug @ K.reshape(1, -1)
    Bcl = np.array([[0.0], [0.0], [1.0]])
    Ccl = np.block([C, np.zeros((1, 1))])
    return Acl, Bcl, Ccl, np.zeros((1, 1))


# ---------------- simulation 1: unit step ----------------
def run_unit_step(lqr_g=None, tag=""):
    t = np.linspace(0.0, 0.35, 7001)
    u = np.ones_like(t)
    # open loop
    _, y_ol, _ = signal.lsim((A, B, C, D), u, t)
    # PI
    Acl, Bcl, Ccl, Dcl = pi_closed_loop(KP_PUB, KI_PUB)
    _, y_pi, _ = signal.lsim((Acl, Bcl, Ccl, Dcl), u, t)
    # LQR (use supplied gains, else those from the published Q/R)
    if lqr_g is None:
        K, _, _ = lqr_gains(A, B, C, Q_PUB, R_PUB)
    else:
        K = lqr_g
    Acl, Bcl, Ccl, Dcl = lqr_closed_loop(K)
    _, y_lqr, _ = signal.lsim((Acl, Bcl, Ccl, Dcl), u, t)

    m_ol = step_metrics(t, y_ol)
    m_pi = step_metrics(t, y_pi)
    m_lqr = step_metrics(t, y_lqr)

    print("== Unit-step transient metrics ==")
    for name, m in (("Open loop", m_ol), ("PI", m_pi), ("LQR", m_lqr)):
        print(f"{name:10s} tr={fmt(m[0],3):>8s} s  ts={fmt(m[1],3):>8s} s  "
              f"Mp={fmt(m[2],1):>7s}%  final={fmt(m[3],3)}")
    print("Paper Table II: OL tr=0.0573 ts=0.0946 Mp=0% | PI tr=0.0105 ts=0.111 Mp=36.9% |"
          " LQR tr=0.0965 ts=0.173 Mp=0%")

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(t, y_ol, label="Open loop")
    ax.plot(t, y_pi, label="PI controlled")
    ax.plot(t, y_lqr, label="LQR controlled")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 0.35); ax.set_ylim(0, 1.45)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Unit-step response (replication of paper Fig. 5)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"fig5_step_response{tag}.png"), dpi=200)
    plt.close(fig)
    return {"t": t, "y_ol": y_ol, "y_pi": y_pi, "y_lqr": y_lqr,
            "m_ol": m_ol, "m_pi": m_pi, "m_lqr": m_lqr, "K": K}


# ---------------- simulation 2: 1000 -> 1500 rpm tracking ----------------
def run_tracking(track_t=1.0, step_t=0.5, r1=1000.0, r2=1500.0, q=None, r=None):
    r1_rad, r2_rad = r1 * RPM2RAD, r2 * RPM2RAD
    t = np.linspace(0.0, track_t, 8001)
    ref = np.where(t < step_t, r1_rad, r2_rad)

    results = {}
    # PI
    Acl, Bcl, Ccl, Dcl = pi_closed_loop(KP_PUB, KI_PUB)
    _, y_pi, x_pi = signal.lsim((Acl, Bcl, Ccl, Dcl), ref, t)
    v_pi = KP_PUB * (ref - y_pi) + KI_PUB * x_pi[:, 2]
    i_pi = x_pi[:, 1]
    results["PI"] = (y_pi, v_pi, i_pi)
    # LQR
    K, _, _ = lqr_gains(A, B, C, Q_PUB, R_PUB)
    Acl, Bcl, Ccl, Dcl = lqr_closed_loop(K)
    _, y_lqr, x_lqr = signal.lsim((Acl, Bcl, Ccl, Dcl), ref, t)
    v_lqr = -(K @ x_lqr.T).T
    i_lqr = x_lqr[:, 1]
    results["LQR"] = (y_lqr, v_lqr, i_lqr)

    for name, (y, v, i) in results.items():
        # steady-state values after the step
        seg = t >= step_t + 0.25
        print(f"== {name}: tracking 1000->1500 rpm ==")
        print(f"  steady speed = {np.mean(y[seg]) / RPM2RAD:8.1f} rpm "
              f"(ref {r2:.0f}) | V = {np.mean(v[seg]):6.1f} V | I = {np.mean(i[seg]):4.2f} A")
        fig, axes = plt.subplots(3, 1, figsize=(5.2, 5.6), sharex=True)
        axes[0].plot(t, ref / RPM2RAD, "g--", lw=1.2, label="Reference speed")
        axes[0].plot(t, y / RPM2RAD, label="Actual speed")
        axes[0].set_ylabel("Speed (rpm)"); axes[0].legend(loc="lower right", fontsize=8)
        axes[1].plot(t, v); axes[1].set_ylabel("Armature voltage (V)")
        axes[2].plot(t, i); axes[2].set_ylabel("Armature current (A)")
        axes[2].set_xlabel("Time (s)")
        fig.suptitle(f"{name} controlled system (replication of paper Fig. 6)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"fig6_{name.lower()}_tracking.png"), dpi=200)
        plt.close(fig)
    return t, ref, results


if __name__ == "__main__":
    check_model()
    print()
    K175, A_aug, B_aug = lqr_gains(A, B, C, Q_PUB, R_PUB)
    K1, _, _ = lqr_gains(A, B, C, Q_PUB, 1.0)
    print("== LQR gains ==")
    print("K (R=1.75, as published):", np.round(K175, 4))
    print("K (R=1.0,  diagnostic)  :", np.round(K1, 4))
    print("paper reports |K| = [0.9763, 2.712, 31.6228]")
    print()
    run_unit_step()
    # diagnostic: LQR implemented with the gains implied by R = 1
    # (these match the paper's reported k1/k2/ki exactly)
    K1, _, _ = lqr_gains(A, B, C, Q_PUB, 1.0)
    print()
    print("== Diagnostic: LQR with gains from R = 1.0 ==")
    m = run_unit_step(lqr_g=K1, tag="_R1")["m_lqr"]
    print(f"LQR(R=1) tr={fmt(m[0],3)} s  ts={fmt(m[1],3)} s  Mp={fmt(m[2],1)}%")
    print("Paper Table II (LQR): tr=0.0965 ts=0.173 Mp=0%")
    print()
    run_tracking()
    print()
    print("Figures saved to:", FIGDIR)
