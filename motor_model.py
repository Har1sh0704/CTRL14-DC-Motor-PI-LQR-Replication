"""DC motor plant reconstructed from Roy et al. (2024), Table I.

Paper: S. Roy, A. Basak, A. Majumder, S. Chowdhuri, "A Comparative Analysis
on Application of PI and LQR Based Control for Speed Tracking of a DC Motor,"
2024 IEEE CIEC, pp. 237-242, doi:10.1109/CIEC59440.2024.10468133.

State x = [omega (rad/s), i_a (A)]^T, input u = armature voltage v (V),
output y = omega. Continuous-time LTI model.
"""
import numpy as np
from scipy import signal

# Motor parameters (paper Table I)
RA = 12.0      # armature resistance, Ohm
LA = 0.13      # armature inductance, H
KE = 1.113     # back-emf constant, V.s/rad
KM = 1.14      # torque constant, N.m/A
JM = 0.004     # rotor moment of inertia, kg.m^2
BM = 0.01      # viscous friction, N.m.s/rad

PARAMS = {"Ra": RA, "La": LA, "Ke": KE, "Km": KM, "Jm": JM, "Bm": BM}

A = np.array([[-BM / JM, KM / JM],
              [-KE / LA, -RA / LA]])
B = np.array([[0.0], [1.0 / LA]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

RPM2RAD = 2.0 * np.pi / 60.0


def open_loop_tf():
    """Return (num, den) of C(sI-A)^-1 B, normalised to leading den coeff 1."""
    num, den = signal.ss2tf(A, B, C, D)
    return num[0], den


def check_model(verbose=True):
    """Verify model against the values printed in the paper."""
    num, den = open_loop_tf()
    # Normalise denominator and strip leading/trailing zero padding
    den_n = den / den[0]
    num_t = np.trim_zeros(num)
    num_t = num_t[np.abs(num_t) > 1e-9]  # drop numerical -0.0 padding
    num_n = (num_t / den[0]) if num_t.size else np.array([0.0])
    eig = np.linalg.eigvals(A)
    ctrb = np.hstack([B, A @ B])
    obsv = np.vstack([C, C @ A])
    info = {
        "A": A,
        "B": B.flatten(),
        "C": C.flatten(),
        "tf_num": num / den[0],
        "tf_den": den_n,
        "eig": eig,
        "rank_ctrb": int(np.linalg.matrix_rank(ctrb)),
        "rank_obsv": int(np.linalg.matrix_rank(obsv)),
    }
    if verbose:
        print("== Model verification ==")
        print("A =", np.round(A, 4).tolist(), " (paper: [[-2.5,285],[-8.562,-92.307]])")
        print("B =", np.round(B.flatten(), 4).tolist(), " (paper: [0, 7.69])")
        print("C =", C.flatten().tolist(), " (paper: [1, 0])")
        terms = []
        for i, c in enumerate(den_n[::-1]):
            terms.append(f"{c:.4f}" if i == 0 else f"{c:.4f} s^{i}")
        print("Open-loop TF: %s / (%s)" % (" + ".join(f"{c:.4f}" for c in num_n),
                                           " + ".join(terms)))
        print("  paper eq.(7): 2192.31 / (s^2 + 94.81s + 2671)")
        print("eig(A) =", np.round(eig, 4), " (paper: -47.4035 +/- 20.5875i)")
        print("rank(ctrb) =", info["rank_ctrb"], " rank(obsv) =", info["rank_obsv"],
              " (paper: full rank -> controllable/observable)")
    return info


if __name__ == "__main__":
    check_model()
