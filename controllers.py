"""PI and LQR controller design for the DC motor (paper Section III).

PI:   u = Kp*(r - omega) + Ki*int(r - omega), Kp = 7, Ki = 200 (paper III-A).
LQR:  augment state with z = int(r - omega); solve the continuous-time ARE
      with Q = diag(0.1, 0.1, 1000), R = 1.75 (paper III-B); u = -K_aug*[x; z].
"""
import numpy as np
from scipy.linalg import solve_continuous_are

# Published values
KP_PUB, KI_PUB = 7.0, 200.0
Q_PUB = np.diag([0.1, 0.1, 1000.0])
R_PUB = 1.75
K_PUB = np.array([0.9763, 2.712, 31.6228])  # |K| as reported (paper: [k1 k2 -ki])


def augmented_system(A, B, C):
    """Augmented system for reference tracking: E = [x; z], z=int(r-y)."""
    n = A.shape[0]
    A_aug = np.block([[A, np.zeros((n, 1))], [-C, np.zeros((1, 1))]])
    B_aug = np.vstack([B, np.zeros((1, B.shape[1]))])
    return A_aug, B_aug


def lqr_gains(A, B, C, Q, R):
    """Return K_aug (row vector) such that u = -K_aug*[x; z]."""
    A_aug, B_aug = augmented_system(A, B, C)
    S = solve_continuous_are(A_aug, B_aug, Q, R)
    R = np.atleast_2d(R) if np.ndim(R) == 0 else R
    K = np.linalg.inv(R) @ B_aug.T @ S
    return K.flatten(), A_aug, B_aug


if __name__ == "__main__":
    from motor_model import A, B, C

    K175, A_aug, B_aug = lqr_gains(A, B, C, Q_PUB, R_PUB)
    K1, _, _ = lqr_gains(A, B, C, Q_PUB, 1.0)  # diagnostic: R = 1
    print("== LQR gains ==")
    print("K (Q=diag(0.1,0.1,1000), R=1.75) =", np.round(K175, 4),
          " | paper reports |K| = [0.9763, 2.712, 31.6228]")
    print("K (same Q, R=1.0, diagnostic)    =", np.round(K1, 4))
    print("rank(augmented ctrb) =",
          int(np.linalg.matrix_rank(np.hstack([B_aug, A_aug @ B_aug,
                                               A_aug @ A_aug @ B_aug]))))
