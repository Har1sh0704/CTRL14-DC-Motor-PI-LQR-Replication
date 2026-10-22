# CTRL_14 — DC Motor PI vs LQR Speed Control: Baseline Replication

Team CTRL_14 · EEEN3015J Control Theory — Published Paper Replication Project
Zhang Yuxiang · Wang Haoran · Zhang Boyu · Xiong Ziyu · Zhang Chengsheng

Paper: S. Roy, A. Basak, A. Majumder and S. Chowdhuri, "A Comparative Analysis on
Application of PI and LQR Based Control for Speed Tracking of a DC Motor,"
2024 IEEE CIEC, pp. 237–242, doi:10.1109/CIEC59440.2024.10468133.

This repository contains the working baseline replication of the paper's central
comparison: reference-speed tracking of a separately excited DC motor under (i) a
PI controller (Kp = 7, Ki = 200) and (ii) an LQR controller with integral action.

## Files

| File | Purpose |
|---|---|
| `motor_model.py` | Table I parameters; state-space model A, B, C, D; model verification |
| `controllers.py` | PI gains; LQR via continuous-time algebraic Riccati equation |
| `simulate.py` | Unit-step + 1000→1500 rpm tracking simulations; metrics; figures |
| `figures/` | Step-response and tracking plots (Fig. 5 / Fig. 6 style) |

## How to run

Requires Python 3.10+ with numpy, scipy, matplotlib:

```
python simulate.py
```

Runs in seconds on a desktop; prints the model check, LQR gains, transient
metrics and tracking steady states; writes figures to `figures/`.

Baseline commit (Week 7 checkpoint): see `git log`; archive distributed as
`CTRL14_replication.zip` (git archive of this repository).

Sample run output (2026-10-22, exit code 0):

```
== Model verification ==
A = [[-2.5, 285.0], [-8.5615, -92.3077]]  (paper: [[-2.5,285],[-8.562,-92.307]])
Open-loop TF: 2192.3077 / (2670.8077 + 94.8077 s^1 + 1.0000 s^2)
== Unit-step transient metrics ==
Open loop  tr=   0.057 s  ts=   0.095 s  Mp=    0.1%  final=0.821
PI         tr=   0.010 s  ts=   0.111 s  Mp=   36.9%  final=1.000
LQR        tr=   0.096 s  ts=   0.173 s  Mp=    0.0%  final=1.000
```

## Results summary (baseline)

Model verification:
- A = [[-2.5, 285], [-8.5615, -92.3077]], B = [0, 7.6923]ᵀ, C = [1, 0] — match paper.
- Open-loop TF 2192.31 / (s² + 94.81 s + 2671) — match paper eq. (7).
- Eigenvalues −47.40 ± 20.58i; full controllability/observability — match paper.

Unit-step transient metrics (published | replicated):

| System | Rise t_r (s) | Settling t_s (s) | Overshoot M_p |
|---|---|---|---|
| Open loop | 0.0573 | 0.057 | 0.0946 | 0.095 | 0% | ~0% |
| PI | 0.0105 | 0.010 | 0.111 | 0.111 | 36.9% | 36.9% |
| LQR (R=1 gains) | 0.0965 | 0.096 | 0.173 | 0.173 | 0% | 0% |

LQR implemented with the published R = 1.75 gives tr = 0.121 s, ts = 0.222 s,
M_p = 0% (form matches; numbers differ — see Deviations item 1).

Tracking 1000→1500 rpm (step at t = 0.5 s): both controllers reach the 1500 rpm
reference; steady armature voltage ≈ 191 V, armature current ≈ 1.4 A, consistent
in form with paper Fig. 6.

## Deviations / observations from the published method

1. **LQR R inconsistency (main finding).** Implementing the paper's stated
   Q = diag(0.1, 0.1, 1000), R = 1.75 gives K = [0.755, 2.142, −23.90], which does
   not reproduce the reported k = [0.9763, 2.712, 31.6228] or the Table II LQR row.
   Solving the same ARE with R = 1.0 reproduces both exactly
   (tr 0.096 s, ts 0.173 s, M_p = 0%). The paper's printed R therefore appears
   inconsistent with its reported results; we implement both and document this.
2. **PI closed-loop TF typo.** The printed H(s) (paper eq. (19)) implies
   Kp = 45, Ki = 0.1, inconsistent with the stated Kp = 7, Ki = 200. Implementing
   the stated gains reproduces the Table II PI row exactly (M_p = 36.9%).
3. **Symbolic TF notation errors.** Eq. (6) uses J_a/B_a and K_e·K_e; the physical
   state-space model (consistent with eq. (7)) is used instead.
4. **Unspecified simulation details.** Solver, step timing and rpm↔rad/s scaling
   are not given in the paper; choices are recorded here (rk-based `lsim`,
   t = 0–0.35 s step window, step at 0.5 s, ω in rad/s internally, plotted in rpm).
5. **No published code.** Full reconstruction from the paper; no external code used.

## Risks and next steps

- Exact Fig. 6 waveform matching is sensitive to unspecified Simulink details;
  we target same form + same order of magnitude and quantify residual error.
- Remaining: MATLAB/Simulink build (primary implementation per replication plan),
  full Fig. 6 pass, modest extension (load-torque step; q33 sensitivity at fixed R),
  draft results for the Week 8 checkpoint.
