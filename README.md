# From Observed Trade-Offs to Learned Behavior in Autonomous Driving

Code for the paper *From Observed Trade-Offs to Learned Behavior in Autonomous Driving* (Elayan, University of Nebraska–Lincoln).

Behavioral trade-offs among safety, efficiency, and interaction are extracted from naturalistic AV trajectories (TGSIM) and formalized as an empirical Pareto surface. A Soft Actor-Critic agent learns to modulate time headway and comfortable deceleration on top of a heterogeneous IDM baseline in SUMO, using proximity to that surface as its reward. The trained policy is then evaluated without retraining on five unseen Highway Capacity Manual facility types.

## Pipeline

| Step | Stage | Script | Status |
|---|---|---|---|
| 1 | Data download and processing | `scripts/01_prepare_data.py` | ✅ |
| 2 | Real-time behavioral metrics | `scripts/02a_gssm.py`, `scripts/02b_metrics.py` | ✅ |
| 3 | Pareto analysis and surface | `scripts/03_pareto.py` | ✅ |
| 4 | Domain adaptation (quantile mapping) | `scripts/04a_baseline_sim.py`, `scripts/04b_domain_adapt.py` | ✅ |
| 5 | SAC training | `scripts/05_train.py` | ✅ |
| 6 | Evaluation on the training network | `scripts/06_evaluate_training.py` | ✅ |
| 7 | Zero-shot evaluation on unseen networks | `scripts/07_evaluate_scenarios.py` | ✅ |

## Repository layout

```
├── configs/            YAML settings per stage
├── data/               raw and processed data (not tracked, see data/README.md)
├── models/             GSSM, Pareto surface, adapted surface, and trained SAC actor used in the paper
├── scripts/            numbered entry points, one per stage
├── src/avconsensus/    library code
│   ├── data/           coordinate transform, smoothing, kinematics
│   ├── metrics/        GSSM, PET, headway, gain, jerk, deceleration
│   ├── pareto/         rank normalization, penalty, Pareto front, convex hull
│   ├── adaptation.py   quantile mapping between TGSIM and simulation
│   ├── sim/            SUMO interface, traffic heterogeneity, online metrics
│   ├── rl/             SAC agent, Pareto reward, training episode
│   └── evaluation/     baseline and trained episodes, comparison report
├── outputs/            regenerated models, surfaces, and figures (not tracked)
└── sumo/               training and evaluation networks (see sumo/README.md)
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/mselayan/morl-sac-av-consensus.git
cd morl-sac-av-consensus
pip install -e .
```

This installs SUMO through `eclipse-sumo`. To use a container or an existing installation, see [`sumo/README.md`](sumo/README.md).

## Quickstart

1. Download the TGSIM data into `data/raw/` following [`data/README.md`](data/README.md).
2. Run the pipeline:

```bash
python scripts/01_prepare_data.py   # smooth trajectories, derive kinematics
python scripts/02a_gssm.py          # GSSM risk with the pretrained model
python scripts/02b_metrics.py       # PET, headway, gain, jerk, decel; unified table
python scripts/03_pareto.py         # empirical Pareto surface
python scripts/04a_baseline_sim.py  # baseline SUMO run (EAB-IDM, no RL)
python scripts/04b_domain_adapt.py  # quantile-mapped surface for training
python scripts/05_train.py          # parallel SAC training
python scripts/06_evaluate_training.py   # baseline vs trained, training network
python scripts/07_evaluate_scenarios.py  # zero-shot transfer, S1 to S5
```

Every step ships its output in `models/`, so any step can be run on its own. Regenerated outputs go to `outputs/` and never overwrite the shipped files.

## Behavioral metrics

Six metrics are computed per AV timestep from positions and velocities only.

| Dimension | Metric | Definition | Threshold |
|---|---|---|---|
| Safety | GSSM risk $M$ | Max over nearby agents of $\log_{10}(\ln 0.5 / \ln P(S > s^* \mid X))$ | 2.07 |
| Safety | PET | Min time between the AV leaving a 4 m × 4 m zone and another road user entering it | 2.25 s |
| Efficiency | Time headway | Bumper gap to nearest same-lane leader / ego speed | 4.0 s |
| Efficiency | String stability gain | RMS(follower accel) / RMS(AV accel), trailing 5 s | 1.0 |
| Interaction | \|Jerk\| | Time derivative of longitudinal acceleration | 2.5 m/s³ |
| Interaction | Deceleration | \|accel\| when braking | 2.0 m/s² |

**GSSM.** For each AV timestep, agents within $\mathrm{clip}(3v, 20, 150)$ m are paired with the AV and described by 18 kinematic features. A two-layer MLP (128, 64) predicts lognormal spacing parameters. The model is trained by negative log-likelihood, with AVs split into train, validation, and test sets. A 3-component GMM on positive $M$ gives the risk threshold. The pretrained model in `models/gssm/` reproduces the paper. Run `02a_gssm.py --train` to retrain; results go to `outputs/gssm/`.

Settings live in [`configs/metrics.yaml`](configs/metrics.yaml).

**Outputs** (in `data/processed/`):

| File | Content |
|---|---|
| `gssm_interactions.csv` | AV-agent pairs with features and spacing (cached) |
| `gssm_max_risk.csv` | Max GSSM risk per AV timestep |
| `AV_Unified_Metrics.csv` | One row per AV timestep: `id, time, type, speed, accel, jerk, abs_jerk, decel, maximum_risk, pet, headway, gain, dataset` |

## Pareto surface

Each metric is replaced by its percentile rank (1 = best), and pairs of ranks are averaged into Safety, Efficiency, and Interaction scores. Rows with fewer than two dimensions are dropped. A single missing dimension is imputed by distance-weighted kNN ($k = 10$) within each dataset.

A dimension is violated when any of its metrics crosses its threshold. All three scores are reduced by $0.1\,k^2$, where $k$ is the number of violated dimensions. Pareto-optimal points are the non-dominated observations scoring at least 0.5 in every dimension. A convex hull is fitted to them, and its facets facing $(1, 1, 1)$ form the frontier used as the training reward reference.

The surface used in the paper is shipped as `models/pareto/TGSIM_pareto_surface.joblib`: 14 Pareto points, 20 hull facets, 13 upper facets. `03_pareto.py` regenerates it into `outputs/pareto/` along with the scored composites and the surface figure. Settings live in [`configs/pareto.yaml`](configs/pareto.yaml).

## Domain adaptation

Simulated metric distributions differ from TGSIM. `04a_baseline_sim.py` runs the training network with EAB-modulated IDM AVs and no RL, and records the six metrics for every AV step. `04b_domain_adapt.py` builds an ECDF for each metric. During training, a simulated value is scored by its rank in that ECDF, which places it in the same coordinates as the Pareto surface. Each TGSIM threshold is translated through its TGSIM percentile to the simulation value at the same percentile.

| Metric | TGSIM threshold | TGSIM percentile | Simulation threshold |
|---|---|---|---|
| GSSM risk | 2.07 | 93.2 | 8.205 |
| PET (s) | 2.25 | 27.4 | 2.600 |
| Headway (s) | 4.00 | 64.0 | 6.336 |
| Gain | 1.00 | 57.5 | 1.035 |
| \|Jerk\| (m/s³) | 2.50 | 88.9 | 2.912 |
| Deceleration (m/s²) | 2.00 | 90.1 | 1.020 |

The adapted surface used in the paper is `models/adaptation/adapted_surface.joblib`.

**Simulation metrics.** Online metrics use the live SUMO state, with leader and follower taken from SUMO's lane topology:

- Headway uses SUMO's leader gap, which is measured from the ego front bumper plus minGap.
- The follower acceleration history for gain is stored per AV.
- Jerk is computed from a 3-point moving average of acceleration.
- PET takes, for each nearby agent, the first matching past AV position.
- GSSM returns 0 below 1 m/s and clips negative risk to 0.

## Training

Each AV follows IDM, and its time headway and minimum gap are scaled by an EAB multiplier $\eta(t)$ sampled per AV. At every step, the SAC actor observes a 10-dimensional state:

- GSSM risk, PET, speed, and gap to leader
- relative speed, leader acceleration, and local density
- $\eta$ and its own previous actions

It outputs two factors in $[0.3, 2.0]$, which are exponentially smoothed with weights 0.85 and 0.60 and applied as

$$T_t = 1.55\,\eta_t\,\bar\alpha_{\tau,t}, \qquad b_t = 1.02\,\bar\alpha_{b,t}.$$

The reward is

$$r_t = -d(\mathbf{x}_t', \mathcal{F})^{0.5} + 0.5 \min(S_t', E_t', I_t'),$$

where $\mathbf{x}'$ is the penalized $(S, E, I)$ point and $\mathcal{F}$ is the set of upper hull facets. When one dimension is missing, it holds its last value. No reward is computed while the AV is slower than 1 m/s.

Training runs 1,024 one-hour episodes in batches of 16 parallel SUMO instances. Each batch is followed by up to 1,000 gradient steps on minibatches of 256. Checkpoints and `sac_training_progress.csv` are written to `outputs/training/` after every batch, and the script resumes from the latest checkpoint. The trained actor used in the paper is `models/sac/sac_actor.pth`.

Settings live in [`configs/sim.yaml`](configs/sim.yaml), [`configs/adaptation.yaml`](configs/adaptation.yaml), and [`configs/train.yaml`](configs/train.yaml).

## Evaluation

Both evaluation scripts run each network twice under the same seed, demand, and HDV parameter draws. The first run is the baseline (EAB-IDM, no RL); the second applies the trained actor deterministically, with no weight updates and no surface adaptation. Every AV step at or above 1 m/s is logged with:

- the six raw metrics
- the $(S, E, I)$ scores
- the reward
- one violation flag per dimension

`summary.txt` compares the two runs:

- median dimension scores, with Mann-Whitney U tests and Cohen's $d$
- compliance rates and co-occurring violations, with two-proportion $z$-tests
- raw metric medians
- applied modulation factors
- results per route group

Route groups come from AV route ids, so naming routes by sub-scenario (`S1a`, `S1b`, ...) gives the sub-scenario breakdown.

| Script | Network | Output |
|---|---|---|
| `06_evaluate_training.py` | training network | `outputs/evaluation/training/` |
| `07_evaluate_scenarios.py [S1 ...]` | unseen networks in `sumo/scenarios/` | `outputs/evaluation/S*/`, `scenario_results.csv` |

| Scenario | Facility type | Sub-scenarios |
|---|---|---|
| S1 | Two-way stop-controlled intersection | protected left, right with pedestrian yield, through |
| S2 | Signalized intersection (multimodal) | protected/permitted left, right on green, through |
| S3 | Signalized arterial corridor | through, coordinated signals |
| S4 | Basic freeway segment | through |
| S5 | Freeway weaving segment | mainline, on-ramp merge, off-ramp diverge |

Use `--summary-only` to rebuild the reports from existing CSVs. Settings live in [`configs/eval.yaml`](configs/eval.yaml).

## Citation

```bibtex
@article{elayan_observed_tradeoffs,
  title  = {From Observed Trade-Offs to Learned Behavior in Autonomous Driving},
  author = {Elayan, Mohammad},
  year   = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE). TGSIM data are public domain and distributed by FHWA.
