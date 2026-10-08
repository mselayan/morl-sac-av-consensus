# Data

This study uses two public trajectory datasets from the FHWA Third Generation Simulation (TGSIM) project. They are not stored in this repository because of their size.

| Dataset | Setting | AVs | Size | Source |
|---|---|---|---|---|
| Foggy Bottom | Urban, 4 intersections, Washington, D.C. | SAE L3 | ~350 MB | [data.gov](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories) |
| I-395 | Freeway weaving section, Washington, D.C. | SAE L2 | ~232 MB | [data.gov](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-395-trajectories) |

Both are sampled at 10 Hz and released in the public domain.

## Download

1. Open each data.gov page above.
2. Under **Resources**, download the **CSV** option.
3. Place both files in `data/raw/` with these names:

```
data/raw/
├── Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv
└── Third_Generation_Simulation_Data__TGSIM__I-395_Trajectories.csv
```

If your downloaded file names differ (for example, a date suffix), either rename them or edit `raw_file` in `configs/data.yaml`.

Direct CSV links:

```bash
curl -L -o data/raw/Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv \
  "https://data.transportation.gov/api/v3/views/brzy-6zfh/export.csv?accessType=DOWNLOAD"
curl -L -o data/raw/Third_Generation_Simulation_Data__TGSIM__I-395_Trajectories.csv \
  "https://data.transportation.gov/api/v3/views/97n2-kuqi/export.csv?accessType=DOWNLOAD"
```

## Raw columns used

| Column | Meaning |
|---|---|
| `id` | Agent ID |
| `time` | Time (s) |
| `xloc_kf`, `yloc_kf` | Position (m), Kalman filtered |
| `speed_kf_y`, `acceleration_kf_y` | y components of speed and acceleration |
| `type_most_common` | Road user type code |
| `lane_kf` | Lane or region ID |
| `length_smoothed`, `width_smoothed` | Agent dimensions (m) |

Type codes:

| Code | Foggy Bottom | I-395 |
|---|---|---|
| 0 | pedestrian | |
| 1 | bicycle | hdv |
| 2 | scooter | truck |
| 3 | hdv | bus |
| 4 | av | av |
| 5 | motorcycle | |
| 6 | bus | |
| 7 | truck | |

## Processing

```bash
python scripts/01_prepare_data.py
```

Steps:

1. **Flip y axis.** TGSIM uses an image origin (top-left, y down). y is mirrored as `H * c - y`, where `H` is the image height spanned by y in pixels and `c` is the pixel-to-meter factor (0.0186613838586 for Foggy Bottom, 0.3 for I-395).
2. **Drop invalid agents.** Trajectories shorter than 5 samples or with missing positions are removed.
3. **Smooth positions.** Gaussian filter, σ = 1.0 sample.
4. **Derive kinematics.** Velocity and acceleration via `np.gradient` on the time array. Longitudinal acceleration is the projection of acceleration on heading. Below 0.1 m/s it falls back to the derivative of speed.

## Output

`data/processed/TGSIM_Foggy_Bottom_smoothed.csv` and `data/processed/TGSIM_I-395_smoothed.csv`, one row per agent-timestep:

| Column | Unit | Description |
|---|---|---|
| `id` | | Agent ID |
| `time` | s | Timestamp |
| `x`, `y` | m | Smoothed position |
| `vx`, `vy` | m/s | Velocity components |
| `ax`, `ay` | m/s² | Acceleration components |
| `speed` | m/s | Scalar speed |
| `accel` | m/s² | Signed longitudinal acceleration |
| `type` | | Road user type |
| `dataset` | | `FB` or `I-395` |
| `lane` | | Lane or region ID |
| `length`, `width` | m | Agent dimensions |

These files are the input to metric computation (Step 2).

## Citation

Federal Highway Administration. *Third Generation Simulation Data (TGSIM)*. U.S. Department of Transportation, 2024. Project report: https://rosap.ntl.bts.gov/view/dot/74647
