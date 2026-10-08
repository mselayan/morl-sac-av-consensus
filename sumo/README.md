# SUMO networks

```
sumo/
├── training/        arterial network used for baseline and SAC training
└── scenarios/
    ├── S1/          two-way stop-controlled intersection
    ├── S2/          signalized intersection (multimodal)
    ├── S3/          signalized arterial corridor
    ├── S4/          basic freeway segment
    └── S5/          freeway weaving segment
```

Each folder holds one `.sumocfg` and every file it references. To see which files a configuration needs:

```bash
python scripts/sumo_files.py sumo/training/osm.sumocfg
```

Only the files listed as referenced are required. Paths inside the `.sumocfg` are resolved relative to it.

## Vehicle types

The code identifies vehicles by type id (see `configs/sim.yaml`):

| Setting | Default | Meaning |
|---|---|---|
| `av_prefix` | `av` | Types whose id starts with this are controlled AVs |
| `hdv_type` | `veh_passenger` | Type that receives sampled heterogeneous IDM parameters |

All vehicles must use `carFollowModel="IDM"`.

In the scenario networks, name AV routes by sub-scenario (for example `S1a`, `S1b`, `S1c`). Vehicle ids then look like `S1a.0`, and the evaluation report groups results by the part before the final dot.

Each scenario folder must contain an `osm.sumocfg`. Different names or folders can be set in `configs/eval.yaml`.

## Running SUMO

Install SUMO and TraCI with pip:

```bash
pip install eclipse-sumo traci
```

This puts a `sumo` binary on your path, which is the default in `configs/sim.yaml`. To use a container instead, for example on an HPC cluster, set the command:

```yaml
sumo:
  cmd: [singularity, exec, sumo.sif, /usr/bin/sumo]
```
