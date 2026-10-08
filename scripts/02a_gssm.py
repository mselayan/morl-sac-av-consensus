"""Step 2a: GSSM safety risk for every AV timestep.

Builds AV-agent interaction features from the processed TGSIM data,
scores each interaction with the GSSM, and keeps the maximum risk per
AV timestep. Uses the pretrained model in models/gssm by default.

Usage:
    python scripts/02a_gssm.py            # score with pretrained model
    python scripts/02a_gssm.py --train    # retrain, recalibrate, then score
    python scripts/02a_gssm.py --rebuild  # recompute cached interactions
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch
import yaml

from avconsensus.metrics import prepare_frame
from avconsensus.metrics import gssm
from avconsensus.plotting import plot_gmm_threshold


def load_interactions(cfg, rebuild: bool) -> pd.DataFrame:
    """Load cached interactions or build them from the processed data."""
    cache = Path(cfg["processed_dir"]) / "gssm_interactions.csv"
    if cache.exists() and not rebuild:
        print(f"Loading cached {cache}")
        return pd.read_csv(cache)

    g = cfg["gssm"]
    parts = []
    for name, file in cfg["datasets"].items():
        print(f"[{name}] building interactions")
        df = prepare_frame(pd.read_csv(Path(cfg["processed_dir"]) / file),
                           cfg["neighbors"]["default_length"])
        parts.append(gssm.build_interactions(df, name, g))
    df = gssm.filter_interactions(pd.concat(parts, ignore_index=True), **g["filter"])
    df.to_csv(cache, index=False)
    print(f"  {len(df):,} interactions saved to {cache}")
    return df


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output_dir"])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_interactions(cfg, args.rebuild)

    if args.train:
        print("Training GSSM")
        model, scaler, _ = gssm.train_gssm(df, cfg["gssm"], cfg["seed"])
        M = gssm.score(df, model, scaler)
        threshold, gmm = gssm.fit_threshold(M, cfg["gssm"]["gmm_components"], cfg["seed"])

        model_dir = out_dir / "gssm"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "gssm_model.pth")
        joblib.dump({"scaler": scaler, "feature_names": gssm.FEATURES,
                     "input_dim": len(gssm.FEATURES)}, model_dir / "gssm_preprocessor.pkl")
        joblib.dump({"threshold": threshold, "gmm": gmm}, model_dir / "gssm_threshold.joblib")
        print(f"  saved retrained model to {model_dir}")
    else:
        model_dir = Path(cfg["model_dir"])
        print(f"Scoring with pretrained GSSM from {model_dir}")
        model = gssm.GSSM()
        model.load_state_dict(torch.load(model_dir / "gssm_model.pth", weights_only=True))
        scaler = joblib.load(model_dir / "gssm_preprocessor.pkl")["scaler"]
        calib = joblib.load(model_dir / "gssm_threshold.joblib")
        threshold, gmm = calib["threshold"], calib["gmm"]
        M = gssm.score(df, model, scaler)

    risk = gssm.max_risk_per_timestep(df, M)
    dst = Path(cfg["processed_dir"]) / "gssm_max_risk.csv"
    risk.to_csv(dst, index=False)

    fig = out_dir / "figures" / "gmm_risk_threshold.png"
    plot_gmm_threshold(M, gmm, threshold, fig)

    share = (risk["maximum_risk"] > threshold).mean()
    print(f"  GMM threshold {threshold:.3f}: {share:.1%} of AV timesteps above")
    print(f"  saved {dst} ({len(risk):,} rows) and {fig}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/metrics.yaml")
    p.add_argument("--train", action="store_true", help="retrain GSSM and recalibrate threshold")
    p.add_argument("--rebuild", action="store_true", help="recompute interaction features")
    main(p.parse_args())
