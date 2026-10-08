"""Starting SUMO and checking its input files."""

import xml.etree.ElementTree as ET
from pathlib import Path

import traci

INPUT_KEYS = ("net-file", "route-files", "additional-files")


def referenced_files(sumocfg: str | Path) -> list[Path]:
    """Input files listed in a .sumocfg, resolved relative to it."""
    sumocfg = Path(sumocfg)
    root = ET.parse(sumocfg).getroot()
    files = []
    for key in INPUT_KEYS:
        for el in root.iter(key):
            for name in el.get("value", "").split(","):
                if name.strip():
                    files.append(sumocfg.parent / name.strip())
    return files


def check_config(sumocfg: str | Path) -> None:
    """Raise if the .sumocfg or any file it references is missing."""
    sumocfg = Path(sumocfg)
    if not sumocfg.exists():
        raise FileNotFoundError(f"{sumocfg} not found. See sumo/README.md.")
    missing = [str(f) for f in referenced_files(sumocfg) if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Files referenced by {sumocfg} are missing: {missing}")


def start(cfg: dict, port: int | None = None) -> None:
    """Launch SUMO with the sim config and connect TraCI."""
    s = cfg["sumo"]
    check_config(s["config"])
    cmd = [*s["cmd"], "-c", str(Path(s["config"]).resolve()),
           "--step-length", str(s["step_length"]), *map(str, s["extra_args"])]
    traci.start(cmd, port=port, numRetries=50)
