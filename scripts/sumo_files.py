"""List the files a SUMO configuration needs and whether each exists.

Use it to see which files to copy next to a .sumocfg: the network, route
or trip files, and additional files (vehicle types, detectors, signals).

Usage:
    python scripts/sumo_files.py path/to/osm.sumocfg
"""

import argparse
from pathlib import Path

from avconsensus.sim.sumo import referenced_files


def main(sumocfg: str) -> None:
    cfg = Path(sumocfg)
    if not cfg.exists():
        raise SystemExit(f"{cfg} not found")
    files = referenced_files(cfg)
    print(f"{cfg} references {len(files)} files:")
    for f in files:
        print(f"  [{'ok' if f.exists() else 'MISSING'}] {f}")
    if all(f.exists() for f in files):
        print("all present")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sumocfg")
    main(p.parse_args().sumocfg)
