#!/usr/bin/env python3
"""Download the selected *original TUM RGB-D* sequence; safely extract the .tgz.

Large public dataset is never committed to this source repository. No automatic
network call occurs at import; choose an explicit sequence and directory.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import tarfile
from urllib.request import Request, urlopen

SEQUENCES = {
    "freiburg1_xyz": "freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    "freiburg1_desk": "freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "freiburg1_rpy": "freiburg1/rgbd_dataset_freiburg1_rpy.tgz",
}
BASE = "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/"

def safe_extract(source: Path, dest: Path):
    """Reject links/devices/traversal, then extract regular files and dirs only."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(source, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Refusing unsafe tar archive entry: {member.name}")
        for member in members:
            target = root / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", choices=sorted(SEQUENCES), default="freiburg1_xyz")
    p.add_argument("--destination", type=Path, default=Path("data"))
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args(argv)
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / Path(SEQUENCES[args.sequence]).name
    url = BASE + SEQUENCES[args.sequence]
    print("Downloading the original TUM RGB-D archive from:", url, flush=True)
    # Require HTTPS; don't allow a third-party CLI URL substitution.
    with urlopen(Request(url, headers={"User-Agent": "rgbd-lab-research/1.0"}), timeout=80) as source:
        if not source.geturl().startswith("https://"):
            raise ValueError("Refusing non-HTTPS download")
        with archive.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    print("Extracting", archive, flush=True)
    safe_extract(archive, args.destination)
    if not args.keep_archive:
        archive.unlink()
    print("Done:", args.destination / f"rgbd_dataset_{args.sequence}")

if __name__ == "__main__":
    main()
